"""Configure the selected LangChain text-generation provider and fallbacks."""
import os

from groq import BadRequestError, RateLimitError
from google.genai.errors import APIError as GoogleAPIError
from langchain_core.exceptions import OutputParserException, ModelRateLimitError
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai._common import GoogleGenerativeAIError
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APIStatusError
from pydantic import ValidationError


class ModelPool:
    def __init__(self, models=None, factory=None, reasoning_effort=None):
        self.provider = os.getenv("TEXT_GENERATION_PROVIDER", "groq").strip().casefold()
        if self.provider == "huggingface":
            primary = os.getenv("HF_MODEL", "openai/gpt-oss-120b:cerebras")
            fallback = os.getenv("HF_FALLBACK_MODELS", "").split(",")
        elif self.provider == "cloudflare":
            primary = os.getenv("CLOUDFLARE_TEXT_MODEL", "@cf/openai/gpt-oss-120b")
            fallback = os.getenv("CLOUDFLARE_TEXT_FALLBACK_MODELS", "").split(",")
        elif self.provider == "gemini":
            primary = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
            fallback = os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash,gemini-3.5-flash-lite").split(",")
        elif self.provider == "groq":
            primary = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
            fallback = os.getenv("GROQ_FALLBACK_MODELS", "openai/gpt-oss-20b,qwen/qwen3.8-27b").split(",")
        else:
            raise ValueError("TEXT_GENERATION_PROVIDER must be 'huggingface', 'cloudflare', 'gemini' or 'groq'")
        self.models = list(dict.fromkeys(name.strip() for name in (models or [primary, *fallback]) if name.strip()))
        if not self.models:
            raise ValueError("At least one text-generation model must be configured")
        self.index = 0
        self.clients = {}
        self.reasoning_effort = reasoning_effort or 'low'
        self.factory = factory or self._factory

    @property
    def model_name(self):
        return self.models[self.index]

    def _factory(self, name):
        if self.provider == "huggingface":
            token = os.getenv("HF_TOKEN", "")
            if not token:
                raise ValueError("Set HF_TOKEN in .env")
            return ChatOpenAI(
                model=name, api_key=token,
                base_url="https://router.huggingface.co/v1",
                temperature=0, max_tokens=int(os.getenv("HF_MAX_TOKENS", "4096")),
                timeout=float(os.getenv("HF_TIMEOUT_SECONDS", "60")), max_retries=0,
                use_responses_api=False, stream_usage=False,
                **({'reasoning_effort': self.reasoning_effort} if name.startswith('openai/gpt-oss-') else {}),
            )
        if self.provider == "cloudflare":
            account = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
            token = os.getenv("CLOUDFLARE_API_TOKEN", "")
            if not account or not token:
                raise ValueError("Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in .env")
            return ChatOpenAI(
                model=name, api_key=token,
                base_url=f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1",
                temperature=0, max_tokens=int(os.getenv("CLOUDFLARE_TEXT_MAX_TOKENS", "4096")),
                timeout=float(os.getenv("CLOUDFLARE_TEXT_TIMEOUT_SECONDS", "60")), max_retries=0,
                use_responses_api=False, stream_usage=False, disable_streaming=True,
                **({'reasoning_effort':self.reasoning_effort} if name.startswith('@cf/openai/gpt-oss-') else {}),
            )
        if self.provider == "gemini":
            return ChatGoogleGenerativeAI(
                model=name,
                temperature=0,
                max_output_tokens=int(os.getenv("GEMINI_MAX_TOKENS", "2000")),
                timeout=float(os.getenv("GEMINI_TIMEOUT_SECONDS", "45")),
                max_retries=0,
                **({'thinking_config': {'thinking_level': self.reasoning_effort.upper()}} if name.startswith('gemini-3') else {}),
            )
        options = {"reasoning_format": "hidden", "reasoning_effort": "none" if name.startswith("qwen/") else self.reasoning_effort} if name.startswith(("openai/gpt-oss-", "qwen/")) else {}
        token_budget = int(os.getenv("GROQ_MAX_TOKENS", "2000"))
        if name.startswith("qwen/"):
            token_budget = min(token_budget, int(os.getenv("GROQ_QWEN_MAX_TOKENS", "800")))
        return ChatGroq(
            model=name,
            temperature=0,
            **options,
            max_tokens=token_budget,
            timeout=float(os.getenv("GROQ_TIMEOUT_SECONDS", "30")),
            max_retries=0,
        )

    def _runnable(self, options=None, schema=None):
        targets = []
        order = [self.index, *(index for index in range(len(self.models)) if index != self.index)]
        for index in order:
            name = self.models[index]
            if name not in self.clients:
                self.clients[name] = self.factory(name)
            client = self.clients[name]
            if schema and options:
                client = client.model_copy(update=options)
            if schema:
                if self.provider == "cloudflare":
                    parser = PydanticOutputParser(pydantic_object=schema)
                    prompt = ChatPromptTemplate.from_messages([
                        ('system', 'Return ONLY one valid JSON object, with no explanation or markdown. Follow the schema exactly. {format_instructions}'),
                        MessagesPlaceholder('messages'),
                    ]).partial(format_instructions=parser.get_format_instructions())
                    if name.startswith('@cf/openai/gpt-oss-'):
                        # Workers AI does not consistently force a tool call for
                        # GPT-OSS, and JSON mode is not documented for it. Native
                        # LangChain prompting + Pydantic parsing is the portable
                        # structured-output path; no custom JSON extraction.
                        target = prompt | client | parser
                        def repair_format(messages, config, chain=target):
                            try:
                                return chain.invoke(messages, config=config)
                            except (OutputParserException, ValidationError) as error:
                                original = messages['messages'] if isinstance(messages, dict) else messages
                                corrected = [*original, HumanMessage(content=
                                    'The previous JSON failed schema validation. Correct the specific errors below '
                                    'while preserving the original task and source evidence. Use fewer claims '
                                    'and the smallest sufficient citation set rather than omitting required support. '
                                    'Return only the corrected JSON object. Validation error: ' + str(error))]
                                return chain.invoke(corrected, config=config)
                        target = RunnableLambda(repair_format)
                    else:
                        target = prompt | client.with_structured_output(schema, method='json_schema', strict=False)
                elif self.provider == "huggingface":
                    target = client.with_structured_output(schema, method="json_schema", strict=False)
                elif self.provider == "gemini":
                    target = client.with_structured_output(schema, method="json_schema")
                else:
                    method = "json_schema" if name in ("openai/gpt-oss-120b", "openai/gpt-oss-20b") else "function_calling"
                    target = client.with_structured_output(schema, method=method)
                if not (self.provider == 'cloudflare' and name.startswith('@cf/openai/gpt-oss-')):
                    target = target.with_retry(
                        retry_if_exception_type=(OutputParserException, ValidationError),
                        stop_after_attempt=2,
                    )
            else:
                target = client.bind(**(options or {}))
            if self.provider == 'gemini':
                target = target.with_retry(
                    retry_if_exception_type=(ModelRateLimitError,),
                    exponential_jitter_params={'initial': 30, 'max': 45, 'jitter': 1},
                    stop_after_attempt=2,
                )
            if self.provider == 'groq':
                # A multi-step graph can exceed the free tier's minute bucket
                # even when every individual request fits. Pace bounded retries
                # instead of immediately exhausting another model's bucket.
                target = target.with_retry(
                    retry_if_exception_type=(RateLimitError,),
                    exponential_jitter_params={'initial': 30, 'max': 45, 'jitter': 1},
                    stop_after_attempt=3,
                )

            def remember(messages, selected=index):
                self.index = selected
                return messages

            targets.append(RunnableLambda(remember) | target)
        if not targets:
            raise ValueError("No qualified model remains for this request")
        provider_errors = (RateLimitError, BadRequestError, GoogleGenerativeAIError, GoogleAPIError, APIStatusError, APIConnectionError)
        recoverable = provider_errors + (OutputParserException, ValidationError) if schema else provider_errors
        return targets[0].with_fallbacks(targets[1:], exceptions_to_handle=recoverable)

    def invoke(self, messages):
        return self._runnable().invoke(messages)

    def stream(self, messages):
        return self._runnable().stream(messages)

    def bind(self, **kwargs):
        if self.provider == "gemini" and "max_tokens" in kwargs:
            kwargs["max_output_tokens"] = kwargs.pop("max_tokens")
        return self._runnable(options=kwargs)

    def with_structured_output(self, schema, **options):
        return self._runnable(options=options, schema=schema)
