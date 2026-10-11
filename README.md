# Quran RAG backend

Python 3.11+, FastAPI, LangGraph, SQLite, Qdrant, BM25, Cloudflare multilingual embeddings and Hugging Face, Cloudflare, Gemini or Groq text generation.

## Setup (PowerShell)

Set CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID in .env.
For Hugging Face text generation, set TEXT_GENERATION_PROVIDER=huggingface, HF_TOKEN and HF_MODEL=openai/gpt-oss-120b:cerebras. LangChain ChatOpenAI uses the Hugging Face inference router, with native JSON-schema structured output. HF_REVIEW_MODEL optionally selects a separate evidence-review model. Inference credits and a token with Inference Providers permission are required. HTTP 402 reports insufficient credits; changing models cannot replenish the shared account balance. Embeddings continue to use Cloudflare BGE-M3 and the existing Qdrant index.
For Cloudflare text generation, set TEXT_GENERATION_PROVIDER=cloudflare and CLOUDFLARE_TEXT_MODEL=@cf/openai/gpt-oss-120b. The same Workers AI credentials serve embeddings and text generation. LangChain ChatOpenAI uses Cloudflare's OpenAI-compatible endpoint. GPT-OSS structured calls use native ChatPromptTemplate and PydanticOutputParser, with one bounded correction that receives the actual schema-validation error; Cloudflare does not consistently force tool calls and does not document GPT-OSS JSON mode. Draft generation is non-streaming upstream, and validated answer text is delivered through the existing SSE endpoint.
For Gemini, set TEXT_GENERATION_PROVIDER=gemini, GEMINI_API_KEY and GEMINI_MODEL. GEMINI_REVIEW_MODEL and GEMINI_REVIEW_FALLBACK_MODELS configure evidence review separately. Gemini uses native JSON-schema structured output, bounded rate-limit retries and configured model fallbacks; embeddings remain on Cloudflare.
For Groq, set TEXT_GENERATION_PROVIDER=groq and GROQ_API_KEY.
Model access and free quotas depend on the provider account; listing a model does not guarantee generation quota.
Use EMBEDDING_MODEL=@cf/baai/bge-m3.

```powershell
uv sync --locked
.venv/Scripts/python.exe -m quran_rag.ingest --batch-size 32
.venv/Scripts/python.exe -m uvicorn quran_rag.api:app --host 127.0.0.1 --port 8000
```

Embeddings run on Cloudflare Workers AI. No local weights, tokenizer, PyTorch
inference or embedding prewarming are needed. Both Arabic and English texts are
embedded with the same API used for runtime queries. A complete index contains
6,236 vectors per language. Cloudflare collections have separate fingerprints
from the previous local int8 collections. Stop the API before local Qdrant ingestion.
Indexing resumes from saved batches after interruption. API quotas, network errors
and provider outages can still interrupt requests; transient errors receive bounded
retries. Credentials stay in the backend .env, never in the React application.

API documentation: http://127.0.0.1:8000/docs

## Endpoints

- GET `/health`: corpus count, per-language vector counts, and bilingual index readiness.
- GET `/verses/2/44`: canonical Arabic, English and metadata.
- GET `/search/word?word=sun`: exhaustive whole-word English matches for listing verses.
- GET `/search/word/count?word=مغفرة`: Arabic-only frequency count, matching verse count, canonical records and matching method.
- POST `/chat`: JSON answer, canonical verse records and notices.
- POST `/chat/stream`: SSE status events, validated answer text, final result and done.

Example:

```powershell
$body = @{question='What does the Quran say about patience?'; thread_id='my-chat'} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/chat -Method Post -ContentType application/json -Body $body
```

Reuse a thread_id to continue the conversation. Use a new thread_id for a new conversation. SSE event types are `status`, `answer_token`, `result`, `done`, and `error`. Thematic answers, explanations and summaries may faithfully paraphrase the stored translation, with citations for every factual paragraph. Canonical Arabic and English records remain unchanged and are returned separately. Evidence review reads both languages, explicitly compares actors, recipients, conditions and amounts, and checks exact English evidence spans. TEXT_REVIEW_REASONING_EFFORT configures GPT-OSS review reasoning; TEXT_ANSWER_REASONING_EFFORT configures final-answer reasoning while retrieval planning remains lightweight. At most two evidence-based answer repairs are permitted; malformed reviewer evidence has its own bounded correction. Generated answers use a typed, bounded set of short statements and attached verse IDs. Draft text stays private while local citation/quotation checks and a source-grounded model review run. Review evidence must match exact spans in each claim's own cited verses. Invalid review evidence is corrected separately; unsupported answer claims receive a bounded repair. Unverified drafts are withheld. Status events stream during execution, followed by the checked final answer and result. These checks reduce errors but do not establish perfect accuracy; live answers still require evaluation.

## Storage and retrieval

- `data/quran-uthmani.txt`, `data/en.sahih.txt`, `data/quran-data.xml`: unmodified Tanzil sources.
- `data/manifest.json`: download URLs, timestamps and SHA-256 hashes.
- `data/quran.sqlite`: canonical metadata/text, one row per ayah.
- `data/qdrant/`: persistent local vectors (or configure QDRANT_URL).
- `data/checkpoints.sqlite`: separate durable LangGraph state.

The custom loader verifies all expected chapter/verse keys and pairs translations by key. The loader emits one complete ayah per document, so no splitter is needed. Semantic retrieval combines English BM25, English dense, and Arabic dense candidates using reciprocal rank fusion, deduplicates by ayah ID, then fetches both canonical texts from SQLite. Queries are embedded once and searched against both language collections. Literal search never uses top-k retrieval. Surah summaries read the entire chapter, select representative verse IDs from each section of long chapters, and explain their supported teachings. Short chapters are covered in full. Intermediate model prose is never used as source evidence. All 114 surahs resolve from the complete Tanzil catalog using Arabic names, transliterations, English names, or chapter numbers. Ambiguous shared names prompt clarification. Open questions use a general planner without example-specific aliases or narrative ranges. For narratives, hybrid retrieval expands the requested actors into English search terms, filters candidate seeds for direct relevance, and selects contiguous ranges from wider canonical context. Provider errors propagate rather than promoting raw, potentially unrelated hits into answer evidence. Coverage of every occurrence is not guaranteed. Answers use Quran text and its stored translation only. Historical occasions of a surah's revelation receive a statement that this information is not stated in the Quran; no external commentary is fetched. Retrieval candidates are filtered for relevance to the requested actors and actions before generation. Missing requested details cause abstention rather than supplementation from model memory.

## Verification and model execution

```powershell
.venv/Scripts/python.exe quran-rag-frontend/verify_threads.py
node quran-rag-frontend/verify_stream.mjs
npm --prefix quran-rag-frontend run build
```

These checks verify persistent thread history, SSE transport and the React build.
They do not establish Quran-answer accuracy or replace live retrieval evaluation.

Groq requests use LangChain's native `with_structured_output`: JSON schema for
GPT-OSS and tool calling for other configured models. Pydantic validates the
returned schema. Native `with_retry` retries parsing failures twice at most;
Groq's client handles its configured request retries. Native `with_fallbacks`
switches models on quota errors, without hiding unrelated failures or switching
after a stream has started emitting. `provider.py` only configures these models
and runnables; no custom JSON extraction or fallback engine is used.

The backend starts with `GROQ_MODEL`, then uses `GROQ_FALLBACK_MODELS` when quota
is exhausted. The default alternatives are GPT-OSS 20B and Qwen 3.8 27B.
`GROQ_REVIEW_MODEL` and `GROQ_REVIEW_FALLBACK_MODELS` configure evidence review.
`GROQ_QWEN_MAX_TOKENS` caps Qwen's default response budget (800 by default),
so a request does not exceed the current free-tier output-token allowance.
Quota errors fall back immediately through all configured models, including
earlier models whose quotas may have recovered since a previous fallback.
Thematic and narrative search use up to three focused query variants, each combining BM25 and bilingual dense retrieval, then fuse their rankings by reciprocal rank fusion.
It inspects the bounded pool of up to 80 ranked candidates
before accepting an answerable topic, retains relevant evidence across batches,
and selects up to eight distinct supporting verses. A separate reviewer checks
the primary verses together in one structured response, with one verdict per
verse. Exact references and literal searches bypass this semantic pipeline.
Cloudflare reranking is evaluated separately; it is not enabled by default.
Safeguard and ALLaM are excluded from factual evidence review. Quran-specific
routing, Arabic morphology, bilingual rank fusion and canonical sentence checks
remain application logic.

An ayah explanation fetches the selected ayah and one neighboring ayah on either side, clipped to the same chapter. This also applies when the ayah is selected by semantic retrieval. The answer focuses on the requested ayahs and uses neighbors as context. Numeric ranges accept `through`, repeated chapter references, and ranges crossing chapters. Chapter requests support the first or last several verses.

Cloudflare embeds each hybrid query once; both dense collections use that vector.

## Scope

This is a local development backend. Run one worker with local Qdrant; a lock serializes graph requests to protect its single-process persistence resources. No authentication or tenant isolation is included. Keep it bound to localhost. For deployment add authentication, authorize thread ownership, use a Qdrant server and a production checkpointer. Semantic search returns relevant candidates, not a proven exhaustive list. No fine-tuning is included. Arabic word counts use the Quranic Arabic Corpus v0.4 morphology annotations.

Read [Tanzil attribution and translation usage terms](data/ATTRIBUTION.md). Explanations and summaries stay within the supplied Quran translation; they do not provide scholarly tafsir. Never commit .env or checkpoint databases, which contain private chat history.

English indexing resumes from existing vectors; adding Arabic does not discard English progress. Language-specific fingerprints update only changed texts. Ingestion alternates English and Arabic batches and logs cumulative counts for each. Semantic retrieval becomes ready when both collections cover the corpus. Stop the local indexer before opening the API because local Qdrant permits one process at a time.

Word-frequency questions ("how many times does the word … appear?") use original Arabic only. Complete English words or expressions are resolved by the language model to one Arabic dictionary lemma, which is disclosed in the answer. Multiword English expressions do not trigger clarification merely because of their length. Infinitive expressions resolve to verb lemmas; counts include their conjugations and attached clitics, without combining other derived verbs of the same root. Clarification is reserved for genuinely ambiguous meanings or expressions that have no single-lemma Arabic equivalent. The original expression is returned in `word_count.requested_expression`. Search normalization equates the corpus hamza-plus-alif spelling with alif madda. When a translated dictionary headword has no matching corpus lemma, its root can narrow a list of attested lemma candidates; the selected lemma is validated against that list before counting. Root candidates are never pooled into a root count. SQLite deterministically counts the [Quranic Arabic Corpus v0.4](https://corpus.quran.com/) annotations, including attached articles/clitics and inflections of that lemma, excluding other lemmas sharing its root. Occurrences and distinct verses are returned separately in `word_count`. No count is estimated by the LLM or vector retrieval. Explicit "surface" or "written token" requests (and `/search/word/count?word=…&mode=surface`) retain exact written-token matching. Canonical Arabic remains unchanged.

The morphology source is preserved verbatim at `data/morphology/quranic-corpus-morphology-0.4.txt`, including Kais Dukes' copyright, GNU license and Tanzil notices. Obtained from the CLTK mirror of the original corpus download; derived SQLite rows are generated automatically. The source terms and attribution are included in the file. Check [the original download page](https://corpus.quran.com/download/) for updates.

Tanzil opening formulas prepended to ayah 1 are separated into `opening_header` metadata, preserving each source component verbatim. Only numbered ayah text is embedded or counted. Literal results return unnumbered openings separately as `matched_headers`. Arabic phrase search recognizes Uthmani spelling and initial clitics using corpus annotations. The API has no external-commentary source-document field.

Qwen requests use `reasoning_effort="none"` so hidden thinking cannot consume the small output allowance before the JSON result. This is the documented instruct mode in [Groq's Qwen 3.8 documentation](https://console.groq.com/docs/model/qwen/qwen3.8-27b). GPT-OSS remains the preferred model; fallback classification and canonical sentence checks still apply.

Cloudflare text generation and embeddings share the account's Workers AI allowance. API `/chat` returns HTTP 429 with a quota message when text generation is throttled. No automatic paid upgrade is performed. `RAG_SOURCE_MAX_CHARS` controls the canonical source budget; GPT-OSS has a larger context, so the configured budget avoids unnecessary section reduction. Generated story/topic responses return cited canonical verse records, rather than every unused retrieval candidate.

Semantic searches retain the main expanded query alongside up to two alternative queries. `CLOUDFLARE_REVIEW_MODEL` can select a different model from the generator for independent evidence review. Draft reviews identify source actors, recipients and exemption conditions before comparing each assertion with its own cited text, and copy separate exact evidence spans. Malformed review evidence receives at most two bounded corrections of the review, rather than triggering an unnecessary rewrite of the answer. Genuine unsupported draft claims still trigger a bounded answer repair; unverified answers are withheld.
