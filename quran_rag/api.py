import json
import logging
import os
import threading
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from .config import CHECKPOINT_DB
from .store import VerseStore
from .retriever import HybridRetriever
from .graph import build_graph
from .provider import ModelPool

lock = threading.Lock()
logger = logging.getLogger(__name__)

def request_error(exc):
    """Report provider throttling without exposing raw credential-bearing errors."""
    status = getattr(exc, 'status_code', None) or getattr(exc, 'code', None)
    if status == 402:
        return 402, 'Text-generation provider has insufficient inference credits. Add credits to the provider account to continue.'
    if status == 429:
        return 429, 'Text-generation provider quota exceeded. Retry after the provider quota resets.'
    return 503, 'Request failed. Check text-generation provider availability, credentials, and vector index readiness.'

@asynccontextmanager
async def lifespan(app):
    store = VerseStore()
    retriever = HybridRetriever(store)
    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as saver:
        saver.setup()
        llm = ModelPool()
        app.state.store = store
        app.state.retriever = retriever
        provider = os.getenv('TEXT_GENERATION_PROVIDER', 'groq').strip().casefold()
        if provider == 'huggingface':
            review_model = os.getenv('HF_REVIEW_MODEL') or os.getenv('HF_MODEL', 'openai/gpt-oss-120b:cerebras')
            review_fallback = os.getenv('HF_REVIEW_FALLBACK_MODELS', os.getenv('HF_FALLBACK_MODELS', '')).split(',')
        elif provider == 'cloudflare':
            review_model = os.getenv('CLOUDFLARE_REVIEW_MODEL') or os.getenv('CLOUDFLARE_TEXT_MODEL', '@cf/openai/gpt-oss-120b')
            review_fallback = os.getenv('CLOUDFLARE_TEXT_FALLBACK_MODELS', '').split(',')
        elif provider == 'gemini':
            review_model = os.getenv('GEMINI_REVIEW_MODEL') or os.getenv('GEMINI_MODEL', 'gemini-3.8-flash')
            review_fallback = os.getenv('GEMINI_REVIEW_FALLBACK_MODELS', 'gemini-3.5-flash,gemini-2.5-flash-lite').split(',')
        else:
            review_model = os.getenv('GROQ_REVIEW_MODEL') or os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b')
            review_fallback = os.getenv('GROQ_REVIEW_FALLBACK_MODELS', 'openai/gpt-oss-20b,openai/gpt-oss-120b,qwen/qwen3.8-27b').split(',')
        reviewer = ModelPool(models=list(dict.fromkeys([
            review_model,
            *review_fallback,
        ])), reasoning_effort=os.getenv('TEXT_REVIEW_REASONING_EFFORT', 'medium'))
        app.state.graph = build_graph(store, retriever, llm, saver, reviewer=reviewer)
        try:
            yield
        finally:
            retriever.close()

app = FastAPI(title="Quran RAG", lifespan=lifespan)

def thread_config(key):
    return {'configurable': {'thread_id': key}}

def thread_title(values):
    title = values.get('chat_title')
    if title and title != 'New conversation':
        return title
    first = next((m.content for m in values.get('messages', []) if m.type == 'human'), '')
    return ' '.join(str(first).split())[:65] or 'New conversation'

def unique_threads():
    """List each persisted LangGraph thread once, using its latest checkpoint."""
    threads = {}
    for checkpoint in app.state.graph.checkpointer.list(None):
        key = checkpoint.config['configurable']['thread_id']
        if key not in threads:
            threads[key] = {'id': key, 'title': thread_title(checkpoint.checkpoint['channel_values']),
                            'updated': checkpoint.checkpoint['ts']}
    return sorted(threads.values(), key=lambda thread: thread['updated'], reverse=True)

class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=100)

@app.get('/conversations')
def list_conversations():
    with lock:
        return unique_threads()

@app.post('/conversations')
def create_conversation():
    key = uuid.uuid4().hex
    with lock:
        app.state.graph.update_state(thread_config(key), {'messages': [], 'chat_title': 'New conversation'}, as_node='generate')
    return {'id': key, 'title': 'New conversation'}

@app.get('/conversations/{key}')
def get_conversation(key: str):
    with lock:
        snapshot = app.state.graph.get_state(thread_config(key))
        if not snapshot.values:
            raise HTTPException(404, 'Conversation not found')
        messages = [{'id': message.id, 'role': 'user' if message.type == 'human' else 'assistant',
                     'content': message.content, **message.additional_kwargs.get('quran_response', {})}
                    for message in snapshot.values.get('messages', []) if message.type in ('human', 'ai')]
        return {'id': key, 'title': thread_title(snapshot.values), 'messages': messages}

@app.patch('/conversations/{key}')
def rename_conversation(key: str, request: RenameRequest):
    if not request.title.strip():
        raise HTTPException(422, 'A title cannot be empty')
    with lock:
        if not app.state.graph.get_state(thread_config(key)).values:
            raise HTTPException(404, 'Conversation not found')
        app.state.graph.update_state(thread_config(key), {'chat_title': request.title.strip()}, as_node='generate')
    return {'ok': True}

@app.delete('/conversations/{key}')
def delete_conversation(key: str):
    with lock:
        app.state.graph.checkpointer.delete_thread(key)
    return {'ok': True}

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    thread_id: str = Field(min_length=1, max_length=128, pattern=r"^[\w-]+$")

def configuration(request):
    return {"configurable": {"thread_id": request.thread_id}}

@app.get("/health")
def health():
    # Health checks must not wait behind a slow model generation request.
    counts = app.state.retriever.progress()
    total = len(app.state.retriever.verses)
    return {"status": "ok", "verses": total, "vector_index_ready": all(count == total for count in counts.values()), "vector_counts": counts}

@app.get("/verses/{surah}/{ayah}")
def verse(surah: int, ayah: int):
    result = app.state.store.get(f"{surah}:{ayah}")
    if not result:
        raise HTTPException(404, "Verse not found")
    return result

@app.get("/search/word")
def word_search(word: str):
    try:
        verses = app.state.store.word_search(word)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"count": len(verses), "translation": "Saheeh International", "verses": verses}

@app.get("/search/word/count")
def arabic_word_count(word: str, mode: str = "lemma"):
    try:
        return app.state.store.arabic_word_count(word, mode=mode)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

@app.post("/chat")
def chat(request: ChatRequest):
    with lock:
        try:
            state = app.state.graph.invoke({"messages": [HumanMessage(content=request.question)]}, configuration(request))
        except Exception as exc:
            logger.exception('Quran chat request failed')
            status, detail = request_error(exc)
            raise HTTPException(status, detail) from exc
        result = {"thread_id": request.thread_id, "answer": state["answer"], "verses": state["verses"], "notices": state["notices"], "word_count": state.get("word_count"), "matched_headers": state.get("matched_headers", [])}
        return result

@app.post("/chat/stream")
def stream(request: ChatRequest):
    def events():
        with lock:
            try:
                for event in app.state.graph.stream({"messages": [HumanMessage(content=request.question)]}, configuration(request), stream_mode="custom"):
                    yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
                yield 'data: {"type":"done"}\n\n'
            except Exception as exc:
                logger.exception('Quran streaming request failed')
                _, detail = request_error(exc)
                yield 'data: ' + json.dumps({'type':'error','message':detail}) + '\n\n'
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
