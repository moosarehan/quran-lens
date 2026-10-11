"""Cloudflare Workers AI via LangChain's Embeddings interface."""
import math
import os
import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda
from .config import MODEL

class TemporaryEmbeddingError(RuntimeError):
    pass

class CloudflareEmbeddings(Embeddings):
    def __init__(self, client=None):
        account = os.getenv('CLOUDFLARE_ACCOUNT_ID', '')
        token = os.getenv('CLOUDFLARE_API_TOKEN', '')
        if not account or not token:
            raise ValueError('Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in .env')
        self.url = f'https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{MODEL}'
        self.client = client or httpx.Client(timeout=60, headers={'Authorization': f'Bearer {token}'})
        self.request = RunnableLambda(self._embed_batch).with_retry(
            retry_if_exception_type=(httpx.TransportError, TemporaryEmbeddingError), stop_after_attempt=3)

    def _embed_batch(self, texts):
        response = self.client.post(self.url, json={'text': texts, 'truncate_inputs': False})
        if response.status_code == 429 or response.status_code >= 500:
            raise TemporaryEmbeddingError(f'Cloudflare embeddings temporarily unavailable ({response.status_code})')
        if response.is_error:
            raise RuntimeError(f'Cloudflare embeddings rejected the request ({response.status_code}); check credentials and account access')
        body = response.json()
        if not body.get('success'):
            raise RuntimeError('Cloudflare embedding request was unsuccessful')
        vectors = body.get('result', {}).get('data', [])
        if len(vectors) != len(texts):
            raise ValueError('Cloudflare returned an unexpected number of embeddings')
        for vector in vectors:
            if len(vector) != 1024 or not all(isinstance(x, (float, int)) and math.isfinite(x) for x in vector):
                raise ValueError('Cloudflare returned an invalid BGE-M3 vector')
            if not any(vector):
                raise ValueError('Cloudflare returned an empty BGE-M3 vector')
        return vectors

    def embed_documents(self, texts):
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError('Embedding inputs must be nonempty text')
        result = []
        for start in range(0, len(texts), 32):
            result.extend(self.request.invoke(texts[start:start + 32]))
        return result

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    def close(self):
        self.client.close()
