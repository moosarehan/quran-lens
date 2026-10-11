# QuranLens React frontend

The four supplied page designs are implemented as React components: welcome,
conversation, history with delete/rename dialogs, and citation context.

Run the backend from the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn quran_rag.api:app --host 127.0.0.1 --port 8000
```

Run the frontend in a separate terminal:

```powershell
cd E:\QURAN-llm\quran-rag-frontend
npm install
npm run dev
```

Open the URL printed by Vite. Its `/api` proxy connects to port 8000.
Use `npm run build` for the production frontend bundle. Production hosting must
forward `/api/*` to FastAPI, removing the `/api` prefix.

Each conversation ID is its LangGraph thread ID. Titles use the first question
and can be renamed. LangGraph's `SqliteSaver` stores everything in
`data/checkpoints.sqlite`. The thread list deduplicates `checkpointer.list()`;
clicking history uses `workflow.get_state()` to restore messages. Each assistant
message also carries its canonical verses and notices in `additional_kwargs`.
Continuing passes the same thread ID to `workflow.stream()`; deleting calls
`checkpointer.delete_thread()`. No separate conversation database is used.
Local storage holds only the selected thread and theme preference.

Streaming uses fetch and SSE from `/chat/stream`. Backend status events appear
while retrieval runs. Evidence-checked answers stream in text chunks, followed
by canonical verse cards; unchecked model drafts are never shown.

The source mockups' tafsir and simulated audio are omitted because the backend
supports Quran text only. Translation labels match the real Saheeh International
records rather than the mockups' alternative translations.

This is a local single-user application. Authentication and per-user access
control are needed before hosting shared conversation history for multiple users.
