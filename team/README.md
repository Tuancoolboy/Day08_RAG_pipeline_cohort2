# Bài Tập Nhóm — Vite RAG Chatbot

## Mục Tiêu

Xây dựng RAG Chatbot hỏi đáp về pháp luật ma túy và tin tức liên quan.

Stack hiện tại:

```text
Vite React UI → FastAPI Backend → Retrieval + Vector DB → Generation → Citation Display
```

Sản phẩm có:

- Website Vite React chạy localhost.
- FastAPI backend không expose API key ra browser.
- Local persisted vector database.
- Hybrid/vector retrieval + reranking.
- OpenAI generation nếu có `OPENAI_API_KEY`.
- Local fallback generation nếu chưa có key.
- Evaluation pipeline 15 Q&A với A/B comparison.

## Kiến Trúc Hệ Thống

```text
Vite React UI (team/web/)
  │
  └─→ FastAPI Backend (team/api.py)
        │
        ├─→ Retrieval Pipeline (team/src/retrieval.py)
        │     ├─ Local Vector DB (team/vector_store/index.json)
        │     ├─ Lightweight lexical score
        │     ├─ Lightweight semantic score
        │     ├─ Hybrid vector merge
        │     └─ Reranking
        │
        ├─→ Generation Service (team/src/generation.py)
        │     ├─ OpenAI nếu có OPENAI_API_KEY
        │     └─ Local fallback nếu thiếu key/API lỗi
        │
        └─→ JSON response
              ├─ Answer
              ├─ Citations
              ├─ Source documents
              └─ Relevance scores

Evaluation (team/evaluation/eval_pipeline.py)
  ├─ Golden dataset 15 Q&A
  ├─ Metrics: faithfulness, answer relevance, context recall, context precision
  └─ A/B: hybrid_vector_rerank vs vector_only vs hybrid_rerank vs dense_only
```

## API Contract

Backend chạy tại `http://localhost:8000`.

Endpoints:

```text
GET  /api/health
GET  /api/vector-store
POST /api/chat
```

Request `POST /api/chat`:

```json
{
  "query": "Điều 249 quy định gì?",
  "top_k": 5,
  "retrieval_mode": "hybrid_vector",
  "use_reranking": true,
  "use_openai": true
}
```

Response:

```json
{
  "answer": "string",
  "sources": [
    {
      "content": "string",
      "score": 0.0,
      "metadata": {
        "id": "string",
        "title": "string",
        "source": "string",
        "year": "string",
        "doc_type": "string"
      }
    }
  ],
  "metadata": {
    "mode": "openai | local_fallback | no_context",
    "retrieval_mode": "hybrid_vector",
    "use_reranking": true
  }
}
```

## Cài Đặt

```bash
pip install -r requirements.txt
```

Tạo `.env` cho backend:

```bash
cp .env.example .env
```

Ví dụ:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_TIMEOUT_SECONDS=30
```

Nếu chưa có `OPENAI_API_KEY`, app vẫn demo được bằng local fallback generation.

## Build Vector Database

```bash
python3 team/scripts/build_vector_index.py
```

Output:

```text
team/vector_store/index.json
```

Retrieval modes:

- `hybrid_vector`: vector DB + lexical + semantic + reranking.
- `vector`: chỉ dùng vector DB.
- `hybrid`: lexical + semantic, không dùng vector DB.
- `dense_only`: semantic-style score.
- `lexical_only`: lexical score.

## Chạy Localhost

Terminal 1, chạy backend:

```bash
uvicorn team.api:app --reload --port 8000
```

Terminal 2, chạy frontend:

```bash
cd team/web
npm install
npm run dev
```

Mở website:

```text
http://localhost:5173
```

Nếu cần đổi backend URL cho frontend:

```bash
cp team/web/.env.example team/web/.env
```

```env
VITE_API_BASE_URL=http://localhost:8000
```

Frontend không đọc `OPENAI_API_KEY` và không gọi OpenAI trực tiếp. API key chỉ nằm trong `.env` của backend Python.

## Evaluation

```bash
python3 team/evaluation/eval_pipeline.py
```

Kết quả được ghi vào:

```text
team/evaluation/results.md
```

Metrics:

- Faithfulness.
- Answer relevance.
- Context recall.
- Context precision.

## Test

```bash
python3 -m unittest discover team/tests -v
python3 -m compileall team/src team/api.py team/evaluation/eval_pipeline.py
```

API smoke test:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"Điều 249 quy định gì?","top_k":5,"retrieval_mode":"hybrid_vector","use_reranking":true,"use_openai":false}'
```

Frontend build:

```bash
cd team/web
npm run build
```

## Files Chính

- `team/api.py`: FastAPI backend cho Vite UI.
- `team/web/`: Vite React frontend.
- `team/data/knowledge_base.json`: knowledge base demo.
- `team/scripts/build_vector_index.py`: build local vector database.
- `team/vector_store/index.json`: persisted vector database.
- `team/src/retrieval.py`: hybrid/vector retrieval + reranking.
- `team/src/vector_store.py`: local vector DB build/search/status.
- `team/src/pipeline.py`: end-to-end RAG pipeline.
- `team/src/generation.py`: OpenAI generation + error handling.
- `team/evaluation/eval_pipeline.py`: evaluation offline + A/B comparison.
- `team/evaluation/results.md`: kết quả evaluation đã export.

## Phân Công

| Người | Thành viên | Mảng phụ trách | File nên submit |
|------|------------|----------------|-----------------|
| Người 1 | Bùi Ngọc Khánh | API + pipeline backend | `team/api.py`, `team/src/config.py`, `team/src/pipeline.py`, `team/__init__.py` |
| Người 2 | Nguyễn Xuân Hiệp | Retrieval + vector store | `team/src/retrieval.py`, `team/src/vector_store.py`, `team/scripts/build_vector_index.py`, `team/vector_store/index.json` |
| Người 3 | Nguyễn Quang Huy | Generation + citation + OpenAI fallback | `team/src/generation.py`, một phần test trong `team/tests/test_openai_generation.py` |
| Người 4 | Vũ Hải Tuấn | Frontend Vite React UI | `team/web/src/main.jsx`, `team/web/src/styles.css`, `team/web/package.json`, `team/web/package-lock.json`, `team/web/vite.config.js`, `team/web/index.html`, `team/web/.env.example` |
| Người 5 | Nguyễn Văn Dương | Data + evaluation + documentation | `team/data/knowledge_base.json`, `team/evaluation/golden_dataset.json`, `team/evaluation/eval_pipeline.py`, `team/evaluation/results.md`, `team/README.md` |
