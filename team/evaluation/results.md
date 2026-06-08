# RAG Evaluation Results

## Method

Evaluation runs offline with token-overlap metrics so the demo does not require API calls.
The chatbot uses local fallback generation during evaluation.
Vector configs search the persisted local vector database at `team/vector_store/index.json`.

## Overall Scores

| Config | Faithfulness | Answer Relevance | Context Recall | Context Precision |
|--------|--------------|------------------|----------------|-------------------|
| hybrid_vector_rerank | 0.971 | 0.990 | 1.000 | 0.670 |
| vector_only | 0.971 | 0.988 | 1.000 | 0.711 |
| hybrid_rerank | 0.970 | 0.990 | 1.000 | 0.660 |
| dense_only | 0.964 | 0.923 | 1.000 | 0.670 |

## A/B Comparison

Best overall config: `vector_only`.

Compared configs:

- `hybrid_vector_rerank`: persisted vector database + lexical/semantic signals + reranking.
- `vector_only`: persisted vector database only.
- `hybrid_rerank`: hybrid lexical/semantic retrieval with reranking.
- `dense_only`: semantic-style retrieval without reranking.

## Worst Performers

| Question | Top Source | Answer Relevance | Context Recall | Context Precision |
|----------|------------|------------------|----------------|-------------------|
| Nghị định 105/2021/NĐ-CP hướng dẫn những nội dung gì? | Nghi dinh 105/2021/ND-CP | 0.967 | 1.000 | 0.446 |
| Khi đọc tin nghệ sĩ liên quan đến ma túy cần kiểm chứng nguồn tin ra sao? | Tong hop tin tuc phap luat | 1.000 | 1.000 | 0.555 |
| Tiền chất ma túy cần được kiểm soát như thế nào? | Luat Phong, chong ma tuy 2021 va van ban huong dan | 1.000 | 1.000 | 0.644 |
| Tội chiếm đoạt chất ma túy được quy định ở điều nào? | Bo luat Hinh su 2015 sua doi 2017, Dieu 252 | 0.947 | 1.000 | 0.757 |
| Nghị định 57/2022/NĐ-CP dùng để làm gì trong quản lý ma túy? | Nghi dinh 105/2021/ND-CP | 1.000 | 1.000 | 0.706 |

## Recommendations

- Add more real legal documents and crawled news articles to increase context coverage.
- Replace local hashed embeddings with sentence-transformers or Weaviate Cloud when deployment is available.
- Keep vector or hybrid-vector retrieval enabled for demo because it improves source ordering for legal queries.
- Review low-scoring questions and add more granular chunks for specific articles.
