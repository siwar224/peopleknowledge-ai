# RAG

This module will hold the components of the RAG pipeline:

- loading and splitting the documents;
- generating the embeddings;
- indexing and vector search;
- building a prompt-ready context with short, verifiable source labels (`S1`, `S2`, ...);
- generating the answer;
- evaluation on `datasets/evaluation/rag_questions_gold.jsonl`.
