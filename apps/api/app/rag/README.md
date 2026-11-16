# RAG

This module will hold the components of the RAG pipeline:

- loading and splitting the documents;
- generating the embeddings;
- indexing and vector search;
- building a prompt-ready context with short, verifiable source labels (`S1`, `S2`, ...);
- building stable instructions plus a dynamic evidence-first prompt;
- generating and validating structured answers through an OpenAI-compatible server;
- evaluation on `datasets/evaluation/rag_questions_gold.jsonl`.
