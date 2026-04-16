# Papers, Please! 

## Architecture 

![ Alt text](assets/diagram.png)

System is split into 4 services

- Frontend: Build with React with the help of Claude
- Backend: 
  - REST API using FastAPI
  - `fetch` new paper metadata using [SemanticScholar's API](https://www.semanticscholar.org/)
  - `query` registered papers, enabling user to search inside PDFs
- Worker:
  - Performs slow batch processing tasks
  - Download paper PDFs automatically 
  - Extract text from PDF with [RapidOCR](https://github.com/rapidai/rapidocr)
  - Chunk text with [Docling](https://www.docling.ai/) `HybridChunker`
  - Embeds chunks and indexes to [PineCone vector DB](https://www.pinecone.io/)
- Postgres DB
