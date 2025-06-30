# ResearchAgent

**ResearchAgent** is an advanced AI-powered research assistant for academic literature discovery, semantic search, citation analysis, and structured research insights. It leverages LLMs, semantic embeddings, and APIs like arXiv and Semantic Scholar to help you find, analyze, and organize research papers efficiently.

---

## Features

- **ArXiv Search**: Find and store papers by keyword from arXiv.
- **Semantic Scholar Search**: Discover papers with citation data from Semantic Scholar.
- **Semantic Search**: Retrieve relevant papers from your database using semantic similarity.
- **Method Search**: Find papers by specific methods or techniques.
- **Citation Network Analysis**: Explore forward and backward citation relationships.
- **Similar Papers**: Find papers similar to a given one using embeddings.
- **Database Statistics**: Get stats about your research database.
- **Paper Summarization**: Generate concise summaries of stored papers.
- **PDF Download**: Download arXiv PDFs for offline reading.

---

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/BluJay04/researchAgent.git
   cd researchAgent/researchAgent
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   - Copy `.example.env` to `.env` and fill in your API keys (e.g., `GROQ_API_KEY`).

---

## Usage

Run the assistant from the project directory:

```sh
python main.py
```

### Commands

- `research <query>`  
  Search for papers by keyword or method.

- `summarize <paper_title>`  
  Summarize a stored paper.

- `download <paper_title>`  
  Download the PDF of an arXiv paper.

- `quit`  
  Exit the assistant.

---

## Project Structure

```
researchAgent/
│
├── analyzer.py         # Paper analysis and extraction logic
├── dbManager.py        # Database and embedding management
├── main.py             # CLI entry point and agent logic
├── tools.py            # Research tools and integrations
├── requirements.txt    # Python dependencies
├── .env.example        # Example environment variables
├── research_db/        # (Ignored) Database files
├── papers/             # (Ignored) Downloaded PDFs
└── .gitignore
```

---

## Configuration

- **API Keys:**  
  - Set your LLM and API keys in the `.env` file.
- **Database:**  
  - Uses [ChromaDB](https://www.trychroma.com/) for persistent storage and semantic search.

---

## Notes

- The assistant stores all found papers in a local database for fast future retrieval.
- Only metadata and embeddings are stored; PDFs are downloaded on demand.
- The CLI is interactive and supports both keyword and method-based queries.

---

## License

MIT License

---

## Acknowledgements

- [arXiv API](https://arxiv.org/help/api/)
- [Semantic Scholar API](https://www.semanticscholar.org/product/api)
- [LangChain](https://www.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)