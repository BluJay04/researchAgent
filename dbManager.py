import os
import json
from typing import List, Dict, Optional, Any
import chromadb #type: ignore
from chromadb.config import Settings #type: ignore
from sentence_transformers import SentenceTransformer #type: ignore
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages ChromaDB for paper storage and citation relationships"""

    def __init__(self, db_path: str = "./research_db"):
        """Initialize ChromaDB and embedding model"""
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Create collections
        self._initialize_collections()

    def _initialize_collections(self):
        """Initialize ChromaDB collections"""
        try:
            # Papers collection for storage
            self.papers_collection = self.client.get_or_create_collection(
                name="research_papers",
                metadata={"description": "Research papers with embeddings"}
            )

            # Citations collection for citation graph
            self.citations_collection = self.client.get_or_create_collection(
                name="citations",
                metadata={"description": "Citation relationships between papers"}
            )

            # Methods and datasets collection
            self.methods_collection = self.client.get_or_create_collection(
                name="methods_datasets",
                metadata={"description": "Extracted methods and datasets"}
            )

            logger.info("ChromaDB collections initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing collections: {e}")
            raise

    async def add_paper(self, paper_data: Dict[str, Any]) -> str:
        """Add a paper to the database with embeddings"""
        try:
            # Create unique ID
            paper_id = f"{paper_data.get('arxiv_id', '')}{paper_data.get('semantic_scholar_id', '')}{hash(paper_data['title'])}"

            # Create text for embedding (title + abstract)
            embedding_text = f"{paper_data['title']} {paper_data.get('abstract', '')}"

            # Generate embedding
            embedding = self.embedding_model.encode(embedding_text).tolist()

            # Prepare metadata
            metadata = {
                "title": paper_data["title"],
                "authors": json.dumps(paper_data.get("authors", [])),
                "arxiv_id": paper_data.get("arxiv_id", ""),
                "semantic_scholar_id": paper_data.get("semantic_scholar_id", ""),
                "url": paper_data.get("url", ""),
                "publication_date": paper_data.get("publication_date", ""),
                "venue": paper_data.get("venue", ""),
                "citation_count": paper_data.get("citation_count", 0),
                "methods": json.dumps(paper_data.get("methods", [])),
                "datasets": json.dumps(paper_data.get("datasets", [])),
                "keywords": json.dumps(paper_data.get("keywords", [])),
                "added_date": datetime.now().isoformat()
            }

            # Add to papers collection
            self.papers_collection.add(
                ids=[paper_id],
                embeddings=[embedding],
                documents=[embedding_text],
                metadatas=[metadata]
            )

            # Add methods and datasets separately for better search
            await self._add_methods_datasets(paper_id, paper_data)

            logger.info(f"Added paper: {paper_data['title']}")
            return paper_id

        except Exception as e:
            logger.error(f"Error adding paper: {e}")
            raise

    async def _add_methods_datasets(self, paper_id: str, paper_data: Dict[str, Any]):
        """Add methods and datasets to separate collection"""
        try:
            methods = paper_data.get("methods", [])
            datasets = paper_data.get("datasets", [])

            for i, method in enumerate(methods):
                method_id = f"{paper_id}_method_{i}"
                method_embedding = self.embedding_model.encode(method).tolist()

                self.methods_collection.add(
                    ids=[method_id],
                    embeddings=[method_embedding],
                    documents=[method],
                    metadatas=[{
                        "paper_id": paper_id,
                        "type": "method",
                        "content": method,
                        "paper_title": paper_data["title"]
                    }]
                )

            for i, dataset in enumerate(datasets):
                dataset_id = f"{paper_id}_dataset_{i}"
                dataset_embedding = self.embedding_model.encode(dataset).tolist()

                self.methods_collection.add(
                    ids=[dataset_id],
                    embeddings=[dataset_embedding],
                    documents=[dataset],
                    metadatas=[{
                        "paper_id": paper_id,
                        "type": "dataset",
                        "content": dataset,
                        "paper_title": paper_data["title"]
                    }]
                )

        except Exception as e:
            logger.error(f"Error adding methods/datasets: {e}")

    async def find_similar_papers(self, paper_id: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Find papers similar to a given paper"""
        try:
            # Get the paper's embedding
            paper_result = self.papers_collection.get(
                ids=[paper_id],
                include=["embeddings", "metadatas"]
            )

            if not paper_result["embeddings"]:
                return []

            paper_embedding = paper_result["embeddings"][0]

            # Search for similar papers
            results = self.papers_collection.query(
                query_embeddings=[paper_embedding],
                n_results=n_results + 1,  # +1 to exclude the paper itself
                include=["documents", "metadatas", "distances"]
            )

            # Format results (excluding the paper itself)
            formatted_results = []
            for i in range(len(results["ids"][0])):
                if results["ids"][0][i] != paper_id:  # Exclude the input paper
                    paper_data = {
                        "id": results["ids"][0][i],
                        "title": results["metadatas"][0][i]["title"],
                        "similarity_score": 1 - results["distances"][0][i],
                        "url": results["metadatas"][0][i]["url"]
                    }
                    formatted_results.append(paper_data)

            return formatted_results[:n_results]

        except Exception as e:
            logger.error(f"Error finding similar papers: {e}")
            return []

    async def add_citation_relationship(self, citing_paper_id: str, cited_paper_id: str):
        """Add a citation relationship between papers"""
        try:
            citation_id = f"{citing_paper_id}_cites_{cited_paper_id}"

            # Create embedding for the relationship (could be enhanced with context)
            relationship_text = f"citation from {citing_paper_id} to {cited_paper_id}"
            embedding = self.embedding_model.encode(relationship_text).tolist()

            self.citations_collection.add(
                ids=[citation_id],
                embeddings=[embedding],
                documents=[relationship_text],
                metadatas=[{
                    "citing_paper": citing_paper_id,
                    "cited_paper": cited_paper_id,
                    "relationship_type": "cites",
                    "added_date": datetime.now().isoformat()
                }]
            )

        except Exception as e:
            logger.error(f"Error adding citation relationship: {e}")

    async def get_forward_citations(self, paper_id: str) -> List[str]:
        """Get papers that cite this paper (forward chaining)"""
        try:
            results = self.citations_collection.get(
                where={"cited_paper": paper_id},
                include=["metadatas"]
            )

            return [metadata["citing_paper"] for metadata in results["metadatas"]]

        except Exception as e:
            logger.error(f"Error getting forward citations: {e}")
            return []

    async def get_backward_citations(self, paper_id: str) -> List[str]:
        """Get papers that this paper cites (backward chaining)"""
        try:
            results = self.citations_collection.get(
                where={"citing_paper": paper_id},
                include=["metadatas"]
            )

            return [metadata["cited_paper"] for metadata in results["metadatas"]]

        except Exception as e:
            logger.error(f"Error getting backward citations: {e}")
            return []

    async def search_methods_datasets(self, query: str, search_type: str = "both", n_results: int = 10) -> List[Dict[str, Any]]:
        """Search for specific methods or datasets"""
        try:
            query_embedding = self.embedding_model.encode(query).tolist()

            where_clause = {}
            if search_type == "method":
                where_clause = {"type": "method"}
            elif search_type == "dataset":
                where_clause = {"type": "dataset"}
            # If "both", no where clause needed

            results = self.methods_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )

            formatted_results = []
            for i in range(len(results["ids"][0])):
                result_data = {
                    "content": results["metadatas"][0][i]["content"],
                    "type": results["metadatas"][0][i]["type"],
                    "paper_title": results["metadatas"][0][i]["paper_title"],
                    "paper_id": results["metadatas"][0][i]["paper_id"],
                    "similarity_score": 1 - results["distances"][0][i]
                }
                formatted_results.append(result_data)

            return formatted_results

        except Exception as e:
            logger.error(f"Error searching methods/datasets: {e}")
            return []

    async def get_paper_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Get a paper by its title (case-insensitive, trimmed)"""
        try:
            # Fetch all papers and match title case-insensitively
            results = self.papers_collection.get(
                include=["documents", "metadatas"]
            )
            for idx, metadata in enumerate(results["metadatas"]):
                if metadata["title"].strip().lower() == title.strip().lower():
                    return {
                        "id": results["ids"][idx],
                        "title": metadata["title"],
                        "authors": json.loads(metadata["authors"]),
                        "abstract": results["documents"][idx].replace(metadata["title"], "").strip(),
                        "arxiv_id": metadata.get("arxiv_id", ""),
                        "url": metadata["url"],
                        "methods": json.loads(metadata["methods"]),
                        "datasets": json.loads(metadata["datasets"]),
                        "keywords": json.loads(metadata["keywords"])
                    }
            return None
        except Exception as e:
            logger.error(f"Error getting paper by title: {e}")
            return None

    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            papers_count = self.papers_collection.count()
            citations_count = self.citations_collection.count()
            methods_datasets_count = self.methods_collection.count()

            return {
                "total_papers": papers_count,
                "total_citations": citations_count,
                "total_methods_datasets": methods_datasets_count,
                "database_path": self.db_path
            }

        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}

    def reset_database(self):
        """Reset the entire database (use with caution)"""
        try:
            self.client.reset()
            self._initialize_collections()
            logger.info("Database reset successfully")
        except Exception as e:
            logger.error(f"Error resetting database: {e}")
            raise