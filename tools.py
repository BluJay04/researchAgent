import asyncio
from typing import List, Dict, Any, Optional
from langchain.tools import Tool #type: ignore
import arxiv as ar #type: ignore
from semanticscholar import SemanticScholar #type: ignore
from dbManager import DatabaseManager
from analyzer import PaperAnalyzer
import logging

logger = logging.getLogger(__name__)

class ResearchToolkit:
    """Enhanced research toolkit with semantic search and citation analysis"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.semantic_scholar = SemanticScholar()
        self.analyzer = PaperAnalyzer()
        
    def get_all_tools(self) -> List[Tool]:
        """Return all available research tools"""
        return [
            self._create_semantic_search_tool(),
            self._create_arxiv_search_tool(),
            self._create_semantic_scholar_tool(),
            self._create_citation_analysis_tool(),
            self._create_similar_papers_tool(),
            self._create_methods_datasets_search_tool(),
            self._create_paper_analysis_tool(),
            self._create_database_stats_tool()
        ]
    
    def _create_semantic_search_tool(self) -> Tool:
        """Tool for semantic search across stored papers"""
        async def semantic_search(query: str) -> str:
            try:
                results = await self.db_manager.semantic_search(query, n_results=10)
                
                if not results:
                    return f"No papers found for query: {query}"
                
                output = [f"🔍 Semantic Search Results for: {query}\n"]
                
                for i, paper in enumerate(results, 1):
                    output.append(f"{i}. {paper['title']}")
                    output.append(f"   Authors: {', '.join(paper['authors'])}")
                    output.append(f"   Similarity: {paper['similarity_score']:.3f}")
                    output.append(f"   URL: {paper['url']}")
                    if paper['methods']:
                        output.append(f"   Methods: {', '.join(paper['methods'][:3])}")
                    if paper['datasets']:
                        output.append(f"   Datasets: {', '.join(paper['datasets'][:3])}")
                    output.append("")
                
                return "\n".join(output)
                
            except Exception as e:
                return f"Error in semantic search: {str(e)}"
        
        return Tool(
            name="semantic_search",
            func=lambda query: asyncio.run(semantic_search(query)),
            description="Search stored papers using semantic similarity. Use for finding relevant papers based on concepts, not just keywords."
        )
    
    def _create_arxiv_search_tool(self) -> Tool:
        """Enhanced ArXiv search with automatic storage"""
        async def arxiv_search(query: str) -> str:
            try:
                search = ar.Search(
                    query=query,
                    max_results=5,
                    sort_by=ar.SortCriterion.Relevance
                )
                
                results = list(search.results())
                if not results:
                    return f"No ArXiv papers found for: {query}"
                
                output = [f"📚 ArXiv Search Results for: {query}\n"]
                
                for i, paper in enumerate(results, 1):
                    # Analyze paper for methods and datasets
                    analysis = await self.analyzer.extract_methods_datasets(
                        paper.title, paper.summary
                    )
                    
                    # Prepare paper data for storage
                    paper_data = {
                        "title": paper.title,
                        "authors": [author.name for author in paper.authors],
                        "abstract": paper.summary,
                        "arxiv_id": paper.entry_id.split('/')[-1],
                        "url": paper.entry_id,
                        "publication_date": paper.published.strftime("%Y-%m-%d"),
                        "methods": analysis.get("methods", []),
                        "datasets": analysis.get("datasets", []),
                        "keywords": analysis.get("keywords", [])
                    }
                    
                    # Store in database
                    paper_id = await self.db_manager.add_paper(paper_data)
                    
                    output.append(f"{i}. {paper.title}")
                    output.append(f"   Authors: {', '.join([a.name for a in paper.authors])}")
                    output.append(f"   ArXiv ID: {paper.entry_id.split('/')[-1]}")
                    output.append(f"   Published: {paper.published.strftime('%Y-%m-%d')}")
                    output.append(f"   URL: {paper.entry_id}")
                    if analysis.get("methods"):
                        output.append(f"   Methods: {', '.join(analysis['methods'][:3])}")
                    if analysis.get("datasets"):
                        output.append(f"   Datasets: {', '.join(analysis['datasets'][:3])}")
                    output.append(f"   Stored with ID: {paper_id}")
                    output.append("")
                
                return "\n".join(output)
                
            except Exception as e:
                return f"Error searching ArXiv: {str(e)}"
        
        return Tool(
            name="arxiv_search",
            func=lambda query: asyncio.run(arxiv_search(query)),
            description="Search ArXiv for papers and automatically store them with extracted methods and datasets."
        )
    
    def _create_semantic_scholar_tool(self) -> Tool:
        """Enhanced Semantic Scholar search with citation data"""
        async def semantic_scholar_search(query: str) -> str:
            try:
                papers = self.semantic_scholar.search_paper(query, limit=5)
                
                if not papers:
                    return f"No Semantic Scholar papers found for: {query}"
                
                output = [f"🎓 Semantic Scholar Results for: {query}\n"]
                
                for i, paper in enumerate(papers, 1):
                    # Get detailed paper info
                    try:
                        detailed_paper = self.semantic_scholar.get_paper(paper['paperId'])
                        
                        # Analyze abstract for methods and datasets
                        abstract = detailed_paper.get('abstract', '')
                        analysis = await self.analyzer.extract_methods_datasets(
                            paper['title'], abstract
                        )
                        
                        # Prepare paper data
                        paper_data = {
                            "title": paper['title'],
                            "authors": [author.get('name', '') for author in paper.get('authors', [])],
                            "abstract": abstract,
                            "semantic_scholar_id": paper['paperId'],
                            "url": paper.get('url', ''),
                            "publication_date": paper.get('year', ''),
                            "venue": paper.get('venue', ''),
                            "citation_count": paper.get('citationCount', 0),
                            "methods": analysis.get("methods", []),
                            "datasets": analysis.get("datasets", []),
                            "keywords": analysis.get("keywords", [])
                        }
                        
                        # Store in database
                        paper_id = await self.db_manager.add_paper(paper_data)
                        
                        # Store citation relationships
                        if detailed_paper.get('references'):
                            for ref in detailed_paper['references'][:10]:  # Limit to first 10
                                if ref.get('paperId'):
                                    await self.db_manager.add_citation_relationship(
                                        paper_id, ref['paperId']
                                    )
                        
                        output.append(f"{i}. {paper['title']}")
                        output.append(f"   Authors: {', '.join([a.get('name', '') for a in paper.get('authors', [])])}")
                        output.append(f"   Venue: {paper.get('venue', 'N/A')}")
                        output.append(f"   Year: {paper.get('year', 'N/A')}")
                        output.append(f"   Citations: {paper.get('citationCount', 0)}")
                        output.append(f"   URL: {paper.get('url', 'N/A')}")
                        if analysis.get("methods"):
                            output.append(f"   Methods: {', '.join(analysis['methods'][:3])}")
                        if analysis.get("datasets"):
                            output.append(f"   Datasets: {', '.join(analysis['datasets'][:3])}")
                        output.append(f"   Stored with ID: {paper_id}")
                        output.append("")
                        
                    except Exception as e:
                        logger.error(f"Error processing paper {paper['paperId']}: {e}")
                        continue
                
                return "\n".join(output)
                
            except Exception as e:
                return f"Error searching Semantic Scholar: {str(e)}"
        
        return Tool(
            name="semantic_scholar_search",
            func=lambda query: asyncio.run(semantic_scholar_search(query)),
            description="Search Semantic Scholar for papers with citation data and automatically store them."
        )
    
    def _create_citation_analysis_tool(self) -> Tool:
        """Tool for citation network analysis"""
        async def citation_analysis(paper_title: str) -> str:
            try:
                # Find paper in database
                paper = await self.db_manager.get_paper_by_title(paper_title)
                if not paper:
                    return f"Paper '{paper_title}' not found in database. Please search for it first."
                
                paper_id = paper['id']
                
                # Get forward and backward citations
                forward_citations = await self.db_manager.get_forward_citations(paper_id)
                backward_citations = await self.db_manager.get_backward_citations(paper_id)
                
                output = [f"🕸️ Citation Analysis for: {paper_title}\n"]
                
                output.append(f"Papers that cite this paper (Forward citations): {len(forward_citations)}")
                for citation_id in forward_citations[:5]:  # Show first 5
                    citing_paper = await self.db_manager.get_paper_by_title(citation_id)
                    if citing_paper:
                        output.append(f"  - {citing_paper['title']}")
                
                output.append(f"\nPapers cited by this paper (Backward citations): {len(backward_citations)}")
                for citation_id in backward_citations[:5]:  # Show first 5
                    cited_paper = await self.db_manager.get_paper_by_title(citation_id)
                    if cited_paper:
                        output.append(f"  - {cited_paper['title']}")
                
                return "\n".join(output)
                
            except Exception as e:
                return f"Error in citation analysis: {str(e)}"
        
        return Tool(
            name="citation_analysis",
            func=lambda paper_title: asyncio.run(citation_analysis(paper_title)),
            description="Analyze citation network for a paper (forward and backward chaining)."
        )
    
    def _create_similar_papers_tool(self) -> Tool:
        """Tool for finding similar papers"""
        async def find_similar_papers(paper_title: str) -> str:
            try:
                # Find paper in database
                paper = await self.db_manager.get_paper_by_title(paper_title)
                if not paper:
                    return f"Paper '{paper_title}' not found in database. Please search for it first."
                
                similar_papers = await self.db_manager.find_similar_papers(paper['id'], n_results=5)
                
                if not similar_papers:
                    return f"No similar papers found for: {paper_title}"
                
                output = [f"🔗 Papers similar to: {paper_title}\n"]
                
                for i, similar_paper in enumerate(similar_papers, 1):
                    output.append(f"{i}. {similar_paper['title']}")
                    output.append(f"   Similarity Score: {similar_paper['similarity_score']:.3f}")
                    output.append(f"   URL: {similar_paper['url']}")
                    output.append("")
                
                return "\n".join(output)
                
            except Exception as e:
                return f"Error finding similar papers: {str(e)}"
        
        return Tool(
            name="find_similar_papers",
            func=lambda paper_title: asyncio.run(find_similar_papers(paper_title)),
            description="Find papers similar to a given paper using embedding similarity."
        )
    
    def _create_methods_datasets_search_tool(self) -> Tool:
        """Tool for searching methods and datasets"""
        async def search_methods_datasets(query: str, search_type: str = "both") -> str:
            try:
                results = await self.db_manager.search_methods_datasets(
                    query, search_type, n_results=10
                )
                
                if not results:
                    return f"No {search_type} found for query: {query}"
                
                output = [f"🔬 {search_type.title()} Search Results for: {query}\n"]
                
                for i, result in enumerate(results, 1):
                    output.append(f"{i}. {result['content']} ({result['type']})")
                    output.append(f"   Paper: {result['paper_title']}")
                    output.append(f"   Similarity: {result['similarity_score']:.3f}")
                    output.append("")
                
                return "\n".join(output)
                
            except Exception as e:
                return f"Error searching methods/datasets: {str(e)}"
        
        return Tool(
            name="search_methods_datasets",
            func=lambda query_type: asyncio.run(search_methods_datasets(*query_type.split("|"))),
            description="Search for specific methods or datasets. Format: 'query|type' where type is 'method', 'dataset', or 'both'."
        )
    
    def _create_paper_analysis_tool(self) -> Tool:
        """Tool for detailed paper analysis"""
        async def analyze_paper(paper_title: str) -> str:
            try:
                paper = await self.db_manager.get_paper_by_title(paper_title)
                if not paper:
                    return f"Paper '{paper_title}' not found in database."
                
                # Get comprehensive analysis
                analysis = await self.analyzer.comprehensive_analysis(
                    paper['title'], paper['abstract']
                )
                
                output = [f"📊 Comprehensive Analysis: {paper_title}\n"]
                
                output.append(f"Methods Identified: {', '.join(analysis.get('methods', []))}")
                output.append(f"Datasets Identified: {', '.join(analysis.get('datasets', []))}")
                output.append(f"Keywords: {', '.join(analysis.get('keywords', []))}")
                output.append(f"Research Domain: {analysis.get('domain', 'Unknown')}")
                output.append(f"Methodology Type: {analysis.get('methodology_type', 'Unknown')}")
                output.append(f"Contribution Type: {analysis.get('contribution_type', 'Unknown')}")
                
                if analysis.get('strengths'):
                    output.append(f"Strengths: {', '.join(analysis['strengths'])}")
                if analysis.get('limitations'):
                    output.append(f"Limitations: {', '.join(analysis['limitations'])}")
                
                return "\n".join(output)
                
            except Exception as e:
                return f"Error analyzing paper: {str(e)}"
        
        return Tool(
            name="analyze_paper",
            func=lambda paper_title: asyncio.run(analyze_paper(paper_title)),
            description="Perform comprehensive analysis of a paper including methods, datasets, and research characteristics."
        )
    
    def _create_database_stats_tool(self) -> Tool:
        """Tool for database statistics"""
        async def get_database_stats() -> str:
            try:
                stats = await self.db_manager.get_database_stats()
                
                output = ["📈 Database Statistics\n"]
                output.append(f"Total Papers: {stats.get('total_papers', 0)}")
                output.append(f"Total Citation Relationships: {stats.get('total_citations', 0)}")
                output.append(f"Total Methods/Datasets: {stats.get('total_methods_datasets', 0)}")
                output.append(f"Database Path: {stats.get('database_path', 'Unknown')}")
                
                return "\n".join(output)
                
            except Exception as e:
                return f"Error getting database stats: {str(e)}"
        
        return Tool(
            name="database_stats",
            func=lambda: asyncio.run(get_database_stats()),
            description="Get statistics about the research database."
        )