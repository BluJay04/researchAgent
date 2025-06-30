import os
import asyncio
from typing import Dict, List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq #type: ignore
from langchain_core.prompts import ChatPromptTemplate #type: ignore
from langchain_core.output_parsers import PydanticOutputParser #type: ignore
from langchain.agents import create_tool_calling_agent, AgentExecutor #type: ignore
from tools import ResearchToolkit
from dbManager import DatabaseManager
from analyzer import PaperAnalyzer

load_dotenv()

class Paper(BaseModel):
    """Structured representation of a research paper"""
    title: str
    authors: List[str]
    abstract: str
    arxiv_id: Optional[str] = None
    semantic_scholar_id: Optional[str] = None
    url: str
    publication_date: Optional[str] = None
    venue: Optional[str] = None
    citation_count: Optional[int] = None
    methods: List[str] = Field(default_factory=list)
    datasets: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)

class ResearchResponse(BaseModel):
    """Enhanced research response with structured data"""
    topic: str
    summary: str
    papers_found: List[Paper]
    total_papers: int
    tools_used: List[str]
    citation_network: Optional[Dict[str, List[str]]] = None
    similar_papers: Optional[List[str]] = None
    extracted_methods: List[str] = Field(default_factory=list)
    extracted_datasets: List[str] = Field(default_factory=list)

class AdvancedResearchAssistant:
    def __init__(self):
        self.llm = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model_name="llama3-70b-8192",
            temperature=0.1
        )
        self.db_manager = DatabaseManager()
        self.toolkit = ResearchToolkit(self.db_manager)
        self.analyzer = PaperAnalyzer(self.llm)
        self.response_parser = PydanticOutputParser(pydantic_object=ResearchResponse)
        self._setup_prompts()
        self._create_agent()

    def _setup_prompts(self):
        """Setup different prompts for different tasks"""
        self.research_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an advanced AI research assistant with access to academic databases and analysis tools.

            Your capabilities include:
            - Arxiv search across academic papers (by keywords)
            - Method-based search
            - Citation network analysis (forward and backward chaining)
            - Paper similarity analysis
            - Structured data extraction

            Always use the available tools to:
            1. Search for papers using arxiv search (by keywords)
            2. Search for papers by method
            3. Analyze citation networks
            4. Find similar papers
            5. Build comprehensive research summaries

            Return responses in the specified JSON format: {format_instructions}
            """),
            ("placeholder", "{chat_history}"),
            ("human", "{query}"),
            ("placeholder", "{agent_scratchpad}"),
        ]).partial(format_instructions=self.response_parser.get_format_instructions())

        self.summarize_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert at summarizing research papers.
            Provide a concise summary of the following paper, focusing on its main contributions, methods, and findings.
            """),
            ("human", "Title: {title}\n\nAbstract: {abstract}")
        ])
    
    def _create_agent(self):
        """Create the research agent with all tools"""
        tools = self.toolkit.get_all_tools()
        llm_with_tools = self.llm.bind_tools(tools)
        self.agent = create_tool_calling_agent(
            llm=llm_with_tools,
            prompt=self.research_prompt,
            tools=tools
        )
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10
        )

    async def research(self, query: str) -> ResearchResponse:
        """Main research function"""
        try:
            raw_response = await self.agent_executor.ainvoke({
                "query": query
            })
            structured_response = self.response_parser.parse(raw_response["output"])
            return structured_response
        except Exception as e:
            print(f"Error in research: {e}")
            return ResearchResponse(
                topic=query,
                summary=f"Error occurred during research: {str(e)}",
                papers_found=[],
                total_papers=0,
                tools_used=["error_handler"]
            )

    async def summarize(self, paper_title: str) -> str:
        paper_info = await self.db_manager.get_paper_by_title(paper_title)
        if not paper_info:
            return f"Paper '{paper_title}' not found in database."
        chain = self.summarize_prompt | self.llm
        response = await chain.ainvoke({"title": paper_info["title"], "abstract": paper_info["abstract"]})
        return response.content

    async def download_pdf(self, paper_title: str) -> str:
        import requests
        paper_info = await self.db_manager.get_paper_by_title(paper_title)
        if not paper_info or not paper_info.get("arxiv_id"):
            return f"Paper '{paper_title}' not found or does not have an arXiv ID."
        arxiv_id = paper_info["arxiv_id"]
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        papers_dir = os.path.join(os.getcwd(), "papers")
        os.makedirs(papers_dir, exist_ok=True)
        filename = os.path.join(papers_dir, f"{arxiv_id}.pdf")
        try:
            response = requests.get(pdf_url)
            if response.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(response.content)
                return f"PDF downloaded as {filename}"
            else:
                return f"Failed to download PDF from {pdf_url}"
        except Exception as e:
            return f"Error downloading PDF: {e}"

async def main():
    """Main application loop"""
    assistant = AdvancedResearchAssistant()
    
    print("🔬 Advanced Research Assistant Ready!")
    print("Commands:")
    print("  - 'research <query>' - Research a topic (by keywords or method)")
    print("  - 'summarize <paper_title>' - Summarize a paper")
    print("  - 'download <paper_title>' - Download paper PDF (arXiv only, saved to papers/)")
    print("  - 'quit' - Exit")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() == 'quit':
                break

            elif user_input.startswith('research '):
                query = user_input[9:].strip()
                result = await assistant.research(query)
                print_research_result(result)

            elif user_input.startswith('summarize '):
                paper_title = user_input[10:]
                summary = await assistant.summarize(paper_title)
                print(f"\n📝 Summary of '{paper_title}':\n{summary}")

            elif user_input.startswith('download '):
                paper_title = user_input[9:]
                message = await assistant.download_pdf(paper_title)
                print(f"\n{message}")

            else:
                print("Unknown command. Use 'research', 'summarize', 'download', or 'quit'.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("Goodbye!")

def print_research_result(result):
    print(f"\n📊 Research Results for: {result.topic}")
    print(f"Summary: {result.summary}")
    print(f"Papers found: {result.total_papers}")
    for idx, paper in enumerate(result.papers_found, 1):
        print(f"\n{idx}. {paper.title}")
        print(f"   Authors: {', '.join(paper.authors)}")
        print(f"   ArXiv ID: {paper.arxiv_id}")
        print(f"   Published: {paper.publication_date}")
        print(f"   URL: {paper.url}")
        if paper.methods:
            print(f"   Methods: {', '.join(paper.methods)}")
        if paper.datasets:
            print(f"   Datasets: {', '.join(paper.datasets)}")
        print(f"   Stored with ID: {paper.arxiv_id}{hash(paper.title)}")

if __name__ == "__main__":
    asyncio.run(main())