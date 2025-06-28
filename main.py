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

class PaperComparison(BaseModel):
    """Structured comparison between papers"""
    papers: List[str] = Field(description="Paper titles being compared")
    similarities: List[str] = Field(description="Key similarities between papers")
    differences: List[str] = Field(description="Key differences between papers")
    methodological_comparison: str = Field(description="Comparison of methods used")
    dataset_comparison: str = Field(description="Comparison of datasets used")
    strengths_weaknesses: Dict[str, Dict[str, List[str]]] = Field(
        description="Strengths and weaknesses of each paper"
    )

class ResearchCritique(BaseModel):
    """Structured critique of research papers"""
    paper_title: str
    methodology_assessment: str
    data_quality_assessment: str
    novelty_assessment: str
    reproducibility_concerns: List[str]
    limitations: List[str]
    strengths: List[str]
    overall_rating: str = Field(description="Overall assessment: Excellent/Good/Fair/Poor")

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
    """Advanced AI Research Assistant with semantic search and citation analysis"""
    
    def __init__(self):
        self.llm = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model_name="llama3-70b-8192",
            temperature=0.1
        )
        
        # Initialize components
        self.db_manager = DatabaseManager()
        self.toolkit = ResearchToolkit(self.db_manager)
        self.analyzer = PaperAnalyzer(self.llm)
        
        # Setup parsers
        self.response_parser = PydanticOutputParser(pydantic_object=ResearchResponse)
        self.comparison_parser = PydanticOutputParser(pydantic_object=PaperComparison)
        self.critique_parser = PydanticOutputParser(pydantic_object=ResearchCritique)
        
        # Setup prompts
        self._setup_prompts()
        
        # Create agent
        self._create_agent()
    
    def _setup_prompts(self):
        """Setup different prompts for different tasks"""
        
        # Main research prompt
        self.research_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an advanced AI research assistant with access to academic databases and analysis tools.
            
            Your capabilities include:
            - Semantic search across academic papers
            - Citation network analysis (forward and backward chaining)
            - Method and dataset extraction
            - Paper similarity analysis
            - Structured data extraction
            
            Always use the available tools to:
            1. Search for papers using semantic search
            2. Analyze citation networks
            3. Extract methods and datasets
            4. Find similar papers
            5. Build comprehensive research summaries
            
            Return responses in the specified JSON format: {format_instructions}
            """),
            ("placeholder", "{chat_history}"),
            ("human", "{query}"),
            ("placeholder", "{agent_scratchpad}"),
        ]).partial(format_instructions=self.response_parser.get_format_instructions())
        
        # Paper comparison prompt
        self.comparison_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are analyzing and comparing research papers. Provide a detailed comparison
            focusing on methodologies, datasets, results, and contributions.
            
            Format your response as: {format_instructions}
            """),
            ("human", "Compare these papers: {papers}")
        ]).partial(format_instructions=self.comparison_parser.get_format_instructions())
        
        # Paper critique prompt
        self.critique_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are a research paper reviewer. Provide a thorough critique of the paper
            focusing on methodology, data quality, novelty, and reproducibility.
            
            Format your response as: {format_instructions}
            """),
            ("human", "Critique this paper: {paper}")
        ]).partial(format_instructions=self.critique_parser.get_format_instructions())
    
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
            raw_response = await self.agent_executor.ainvoke({"query": query})
            structured_response = self.response_parser.parse(raw_response["output"])
            return structured_response
        except Exception as e:
            print(f"Error in research: {e}")
            # Return a basic response with error info
            return ResearchResponse(
                topic=query,
                summary=f"Error occurred during research: {str(e)}",
                papers_found=[],
                total_papers=0,
                tools_used=["error_handler"]
            )
    
    async def compare_papers(self, paper_titles: List[str]) -> PaperComparison:
        """Compare multiple papers"""
        papers_data = []
        for title in paper_titles:
            paper_info = await self.db_manager.get_paper_by_title(title)
            if paper_info:
                papers_data.append(paper_info)
        
        if not papers_data:
            raise ValueError("No papers found for comparison")
        
        papers_text = "\n\n".join([
            f"Title: {paper['title']}\nAbstract: {paper['abstract']}\nMethods: {paper.get('methods', [])}\nDatasets: {paper.get('datasets', [])}"
            for paper in papers_data
        ])
        
        chain = self.comparison_prompt | self.llm | self.comparison_parser
        result = await chain.ainvoke({"papers": papers_text})
        return result
    
    async def critique_paper(self, paper_title: str) -> ResearchCritique:
        """Provide detailed critique of a paper"""
        paper_info = await self.db_manager.get_paper_by_title(paper_title)
        if not paper_info:
            raise ValueError(f"Paper '{paper_title}' not found in database")
        
        paper_text = f"Title: {paper_info['title']}\nAbstract: {paper_info['abstract']}\nMethods: {paper_info.get('methods', [])}\nDatasets: {paper_info.get('datasets', [])}"
        
        chain = self.critique_prompt | self.llm | self.critique_parser
        result = await chain.ainvoke({"paper": paper_text})
        return result
    
    async def ask_question_about_papers(self, question: str, paper_titles: List[str]) -> str:
        """Answer specific questions about papers"""
        papers_data = []
        for title in paper_titles:
            paper_info = await self.db_manager.get_paper_by_title(title)
            if paper_info:
                papers_data.append(paper_info)
        
        if not papers_data:
            return "No papers found to answer the question about."
        
        context = "\n\n".join([
            f"Paper: {paper['title']}\nAbstract: {paper['abstract']}\nMethods: {paper.get('methods', [])}\nDatasets: {paper.get('datasets', [])}"
            for paper in papers_data
        ])
        
        prompt = f"""
        Based on the following research papers, answer this question: {question}
        
        Papers:
        {context}
        
        Provide a detailed answer based on the information in these papers.
        """
        
        response = await self.llm.ainvoke(prompt)
        return response.content

async def main():
    """Main application loop"""
    assistant = AdvancedResearchAssistant()
    
    print("🔬 Advanced Research Assistant Ready!")
    print("Commands:")
    print("  - 'research <query>' - Research a topic")
    print("  - 'compare <paper1> | <paper2> | ...' - Compare papers")
    print("  - 'critique <paper_title>' - Critique a paper")
    print("  - 'ask <question> about <paper1> | <paper2> | ...' - Ask specific questions")
    print("  - 'quit' - Exit")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            if user_input.startswith('research '):
                query = user_input[9:]
                result = await assistant.research(query)
                print(f"\n📊 Research Results for: {result.topic}")
                print(f"Summary: {result.summary}")
                print(f"Papers found: {result.total_papers}")
                if result.extracted_methods:
                    print(f"Methods identified: {', '.join(result.extracted_methods)}")
                if result.extracted_datasets:
                    print(f"Datasets identified: {', '.join(result.extracted_datasets)}")
            
            elif user_input.startswith('compare '):
                papers = [p.strip() for p in user_input[8:].split('|')]
                result = await assistant.compare_papers(papers)
                print(f"\n📝 Paper Comparison")
                print(f"Similarities: {', '.join(result.similarities)}")
                print(f"Differences: {', '.join(result.differences)}")
                print(f"Methodological comparison: {result.methodological_comparison}")
            
            elif user_input.startswith('critique '):
                paper_title = user_input[9:]
                result = await assistant.critique_paper(paper_title)
                print(f"\n🔍 Critique of: {result.paper_title}")
                print(f"Overall rating: {result.overall_rating}")
                print(f"Strengths: {', '.join(result.strengths)}")
                print(f"Limitations: {', '.join(result.limitations)}")
            
            elif 'ask ' in user_input and ' about ' in user_input:
                parts = user_input.split(' about ')
                question = parts[0][4:]  # Remove 'ask '
                papers = [p.strip() for p in parts[1].split('|')]
                result = await assistant.ask_question_about_papers(question, papers)
                print(f"\n💬 Answer: {result}")
            
            else:
                print("Unknown command. Use 'research', 'compare', 'critique', 'ask', or 'quit'.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("Goodbye!")

if __name__ == "__main__":
    asyncio.run(main())