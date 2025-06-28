import re
from typing import Dict, List, Any
from langchain_core.prompts import ChatPromptTemplate #type: ignore
from langchain_core.output_parsers import PydanticOutputParser #type: ignore
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class PaperAnalysis(BaseModel):
    """Structured analysis of a research paper"""
    methods: List[str] = Field(description="Methods and techniques used in the paper")
    datasets: List[str] = Field(description="Datasets mentioned or used in the paper")
    keywords: List[str] = Field(description="Key technical terms and concepts")
    domain: str = Field(description="Primary research domain/field")
    methodology_type: str = Field(description="Type of methodology (empirical, theoretical, survey, etc.)")
    contribution_type: str = Field(description="Type of contribution (novel method, dataset, analysis, etc.)")
    strengths: List[str] = Field(description="Identified strengths of the work")
    limitations: List[str] = Field(description="Identified limitations or potential issues")

class PaperAnalyzer:
    """Analyzes research papers to extract methods, datasets, and other structured information"""
    
    def __init__(self, llm=None):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=PaperAnalysis)
        self._setup_prompts()
        self._compile_patterns()
    
    def _setup_prompts(self):
        """Setup prompts for different analysis tasks"""
        
        # Main analysis prompt
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert research paper analyzer. Extract detailed information from the given paper.
            
            Focus on identifying:
            1. METHODS: Algorithms, techniques, approaches, models used
            2. DATASETS: Named datasets, data sources, benchmarks used
            3. KEYWORDS: Important technical terms, concepts, abbreviations
            4. DOMAIN: Primary research field (ML, NLP, CV, etc.)
            5. METHODOLOGY: Type of research methodology
            6. CONTRIBUTION: What kind of contribution this work makes
            7. STRENGTHS: What makes this work valuable
            8. LIMITATIONS: Potential weaknesses or limitations
            
            Be specific and extract actual names/terms mentioned in the text.
            
            Format your response as: {format_instructions}
            """),
            ("human", "Title: {title}\n\nAbstract: {abstract}")
        ]).partial(format_instructions=self.parser.get_format_instructions())
        
        # Methods extraction prompt
        self.methods_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Extract all methods, algorithms, techniques, and approaches mentioned in this research paper.
            Include:
            - Machine learning algorithms (e.g., "Random Forest", "BERT", "ResNet")
            - Statistical methods (e.g., "t-test", "ANOVA", "regression analysis")
            - Computational techniques (e.g., "gradient descent", "backpropagation")
            - Evaluation methods (e.g., "cross-validation", "A/B testing")
            - Any named methodological approaches
            
            Return only the method names as a JSON list.
            """),
            ("human", "Title: {title}\n\nAbstract: {abstract}")
        ])
        
        # Datasets extraction prompt
        self.datasets_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Extract all datasets, data sources, and benchmarks mentioned in this research paper.
            Include:
            - Named datasets (e.g., "ImageNet", "GLUE", "MNIST")
            - Benchmark datasets (e.g., "WMT14", "SQuAD", "COCO")
            - Data sources (e.g., "Twitter API", "Wikipedia", "USPTO patents")
            - Custom datasets created by authors
            
            Return only the dataset names as a JSON list.
            """),
            ("human", "Title: {title}\n\nAbstract: {abstract}")
        ])
    
    def _compile_patterns(self):
        """Compile regex patterns for fast extraction"""
        
        # Common method patterns
        self.method_patterns = [
            # ML algorithms
            r'\b(?:Random Forest|Support Vector Machine|SVM|Neural Network|CNN|RNN|LSTM|GRU|Transformer|BERT|GPT|ResNet|VGG|AlexNet)\b',
            r'\b(?:XGBoost|AdaBoost|Gradient Boosting|Decision Tree|K-means|DBSCAN|PCA|ICA)\b',
            r'\b(?:Linear Regression|Logistic Regression|Ridge Regression|Lasso|Elastic Net)\b',
            
            # Deep learning
            r'\b(?:Convolutional Neural Network|Recurrent Neural Network|Long Short-Term Memory|Gated Recurrent Unit)\b',
            r'\b(?:Attention Mechanism|Self-Attention|Multi-Head Attention|Cross-Attention)\b',
            r'\b(?:Encoder-Decoder|Seq2Seq|Autoencoder|Variational Autoencoder|VAE|GAN|Generative Adversarial Network)\b',
            
            # NLP methods
            r'\b(?:Word2Vec|GloVe|FastText|TF-IDF|Bag of Words|N-gram|Named Entity Recognition|NER)\b',
            r'\b(?:Part-of-Speech Tagging|POS|Dependency Parsing|Constituency Parsing|Sentiment Analysis)\b',
            
            # Computer Vision
            r'\b(?:Object Detection|Image Classification|Semantic Segmentation|Instance Segmentation)\b',
            r'\b(?:YOLO|R-CNN|Fast R-CNN|Faster R-CNN|Mask R-CNN|U-Net|DeepLab)\b',
            
            # Statistical methods
            r'\b(?:t-test|ANOVA|Chi-square|Pearson Correlation|Spearman Correlation|Mann-Whitney U)\b',
            r'\b(?:Cross-validation|Bootstrap|Permutation Test|Hypothesis Testing)\b',
        ]
        
        # Common dataset patterns
        self.dataset_patterns = [
            # Vision datasets
            r'\b(?:ImageNet|CIFAR-10|CIFAR-100|MNIST|Fashion-MNIST|COCO|Pascal VOC|Open Images)\b',
            r'\b(?:CelebA|LFW|UTKFace|MS-CELEB-1M|VGGFace|Places365)\b',
            
            # NLP datasets
            r'\b(?:GLUE|SuperGLUE|SQuAD|CoQA|QuAC|Natural Questions|MS MARCO)\b',
            r'\b(?:WMT14|WMT15|WMT16|WMT17|WMT18|WMT19|WMT20|BLEU|ROUGE)\b',
            r'\b(?:Penn Treebank|WSJ|CoNLL|IMDB|Yelp|Amazon Reviews|Stanford Sentiment)\b',
            
            # Speech datasets
            r'\b(?:LibriSpeech|TIMIT|VCTK|Common Voice|VoxCeleb|WSJ0|CHIME)\b',
            
            # Medical datasets
            r'\b(?:MIMIC|ChestX-ray14|NIH|ISIC|Skin Cancer|COVID-19|RadImageNet)\b',
            
            # General
            r'\b(?:UCI Machine Learning Repository|UCI|Kaggle|OpenML|Papers With Code)\b',
        ]
        
        # Compile patterns
        self.compiled_method_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.method_patterns]
        self.compiled_dataset_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.dataset_patterns]
    
    async def extract_methods_datasets(self, title: str, abstract: str) -> Dict[str, List[str]]:
        """Extract methods and datasets using both regex and LLM"""
        try:
            # Combine title and abstract
            text = f"{title} {abstract}"
            
            # Regex-based extraction (fast)
            regex_methods = self._extract_with_regex(text, self.compiled_method_patterns)
            regex_datasets = self._extract_with_regex(text, self.compiled_dataset_patterns)
            
            # LLM-based extraction (more comprehensive)
            llm_methods, llm_datasets = [], []
            if self.llm:
                try:
                    # Extract methods
                    methods_chain = self.methods_prompt | self.llm
                    methods_response = await methods_chain.ainvoke({"title": title, "abstract": abstract})
                    llm_methods = self._parse_json_list(methods_response.content)
                    
                    # Extract datasets
                    datasets_chain = self.datasets_prompt | self.llm
                    datasets_response = await datasets_chain.ainvoke({"title": title, "abstract": abstract})
                    llm_datasets = self._parse_json_list(datasets_response.content)
                    
                except Exception as e:
                    logger.warning(f"LLM extraction failed: {e}")
            
            # Combine and deduplicate results
            all_methods = list(set(regex_methods + llm_methods))
            all_datasets = list(set(regex_datasets + llm_datasets))
            
            # Extract keywords using simple heuristics
            keywords = self._extract_keywords(text)
            
            return {
                "methods": all_methods,
                "datasets": all_datasets,
                "keywords": keywords
            }
            
        except Exception as e:
            logger.error(f"Error extracting methods and datasets: {e}")
            return {"methods": [], "datasets": [], "keywords": []}
    
    def _extract_with_regex(self, text: str, patterns: List[re.Pattern]) -> List[str]:
        """Extract terms using regex patterns"""
        found_terms = set()
        
        for pattern in patterns:
            matches = pattern.findall(text)
            for match in matches:
                if match:
                    found_terms.add(match.strip())
        
        return list(found_terms)
    
    def _parse_json_list(self, response: str) -> List[str]:
        """Parse JSON list from LLM response"""
        try:
            # Try to find JSON array in response
            import json
            
            # Look for array pattern
            array_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if array_match:
                json_str = array_match.group()
                return json.loads(json_str)
            
            # Fallback: split by lines and clean
            lines = response.strip().split('\n')
            items = []
            for line in lines:
                line = line.strip().strip('"-•*').strip()
                if line and not line.startswith('[') and not line.startswith(']'):
                    items.append(line)
            
            return items[:10]  # Limit to 10 items
            
        except Exception as e:
            logger.warning(f"Error parsing JSON list: {e}")
            return []
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords using simple heuristics"""
        # Common technical terms patterns
        keyword_patterns = [
            r'\b[A-Z]{2,}(?:-[A-Z]{2,})*\b',  # Acronyms
            r'\b(?:deep|machine|artificial|neural|learning|intelligence|algorithm|model|network|system)\s+\w+\b',
            r'\b\w+(?:Net|CNN|RNN|LSTM|GRU|Transformer|BERT|GPT)\b',
            r'\b(?:supervised|unsupervised|semi-supervised|self-supervised|reinforcement)\s+learning\b',
            r'\b(?:natural language|computer vision|speech recognition|image classification)\b'
        ]
        
        keywords = set()
        for pattern in keyword_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                keywords.add(match.strip())
        
        # Remove common stop words and keep only meaningful terms
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        filtered_keywords = [kw for kw in keywords if kw.lower() not in stop_words and len(kw) > 2]
        
        return filtered_keywords[:15]  # Limit to 15 keywords
    
    async def comprehensive_analysis(self, title: str, abstract: str) -> Dict[str, Any]:
        """Perform comprehensive analysis using LLM"""
        try:
            if not self.llm:
                # Fallback to basic extraction
                basic = await self.extract_methods_datasets(title, abstract)
                return {
                    **basic,
                    "domain": "Unknown",
                    "methodology_type": "Unknown",
                    "contribution_type": "Unknown",
                    "strengths": [],
                    "limitations": []
                }
            
            # Use LLM for comprehensive analysis
            chain = self.analysis_prompt | self.llm | self.parser
            result = await chain.ainvoke({"title": title, "abstract": abstract})
            
            return result.dict()
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            # Fallback to basic extraction
            basic = await self.extract_methods_datasets(title, abstract)
            return {
                **basic,
                "domain": "Unknown",
                "methodology_type": "Unknown",
                "contribution_type": "Unknown",
                "strengths": [],
                "limitations": []
            }
    
    def analyze_research_trends(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends across multiple papers"""
        try:
            # Aggregate methods and datasets
            all_methods = []
            all_datasets = []
            all_domains = []
            
            for paper in papers:
                all_methods.extend(paper.get('methods', []))
                all_datasets.extend(paper.get('datasets', []))
                all_domains.append(paper.get('domain', 'Unknown'))
            
            # Count frequencies
            from collections import Counter
            
            method_counts = Counter(all_methods)
            dataset_counts = Counter(all_datasets)
            domain_counts = Counter(all_domains)
            
            return {
                "total_papers": len(papers),
                "top_methods": dict(method_counts.most_common(10)),
                "top_datasets": dict(dataset_counts.most_common(10)),
                "domain_distribution": dict(domain_counts),
                "unique_methods": len(method_counts),
                "unique_datasets": len(dataset_counts)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing research trends: {e}")
            return {}
    
    def compare_methodologies(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare methodologies across papers"""
        try:
            comparison = {
                "papers": [paper.get('title', 'Unknown') for paper in papers],
                "methodology_overlap": {},
                "unique_contributions": {},
                "common_datasets": []
            }
            
            # Find overlapping methods
            method_sets = [set(paper.get('methods', [])) for paper in papers]
            if len(method_sets) >= 2:
                common_methods = set.intersection(*method_sets)
                comparison["common_methods"] = list(common_methods)
                
                for i, methods in enumerate(method_sets):
                    unique_methods = methods - common_methods
                    comparison["unique_contributions"][f"paper_{i+1}"] = list(unique_methods)
            
            # Find common datasets
            dataset_sets = [set(paper.get('datasets', [])) for paper in papers]
            if len(dataset_sets) >= 2:
                common_datasets = set.intersection(*dataset_sets)
                comparison["common_datasets"] = list(common_datasets)
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing methodologies: {e}")
            return {}