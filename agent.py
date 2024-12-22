from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Literal, List, Dict, Any
from typing_extensions import TypedDict
from database import DatabaseManager
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph, START
from langchain_core.output_parsers import StrOutputParser
from abc import ABC, abstractmethod
# from IPython.display import Image, display

# Load environment variables
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")

class BaseAgent(ABC):
    """Abstract base class for all agents"""
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    @abstractmethod
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pass

class QueryAgent(BaseAgent):
    """Agent responsible for understanding and transforming queries"""
    def __init__(self, db_manager: DatabaseManager):
        super().__init__()
        self.db = db_manager

    def _load_chat_history(self, session_id, max_messages=10):
        """Load chat history from database"""
        messages = self.db.get_chat_messages(session_id)
        if not messages:
            return []
        context = messages[-max_messages:-1]
        return [{"user": msg["username"], "message": msg["message"]} 
                for msg in context]

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = """Task:
        Rewrite the given query to make it clear and precise. Use the conversation history if the query depends on prior context.

        Instructions:
        Analyze the query to determine if it refers to previous context.
        If context is required:
        Refer to the conversation history to resolve ambiguities and rewrite the query.
        If the query is standalone and no need for improvement, return the query as-is.
        
        Inputs:
        Original Query: {question}
        Conversation History: {history} (formatted as username:message)
        """
        query_build_prompt = ChatPromptTemplate.from_template(prompt)
        question_build = query_build_prompt | self.llm | StrOutputParser()
        
        question = state["question"]
        session_id = state.get("session_id")
        history = self._load_chat_history(session_id) if session_id else []
        better_question = question_build.invoke({"question": question, "history": history})
        
        return {"history": history, "question": better_question, "session_id": session_id}

class RouterAgent(BaseAgent):
    """Agent responsible for routing queries to appropriate data sources"""
    def process(self, state: Dict[str, Any]) -> str:
        class RouteQuery(BaseModel):
            datasource: Literal["vectorstore", "web-search", "llm"] = Field(
                description="Route queries to vectorstore, web search, or llm"
            )

        structured_llm_router = self.llm.with_structured_output(RouteQuery)
        system = """You are an expert at routing user questions.
        The vectorstore contains documents about agents, prompt engineering, and adversarial attacks.
        Use vectorstore for these topics. Otherwise, use web-search.
        For generic conversation choose llm"""
        
        route_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "{question}"),
        ])

        question_router = route_prompt | structured_llm_router
        source = question_router.invoke({"question": state["question"]})
        return source.datasource

class RetrievalAgent(BaseAgent):
    """Agent responsible for retrieving information from vector store"""
    def __init__(self):
        super().__init__()
        self.embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vector_store = FAISS.load_local('faiss_index', self.embedding, allow_dangerous_deserialization=True)
        self.retriever = self.vector_store.as_retriever()

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        documents = self.retriever.invoke(state["question"])
        return {"documents": documents, "question": state["question"]}

class WebSearchAgent(BaseAgent):
    """Agent responsible for web search operations"""
    def __init__(self):
        super().__init__()
        self.web_search_tool = TavilySearchResults(k=3)

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        docs = self.web_search_tool.invoke({"query": state["question"]})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        return {"documents": web_results, "question": state["question"]}
    
class DirectLLMAgent(BaseAgent):
    """Agent responsible for direct LLM interactions without retrieval"""
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        prompt_template = """
        Answer the question as detailed as possible based on your knowledge, make sure to provide all the details,
        if you don't know the answer just say, 'answer is not available in my knowledge base', don't provide
        the wrong answer. If it is generic text, answer accordingly.
        
        Question:\n{question}\n
        Answer:
        """
        
        response = self.llm.invoke(prompt_template.format(question=state["question"]))
        return {
            "documents": [],
            "question": state["question"],
            "generation": response.content
        }

class DocumentGradingAgent(BaseAgent):
    """Agent responsible for grading document relevance"""
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        class GradeDocuments(BaseModel):
            binary_score: str = Field(
                description="Documents are relevant to the question, 'yes' or 'no'"
            )

        structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
        system = """Grade the relevance of retrieved documents to the user question.
        If document contains keywords or semantic meaning related to the question, grade as relevant.
        Give a binary 'yes' or 'no' score."""
        
        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ])

        retrieval_grader = grade_prompt | structured_llm_grader
        
        filtered_docs = []
        for doc in state["documents"]:
            score = retrieval_grader.invoke({
                "question": state["question"], 
                "document": doc.page_content
            })
            if score.binary_score == "yes":
                filtered_docs.append(doc)
                
        return {"documents": filtered_docs, "question": state["question"]}

class GenerationAgent(BaseAgent):
    """Agent responsible for generating answers"""
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = """
        Answer the question as detailed as possible from the provided contexts.
        If answer is not in context say 'answer is not available in the context'.
        Don't provide wrong answers.
        
        Context:\n{context}\n
        Question:\n{question}\n
        Answer:
        """
        prompt_template = ChatPromptTemplate.from_template(prompt)
        rag_chain = prompt_template | self.llm | StrOutputParser()
        
        generation = rag_chain.invoke({
            "context": state["documents"], 
            "question": state["question"]
        })
        return {
            "documents": state["documents"],
            "question": state["question"],
            "generation": generation
        }

class QualityCheckAgent(BaseAgent):
    """Agent responsible for checking generation quality"""
    def process(self, state: Dict[str, Any]) -> str:
        # Check for hallucinations
        class GradeHallucinations(BaseModel):
            binary_score: str = Field(
                description="Answer is grounded in facts, 'yes' or 'no'"
            )

        structured_hallucination_grader = self.llm.with_structured_output(GradeHallucinations)
        hallucination_prompt = ChatPromptTemplate.from_messages([
            ("system", "Grade whether LLM generation is grounded in retrieved facts."),
            ("human", "Facts: \n\n {documents} \n\n Generation: {generation}"),
        ])
        
        hallucination_grader = hallucination_prompt | structured_hallucination_grader
        
        # Check if answer addresses question
        class GradeAnswer(BaseModel):
            binary_score: str = Field(
                description="Answer addresses question, 'yes' or 'no'"
            )

        structured_answer_grader = self.llm.with_structured_output(GradeAnswer)
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", "Grade whether answer addresses/resolves question"),
            ("human", "Question: \n\n {question} \n\n Generation: {generation}"),
        ])
        
        answer_grader = answer_prompt | structured_answer_grader

        # Process checks
        hallucination_score = hallucination_grader.invoke({
            "documents": state["documents"],
            "generation": state["generation"]
        })
        
        if hallucination_score.binary_score == "yes":
            answer_score = answer_grader.invoke({
                "question": state["question"],
                "generation": state["generation"]
            })
            return "useful" if answer_score.binary_score == "yes" else "not useful"
        return "not supported"

class QueryTransformationAgent(BaseAgent):
    """Agent responsible for transforming queries when initial search fails"""
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        system = """You are a question re-writer that converts an input question to a better version optimized \n 
            for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
        re_write_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
        ])

        question_rewriter = re_write_prompt | self.llm | StrOutputParser()
        better_question = question_rewriter.invoke({"question": state["question"]})
        
        return {
            "documents": state["documents"], 
            "question": better_question
        }

class MultiAgentSystem:
    """Orchestrates multiple agents in a workflow"""
    def __init__(self, username: str):
        self.db = DatabaseManager()
        self.username = username
        
        # Initialize agents
        self.query_agent = QueryAgent(self.db)
        self.router_agent = RouterAgent()
        self.retrieval_agent = RetrievalAgent()
        self.web_search_agent = WebSearchAgent()
        self.direct_llm_agent = DirectLLMAgent()
        self.document_grading_agent = DocumentGradingAgent()
        self.generation_agent = GenerationAgent()
        self.quality_check_agent = QualityCheckAgent()
        self.query_transformation_agent = QueryTransformationAgent()
        
        # Build workflow
        self.app = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build and return the workflow graph"""
        class GraphState(TypedDict):
            question: str
            generation: str
            documents: List[str]
            history: List[dict]
            session_id: str
            
        workflow = StateGraph(GraphState)

        # Add nodes for each agent
        workflow.add_node("build_query", self.query_agent.process)
        workflow.add_node("web_search", self.web_search_agent.process)
        workflow.add_node("call_llm", self.direct_llm_agent.process)
        workflow.add_node("retrieve", self.retrieval_agent.process)
        workflow.add_node("grade_documents", self.document_grading_agent.process)
        workflow.add_node("generate", self.generation_agent.process)
        workflow.add_node("transform_query", self.query_transformation_agent.process)

        # Helper function to determine next step after document grading
        def decide_to_generate(state):
            if not state["documents"]:
                return "no relevant document"
            return "found relevant document"

        # Define workflow edges
        workflow.add_edge(START, "build_query")
        workflow.add_conditional_edges(
            "build_query",
            self.router_agent.process,
            {
                "web-search": "web_search",
                "vectorstore": "retrieve",
                "llm": "call_llm"
            }
        )
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("retrieve", "grade_documents")
        
        # Add conditional edges for document grading outcomes
        workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {
                "no relevant document": "transform_query",
                "found relevant document": "generate",
            }
        )
        
        # Add edges for query transformation
        workflow.add_conditional_edges(
            "transform_query",
            self.router_agent.process,
            {
                "web-search": "web_search",
                "vectorstore": "retrieve",
                "llm": "call_llm"
            }
        )
        
        # Add conditional edges for generation quality check
        workflow.add_conditional_edges(
            "generate",
            self.quality_check_agent.process,
            {
                "useful": END,
                "not useful": "transform_query",
                "not supported": "generate"
            }
        )
        workflow.add_edge("call_llm",END)
        # app = workflow.compile()
        # try:
        #     # Generate the image
        #     img = Image(app.get_graph().draw_mermaid_png())
            
        #     # Display the image
        #     display(img)
            
        #     # Save the image to a file
        #     with open("graph_image.png", "wb") as file:
        #         file.write(img.data)
        # except Exception as e:
        #     # Handle exceptions if any
        #     print(f"An error occurred: {e}")
        return workflow.compile()

    def generate_bot_response(self, session_id: str, user_message: str) -> str:
        """Generate response using the multi-agent system"""
        try:
            inputs = {
                "question": user_message,
                "session_id": session_id
            }
            
            final_output = None
            for output in self.app.stream(inputs):
                final_output = output
                
            if final_output and 'generation' in final_output.get(list(final_output.keys())[-1], {}):
                return final_output[list(final_output.keys())[-1]]['generation']
            return "I apologize, but I wasn't able to generate a response."
                
        except Exception as e:
            return f"Error generating response: {str(e)}"