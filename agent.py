from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Literal,List
from typing_extensions import TypedDict
from database import DatabaseManager
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph, START
from langchain_core.output_parsers import StrOutputParser
from ratelimit import limits, sleep_and_retry
from tenacity import retry, wait_exponential, stop_after_attempt


load_dotenv()
os.environ['GOOGLE_API_KEY']=os.getenv('GOOGLE_API_KEY')
os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")

# Rate limiting constants
ONE_MINUTE = 60
MAX_CALLS_PER_MINUTE = 15

class Agent:
    def __init__(self,username):
        self.embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vector_store = FAISS.load_local('faiss_index', self.embedding, allow_dangerous_deserialization=True,normalize_L2=True)
        self.retriever=self.vector_store.as_retriever()
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", 
                                          temperature=0,
                                          retry_on_failure=True,
                                          retry_on_quota=True,
                                          max_retries=3,
                                          initial_delay=2,  # Start with 2 second delay
                                          exponential_base=2,  # Double the delay with each retry
                                          max_delay=10  # Maximum delay between retries
                                          )
        self.web_search_tool = TavilySearchResults(k=3)
        self.app = self._build_workflow()
        self.db = DatabaseManager()
        self.username = username
        
    def _load_chat_history(self,session_id,max_messages=5):
        """Load chat history from database for current session"""
        messages = self.db.get_chat_messages(session_id)
        if not messages:
            return []
        # Use the most recent messages up to `max_messages`
        context = messages[-max_messages:-1]
        return [{"user": msg["username"], "message": msg["message"]} 
                   for msg in context]
        
    def _build_workflow(self):
        class GraphState(TypedDict):
            """
            Represents the state of our graph.

            Attributes:
                question: question
                generation: LLM generation
                documents: list of documents
                history: conversation history
                session_id: chat session id
            """
            question: str
            generation: str
            documents: List[str]
            history: List[dict]
            session_id: str
            
        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("build_query", self._build_query)
        workflow.add_node("web_search", self._web_search)  # web search
        workflow.add_node("call_llm",self._call_llm)
        workflow.add_node("retrieve", self._retrieve)  # retrieve
        workflow.add_node("grade_documents", self._grade_documents)  # grade documents
        workflow.add_node("generate", self._generate)  # generatae
        workflow.add_node("transform_query", self._transform_query)  # transform_query

        # Build graph
        workflow.add_edge(START,"build_query")
        workflow.add_conditional_edges(
            "build_query",
            self._route_question,
            {
                "web-search": "web_search",
                "vectorstore": "retrieve",
                "llm": "call_llm"
            },
        )
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self._decide_to_generate,
            {
                "no relevant document": "transform_query",
                "found relevant document": "generate",
            },
        )
        # workflow.add_edge("transform_query", "retrieve")
        workflow.add_conditional_edges(
            "transform_query",
            self._route_question,
            {
                "web-search": "web_search",
                "vectorstore": "retrieve",
                "llm": "call_llm"
            },
        )
        workflow.add_conditional_edges(
            "generate",
            self._grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "transform_query",
            },
        )
        workflow.add_edge("call_llm",END)
        app = workflow.compile()
        return app
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=20),
        stop=stop_after_attempt(5)
    )
    def _build_query(self,state):
        """
        Transform the query to produce a context aware question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """
        
        # Prompt
        prompt = """Task:
        Rewrite the given query to make it clear and precise. Use the conversation history if the query depends on prior context.

        Instructions:

        Analyze the query to determine if it refers to previous context.
        If context is required:
        Refer to the conversation history to resolve ambiguities and rewrite the query.
        If the query is standalone and no need for improvememt, return the query as-is.
        
        Inputs:
        Original Query: {question}
        Conversation History: {history} (formatted as username:message)

        """
        query_build_prompt = ChatPromptTemplate.from_template(prompt)

        question_build = query_build_prompt | self.llm | StrOutputParser()
        
        print("---BUILD QUERY---")
        question = state["question"]
        session_id = state.get("session_id")
        history = self._load_chat_history(session_id)  if session_id else []

        # Re-write question
        better_question = question_build.invoke({"question": question,"history":history})
        return {"history": history, "question": better_question, "session_id": session_id}
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=20),
        stop=stop_after_attempt(5)
    )        
    def _route_question(self,state):
        """
        Route question to web search or RAG or llm.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """
        # Data model
        class RouteQuery(BaseModel):
            """Route a user query to the most relevant datasource."""

            datasource: Literal["vectorstore", "web-search","llm"] = Field(
                ...,
                description="Given a user question choose to route it to web search or a vectorstore or llm.",
            )

        # LLM with function call
        structured_llm_router = self.llm.with_structured_output(RouteQuery)

        # Prompt
        system = """You are an expert at routing a user question to a vectorstore or web search or llm.
        The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
        Use the vectorstore for questions on these topics. Otherwise, use web-search.
        For generic query/conversation choose llm"""
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{question}"),
            ]
        )

        question_router = route_prompt | structured_llm_router
        
        print("---ROUTE QUESTION---")
        question = state["question"]
        source = question_router.invoke({"question": question})
        print(source.datasource)
        if source.datasource == "web-search":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "web-search"
        elif source.datasource == "vectorstore":
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"
        else:
            print("---ROUTE QUESTION TO LLM---")
            return "llm"
    def _web_search(self,state):
        """
        Web search based on the re-phrased question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """
        print("---WEB SEARCH---")
        question = state["question"]

        # Web search
        docs = self.web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        print(web_results)
        return {"documents": web_results, "question": question}
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=20),
        stop=stop_after_attempt(5)
    )
    def _call_llm(self,state):
        """
        Call LLM for generic conversation.
        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates generation key with llm results
        """
        
        print("---LLM---")
        question = state["question"]
        
        prompt_template = f"""
        Answer the question as detailed as possible based on your knowledge , make sure to provide all the details,
        if you don't known the answer just say, 'answer is not available in my knowledge base', don't provide
        the wrong answer. If it is generic text, answer accordingly
        Question:\n{question}\n

        Answer:
        """
        response=self.llm.invoke(prompt_template)
        print(response.content)
        return {"documents": [], "question": question, "generation": response.content}

    def _retrieve(self,state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question}
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=20),
        stop=stop_after_attempt(5)
    )
    def _grade_documents(self,state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """
        
        # Data model
        class GradeDocuments(BaseModel):
            """Binary score for relevance check on retrieved documents."""

            binary_score: str = Field(
                description="Documents are relevant to the question, 'yes' or 'no'"
            )


        # LLM with function call
        structured_llm_grader = self.llm.with_structured_output(GradeDocuments)

        # Prompt
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )

        retrieval_grader = grade_prompt | structured_llm_grader
        
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            score = retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}

    def _decide_to_generate(self,state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        state["question"]
        filtered_documents = state["documents"]

        if not filtered_documents:
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "no relevant document"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "found relevant document"
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=20),
        stop=stop_after_attempt(5)
    )
    def _generate(self,state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        # Prompt
        prompt = """
        You are assistant is question answering task. Answer the question as detailed as possible from the provided contexts, 
        make sure to provide all the details, if the answer is not in the provided context just say, 
        'answer is not available in the context', don't provide the wrong answer.
        Context:\n{context}\n
        Question:\n{question}\n

        Answer:
        """
        prompt = ChatPromptTemplate.from_template(template=prompt)

        # Post-processing
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)


        # Chain
        rag_chain = prompt | self.llm | StrOutputParser()
        
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=20),
        stop=stop_after_attempt(5)
    )    
    def _grade_generation_v_documents_and_question(self,state):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """
        # Data model
        class GradeHallucinations(BaseModel):
            """Binary score for hallucination present in generation answer."""

            binary_score: str = Field(
                description="Answer is grounded in the facts, 'yes' or 'no'"
            )

        # LLM with function call
        structured_llm_grader = self.llm.with_structured_output(GradeHallucinations)

        # Prompt
        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
            Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
            ]
        )

        hallucination_grader = hallucination_prompt | structured_llm_grader

        # Data model
        class GradeAnswer(BaseModel):
            """Binary score to assess answer addresses question."""

            binary_score: str = Field(
                description="Answer addresses the question, 'yes' or 'no'"
            )


        # LLM with function call
        structured_llm_grader = self.llm.with_structured_output(GradeAnswer)

        # Prompt
        system = """You are a grader assessing whether an answer addresses / resolves a question \n 
            Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
        )

        answer_grader = answer_prompt | structured_llm_grader
        
        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score.binary_score
        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = answer_grader.invoke({"question": question, "generation": generation})
            grade = score.binary_score
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=20),
        stop=stop_after_attempt(5)
    )
    def _transform_query(self,state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        ### Question Re-writer

        # Prompt
        system = """You a question re-writer that converts an input question to a better version that is optimized \n 
            for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Here is the initial question: \n\n {question} \n Formulate an improved question.",
                ),
            ]
        )

        question_rewriter = re_write_prompt | self.llm | StrOutputParser()

        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}
    
    @sleep_and_retry
    @limits(calls=MAX_CALLS_PER_MINUTE, period=ONE_MINUTE)
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=20),
        stop=stop_after_attempt(5)
    )
    def generate_bot_response(self, session_id, user_message):
        """
        Generate a bot response for a given user message.
        
        Args:
            session_id (str): The current chat session ID
            user_message (str): The user's message to respond to
            
        Returns:
            str: The generated bot response
        """
        try:
            inputs = {
                "question": user_message,
                "session_id": session_id
            }
            
            final_output = None
            
            # Process all outputs from the graph
            for output in self.app.stream(inputs):
                final_output = output
                
            # Return the final generation if available
            if final_output and 'generation' in final_output.get(list(final_output.keys())[-1], {}):
                return final_output[list(final_output.keys())[-1]]['generation']
            else:
                return "I apologize, but I wasn't able to generate a response."
                
        except Exception as e:
            return f"Error generating response: {str(e)}"