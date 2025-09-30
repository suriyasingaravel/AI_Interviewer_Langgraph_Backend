import os
import json
import uuid
from typing import List, Dict, Optional, Literal, TypedDict
import re



from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()




app = FastAPI(title="AI Interviewer Backend", version="0.1.0")


@app.get("/")
async def root():
    return {"message": "Hello from AI Interviewer Backend!"}


# ---------------------------
# LLM Setup (use env var OPENAI_API_KEY)
# ---------------------------
OPENAI_API_KEY = os.getenv("GEMINI_API_KEY")


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, api_key=OPENAI_API_KEY)
parser = StrOutputParser()


# ---------------------------
# Prompts
# ---------------------------
GENERATE_QUESTIONS_PROMPT = ChatPromptTemplate.from_template("""
You are an expert technical interviewer. Generate exactly 5 concise, role-specific technical questions.

Return ONLY a JSON array of 5 strings. No keys, no comments, no markdown, no code fences.
Example: ["Q1","Q2","Q3","Q4","Q5"]

Job Role: {job_role}
Experience Level: {experience} years
""")

EVALUATE_ANSWER_PROMPT = ChatPromptTemplate.from_template("""
You are an expert technical interviewer evaluating a candidate's answer.


Job Role: {job_role}
Experience Level: {experience} years
Question: {question}
Candidate's Answer: {answer}


Provide a detailed evaluation including:
1. Technical accuracy (1-10)
2. Completeness of answer (1-10)
3. Clarity of explanation (1-10)
4. Specific feedback and suggestions
5. Overall score (1-10)


Format your response as:
Score: X/10
Technical Accuracy: X/10
Completeness: X/10
Clarity: X/10
Feedback: [Your detailed feedback here]
""")


FINAL_REPORT_PROMPT = ChatPromptTemplate.from_template("""
You are an expert technical interviewer creating a comprehensive interview report.


Job Role: {job_role}
Experience Level: {experience} years


Interview Results:
{interview_results}


Create a professional final report including:
1. Overall assessment
2. Strengths identified
3. Areas for improvement
4. Technical competency score (average of all scores)
5. Recommendation (Pass/Fail/Consider with conditions)
6. Detailed breakdown of each question


Format as a professional report.
""")

def _parse_json_array(text: str) -> list[str]:
    # common cases: ```json ... ``` or prose + array
    if not isinstance(text, str):
        raise ValueError("LLM returned a non-string response.")

    s = text.strip()

    # remove code fences
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.DOTALL).strip()

    # try direct parse
    try:
        val = json.loads(s)
    except json.JSONDecodeError:
        # extract first JSON array
        m = re.search(r"\[\s*(?:\".*?\"|\[.*?\]|\{.*?\}|[^]]|\n)*\]", s, flags=re.DOTALL)
        if not m:
            raise
        val = json.loads(m.group(0))

    if not isinstance(val, list) or not all(isinstance(x, str) for x in val):
        raise ValueError("Expected a JSON array of strings from LLM.")
    return val


# ---------------------------
# LangGraph State Definition
# ---------------------------
class InterviewQA(TypedDict):
    question: str
    answer: str
    feedback: str


class InterviewState(TypedDict):
    job_role: str
    experience: int
    data: List[InterviewQA]
    current_question_idx: int
    interview_complete: bool
    final_report: str
    last_question: str
    last_answer: Optional[str]  # For API input


# ---------------------------
# LangGraph Nodes
# ---------------------------
def get_job_role_and_experience(state: InterviewState) -> InterviewState:
    """Validate job role and experience are provided."""
    if not state.get("job_role") or state.get("experience") is None:
        raise ValueError("job_role and experience must be provided.")
    return state


def generate_questions(state: InterviewState) -> InterviewState:
    """Generate interview questions based on job role and experience."""
    # Use the JSON-mode LLM here
    chain = GENERATE_QUESTIONS_PROMPT | llm | parser
    questions_raw = chain.invoke({
        "job_role": state["job_role"],
        "experience": state["experience"]
    })  # this should already be JSON text, but we still harden

    try:
        questions = _parse_json_array(questions_raw)
    except Exception as e:
        # Give a helpful error so the API doesn't 500 mysteriously
        raise ValueError(f"Failed to parse questions from LLM: {e}")

    # Enforce exactly 5 unique, non-empty questions
    questions = [q.strip() for q in questions if q and q.strip()]
    if len(questions) != 5:
        # attempt to trim or pad with simple variations if you'd rather not hard-error
        raise ValueError(f"Expected 5 questions, got {len(questions)}.")

    question_list: List[InterviewQA] = [{"question": q, "answer": "", "feedback": ""} for q in questions]

    return {
        **state,
        "data": question_list,
        "current_question_idx": 0,
        "interview_complete": False,
        "final_report": "",
        "last_question": question_list[0]["question"] if question_list else ""
    }

def ask_question(state: InterviewState) -> InterviewState:
    """Get the current question."""
    idx = state.get("current_question_idx", 0)
    if idx >= 5:
        return state
    q_text = state["data"][idx]["question"]
    return {**state, "last_question": q_text}


def evaluate_answer(state: InterviewState) -> InterviewState:
    """Evaluate the candidate's answer and generate feedback."""
    idx = state.get("current_question_idx", 0)
    if idx >= 5:
        return state
    
    # Check if we have an answer to evaluate
    last_answer = state.get("last_answer")
    if not last_answer:
        # No answer provided, just return current state
        return state
    
    question_text = state["data"][idx]["question"]
    answer_text = last_answer.strip()
    
    if not answer_text:
        raise ValueError("No answer provided for current question.")


    chain = EVALUATE_ANSWER_PROMPT | llm | parser
    feedback = chain.invoke({
        "job_role": state["job_role"],
        "experience": state["experience"],
        "question": question_text,
        "answer": answer_text
    })
    
    data_copy = state["data"].copy()
    data_copy[idx]["answer"] = answer_text
    data_copy[idx]["feedback"] = feedback


    new_idx = idx + 1
    next_question = data_copy[new_idx]["question"] if new_idx < 5 else ""
    
    return {
        **state,
        "data": data_copy,
        "current_question_idx": new_idx,
        "last_question": next_question,
        "last_answer": None  # Clear the answer after processing
    }


def generate_final_report(state: InterviewState) -> InterviewState:
    """Generate the final interview report."""
    interview_results = ""
    for i in range(5):
        qd = state["data"][i]
        interview_results += f"\nQuestion {i+1}: {qd['question']}\n"
        interview_results += f"Answer: {qd['answer']}\n"
        interview_results += f"Feedback: {qd['feedback']}\n"
        interview_results += "-" * 40 + "\n"


    chain = FINAL_REPORT_PROMPT | llm | parser
    final_report = chain.invoke({
        "job_role": state["job_role"],
        "experience": state["experience"],
        "interview_results": interview_results
    })
    
    return {
        **state, 
        "final_report": final_report, 
        "interview_complete": True
    }


def should_continue_interview(state: InterviewState) -> Literal["continue", "complete"]:
    """Determine if interview should continue or complete."""
    current_idx = state.get("current_question_idx", 0)
    return "continue" if current_idx < 5 else "complete"


# ---------------------------
# LangGraph Workflow Setup
# ---------------------------
workflow = StateGraph(InterviewState)
workflow.add_node("get_job_role_and_experience", get_job_role_and_experience)
workflow.add_node("generate_questions", generate_questions)
workflow.add_node("ask_question", ask_question)
workflow.add_node("evaluate_answer", evaluate_answer)
workflow.add_node("generate_final_report", generate_final_report)


# Simple workflow: just generate questions and ask first question
workflow.add_edge(START, "get_job_role_and_experience")
workflow.add_edge("get_job_role_and_experience", "generate_questions")
workflow.add_edge("generate_questions", "ask_question")
workflow.add_edge("ask_question", END)


# Compile with memory checkpointer for session persistence
checkpointer = MemorySaver()
compiled_graph = workflow.compile(checkpointer=checkpointer)

class CreateSessionRequest(BaseModel):
    job_role:str= Field(..., example="React Developer")
    experience: int = Field(..., example = 2)

class CreateSessionResponse(BaseModel):
    session_id: str
    job_role:str
    experience:int
    questions:List[str]
    current_question_idx:int




@app.post("/sessions", response_model=CreateSessionResponse, status_code=201)
async def create_session(payload: CreateSessionRequest):
    """Create a new interview session using LangGraph workflow."""
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    
    # Initial state for the workflow
    initial_state: InterviewState = {
        "job_role": payload.job_role,
        "experience": payload.experience,
        "data": [],
        "current_question_idx": 0,
        "interview_complete": False,
        "final_report": "",
        "last_question": "",
        "last_answer": None
    }
    
    # Run the workflow up to ask_question
    result = compiled_graph.invoke(initial_state, config=config)
    
    # Get the current state
    state = compiled_graph.get_state(config)
    print(state)
    if not state or not state.values:
        raise HTTPException(status_code=500, detail="Failed to initialize session")
    
    values = state.values
    questions = [row["question"] for row in values.get("data", [])]
    
    return CreateSessionResponse(
        session_id=session_id,
        job_role=values["job_role"],
        experience=values["experience"],
        questions=questions,
        current_question_idx=values.get("current_question_idx", 0),
    )