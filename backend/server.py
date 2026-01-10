import os
import time
import sqlite3
import asyncio
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi import UploadFile, File
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi.responses import StreamingResponse
from langchain_community.tools.tavily_search import TavilySearchResults
from datetime import datetime
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
#fast_api_launcher = FastAPI(title="AI Aggregator API")
app = FastAPI(title="AI Aggregator API")    







current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
db_path = os.path.join(parent_dir, "db_storage")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
web_search_tool = TavilySearchResults(k=3)


def get_web_context(question: str):
    try:
        results = web_search_tool.invoke({"query": question})
        web_context = "\n---\n".join([f"Source: {r['url']}\nContent: {r['content']}" for r in results])
        return web_context
    except Exception as e:
        print(f"Search error: {e}")
        return ""

class QuestionRequest(BaseModel):
    question: str
    model1: Optional[str] = "meta-llama/llama-3.3-70b-instruct:free"
    model2: Optional[str] = "google/gemma-3-27b-it:free"
    use_judge: Optional[bool] = True

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        loader = PyPDFLoader(temp_path)
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(pages)
        vector_db.add_documents(chunks)
        
        return {"status": "success", "message": f"File {file.filename} addet to knowledge base!"}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def get_context(question: str):
    docs = vector_db.similarity_search(question, k=3)
    context = "\n---\n".join([doc.page_content for doc in docs])
    return context

async def get_langchain_answer(model_name, user_text, chat_history=None, context=None):
    current_date = datetime.now().strftime("%d %B %Y")
    start_time = time.time()
    llm = ChatOpenAI(
        model = model_name,
        openai_api_key = API_KEY,
        base_url = "https://openrouter.ai/api/v1",
        max_tokens = 1000
    )
    #context = context
    if chat_history is None:
        chat_history = []
    #prompt = ChatPromptTemplate("{topic}")
    
    active_context = context if (context and context.strip() != "") else "No specific context from PDF or Web was found."
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a high-level AI Assistant. Today is {current_date}.
        
        INSTRUCTIONS:
        1. Use the [CONTEXT] section below to answer. It contains data from user files and the internet.
        2. If the [CONTEXT] is unrelated to the question, answer using your general knowledge.
        3. Never say you don't have internet access, as relevant web data is already provided in the context.
        4. If the user asks about personal data or files, use ONLY [PDF DATA]. Do not mix it with internet news unless explicitly asked.
        5. DO NOT hallucinate connections between PDF data and Web news if they are unrelated.
        
        [CONTEXT]:
        {active_context}"""),
        ("placeholder", "{chat_history}"),
        ("human", "{user_input}")
    ])

    chain = prompt | llm | StrOutputParser()
    result = await chain.ainvoke({
        "chat_history": chat_history or [],
        "user_input": user_text
    })

    duration = time.time() - start_time
    return result, duration



async def judge_answers(user_text, answer1, answer2, model):
    if answer1[:100] == answer2[:100]:
        return answer1, 0.1
    start_time = time.time()
    llm = ChatOpenAI(
        model=model,
        openai_api_key=API_KEY,
        base_url="https://openrouter.ai/api/v1",
        max_tokens=1000  
    )
    judge_template = """
    You are an expert who analyzes AI answers.

    Question: {question}

    ANSWER 1:
    {answer1}

    ANSWER 2:
    {answer2}

    Task:
    Using the two answers above, produce one single, high‑quality and concise final answer. Do not explain your reasoning, do not compare the answers, and do not mention which parts you selected. Simply provide the best possible final answer.

    FINAL ANSWER:
    """
    
    prompt = ChatPromptTemplate.from_template(judge_template)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    result = await chain.ainvoke({
        "question": user_text,
        "answer1": answer1,
        "answer2": answer2
    })

    duration = time.time() - start_time
    return result, duration



@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:

        pdf_task = asyncio.to_thread(get_context, request.question)
        web_task = asyncio.to_thread(get_web_context, request.question)
        pdf_context, web_context = await asyncio.gather(pdf_task, web_task)
        full_context = f"--- PDF DATA ---\n{pdf_context}\n\n--- WEB DATA ---\n{web_context}"

        models = [
            "meta-llama/llama-3.3-70b-instruct:free",
            "google/gemma-3-27b-it:free"
        ]
        context = get_context(request.question)
        #print(f"Claimed context: {context[:200]}...")
        task1 = get_langchain_answer(models[0], request.question, context=full_context)
        task2 = get_langchain_answer(models[1], request.question, context=full_context)
        (answer1, duration1), (answer2, duration2) = await asyncio.gather(task1, task2)
        final_answer, judge_duration = await judge_answers(request.question, answer1, answer2, "mistralai/devstral-2512:free")

        conn = sqlite3.connect('my_aggregator.db')
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO requests_history (question, model_name, answer, duration)
            VALUES (?, ?, ?, ?)
        ''', (request.question, models[0], answer1, duration1))

        cursor.execute('''
            INSERT INTO requests_history (question, model_name, answer, duration)
            VALUES (?, ?, ?, ?)
        ''', (request.question, models[1], answer2, duration2))
        
        cursor.execute('''
            INSERT INTO requests_history (question, model_name, answer, duration)
            VALUES (?, ?, ?, ?)
        ''', (request.question, "judge", final_answer, judge_duration))
        
        conn.commit()
        conn.close()
        
        return {
            "status": "success",
            "question": request.question,
            "model1": {
                "name": models[0],
                "answer": answer1,
                "duration": f"{duration1:.2f}s"
            },
            "model2": {
                "name": models[1],
                "answer": answer2,
                "duration": f"{duration2:.2f}s"
            },
            "final_answer": final_answer,
            "total_duration": f"{duration1 + duration2 + judge_duration:.2f}s"
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }
    


@app.get("/stats")
def get_stats():
    conn = sqlite3.connect('my_aggregator.db')
    cursor = conn.cursor()

    cursor.execute('''
        SELECT 
            model_name,
            COUNT(*) as request_count,
            AVG(duration) as avg_duration,
            MIN(duration) as min_duration,
            MAX(duration) as max_duration
        FROM requests_history
        GROUP BY model_name
        ORDER BY request_count DESC
    ''')

    stats = cursor.fetchall()
    conn.close()
    result = []
    for row in stats:
        result.append({
            "model": row[0],
            "request_count": row[1],
            "avg_duration": f"{row[2]:.2f}s" if row[2] else "0.00s",
            "min_duration": f"{row[3]:.2f}s" if row[3] else "0.00s",
            "max_duration": f"{row[4]:.2f}s" if row[4] else "0.00s"
        })

    return {
        "statistics": result,
        "total_requests": sum(row[1] for row in stats) if stats else 0
    }



@app.get("/history")
def get_history(limit: int = 10):
    conn = sqlite3.connect('my_aggregator.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT timestamp, question, model_name, duration
        FROM requests_history
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (limit,))
    
    history = cursor.fetchall()
    conn.close()

    return {
        "history": [
            {
                "timestamp": row[0],
                "question": row[1],
                "model": row[2],
                "duration": f"{row[3]:.2f}s" if row[3] else "0.00s"
            }
            for row in history
        ]
    }



@app.delete("/clear")
def clear_history():
    conn = sqlite3.connect('my_aggregator.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM requests_history')
    conn.commit()
    conn.close()
    
    return {"message": "History was cleared"}



@app.get("/info")
def get_info():
    return {
        "name": "AI Aggregator API",
        "version": "1.0",
        "available_models": [
            "meta-llama/llama-3.3-70b-instruct:free",
            "google/gemma-3-27b-it:free",
            "mistralai/devstral-2512:free (judge)"
        ],
        "features": [
            "Parallel queries to two models",
            "Automatic response validation",
            "Storage in SQLite database",
            "Detailed statistics"
        ]
    }



def init_db():
    conn = sqlite3.connect('my_aggregator.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS requests_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            question TEXT,
            model_name TEXT,
            answer TEXT,
            duration REAL
        )
    ''')
    conn.commit()
    conn.close()
    print("Datebase is ready to work")
init_db()