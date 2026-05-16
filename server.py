# server.py — FastAPI backend for the chatbot
import chromadb
import ollama
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

DB_DIR = "chroma_db"
COLLECTION = "about_me"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:latest"
TOP_K = 4

SYSTEM_PROMPT = """You are a helpful assistant on Sabbir's personal website.
You answer questions about Sabbir based ONLY on the context provided below.
If the answer isn't in the context, say you don't know — don't make things up.
Keep answers concise and conversational.

Context about Sabbir:
{context}
"""

# Load Chroma once at startup, not on every request
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_collection(COLLECTION)

app = FastAPI()

# Allow the frontend to call this backend from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

def retrieve(question: str):
    response = ollama.embeddings(model=EMBED_MODEL, prompt=question)
    results = collection.query(
        query_embeddings=[response["embedding"]],
        n_results=TOP_K,
    )
    return results["documents"][0]

def stream_answer(question: str):
    chunks = retrieve(question)
    context = "\n\n---\n\n".join(chunks)
    system = SYSTEM_PROMPT.format(context=context)

    stream = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        stream=True,
    )
    for part in stream:
        token = part["message"]["content"]
        if token:
            yield token

@app.post("/chat")
def chat(req: ChatRequest):
    return StreamingResponse(stream_answer(req.message), media_type="text/plain")

@app.get("/")
def index():
    return FileResponse("frontend.html")