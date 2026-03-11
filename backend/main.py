"""
FastAPI server for Prepd question search API.
Uses zvec for vector similarity search.
"""

import zvec
import json
import os
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Paths
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "questions_db"

# OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = "text-embedding-3-small"

# Global collection reference
collection = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage zvec collection lifecycle."""
    global collection
    print(f"Opening zvec collection at {DB_PATH}...")
    collection = zvec.open(path=str(DB_PATH))
    yield
    print("Shutting down...")


app = FastAPI(
    title="Prepd Question API",
    description="Semantic search for math questions",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class SearchRequest(BaseModel):
    query: str
    topk: int = 10
    subject: Optional[str] = None
    topic: Optional[str] = None
    difficulty: Optional[str] = None
    bloom_level: Optional[str] = None


class Question(BaseModel):
    id: str
    question_text: str
    question_type: str
    options: dict
    correct_answer: str
    explanation: str
    subject: str
    topic: str
    subtopic: str
    difficulty: str
    bloom_level: str
    estimated_time_seconds: int
    score: float


class SearchResponse(BaseModel):
    questions: list[Question]
    total: int


def get_embedding(text: str) -> list[float]:
    """Get embedding for a single text."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text]
    )
    return response.data[0].embedding


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "prepd-api"}


@app.post("/search", response_model=SearchResponse)
async def search_questions(request: SearchRequest):
    """
    Search questions by semantic similarity.

    Optionally filter by subject, topic, difficulty, or bloom_level.
    """
    if not collection:
        raise HTTPException(status_code=503, detail="Database not ready")

    # Get query embedding
    query_embedding = get_embedding(request.query)

    # Build filter if any filters specified
    filters = []
    if request.subject:
        filters.append(f"subject = '{request.subject}'")
    if request.topic:
        filters.append(f"topic = '{request.topic}'")
    if request.difficulty:
        filters.append(f"difficulty = '{request.difficulty}'")
    if request.bloom_level:
        filters.append(f"bloom_level = '{request.bloom_level}'")

    filter_expr = " AND ".join(filters) if filters else None

    # Query zvec
    results = collection.query(
        zvec.VectorQuery("embedding", vector=query_embedding),
        topk=request.topk,
        filter=filter_expr
    )

    # Parse results (zvec returns Doc objects)
    questions = []
    for result in results:
        payload = json.loads(result.fields["payload"])
        questions.append(Question(
            id=payload["id"],
            question_text=payload["question_text"],
            question_type=payload["question_type"],
            options=payload["options"],
            correct_answer=payload["correct_answer"],
            explanation=payload["explanation"],
            subject=payload["subject"],
            topic=payload["topic"],
            subtopic=payload["subtopic"],
            difficulty=payload["difficulty"],
            bloom_level=payload["bloom_level"],
            estimated_time_seconds=payload["estimated_time_seconds"],
            score=result.score
        ))

    return SearchResponse(questions=questions, total=len(questions))


@app.get("/search")
async def search_questions_get(
    q: str = Query(..., description="Search query"),
    topk: int = Query(10, description="Number of results"),
    subject: Optional[str] = Query(None),
    topic: Optional[str] = Query(None),
    difficulty: Optional[str] = Query(None),
    bloom_level: Optional[str] = Query(None)
):
    """GET endpoint for search (convenience)."""
    return await search_questions(SearchRequest(
        query=q,
        topk=topk,
        subject=subject,
        topic=topic,
        difficulty=difficulty,
        bloom_level=bloom_level
    ))


@app.get("/question/{question_id}")
async def get_question(question_id: str):
    """Get a specific question by ID."""
    if not collection:
        raise HTTPException(status_code=503, detail="Database not ready")

    # For now, load from JSONL (zvec doesn't have direct ID lookup)
    questions_file = DATA_DIR / "questions.jsonl"
    for line in open(questions_file):
        q = json.loads(line)
        if q["id"] == question_id:
            return q

    raise HTTPException(status_code=404, detail="Question not found")


@app.get("/subjects")
async def list_subjects():
    """List available subjects."""
    return {
        "subjects": [
            "Algebra",
            "Geometry",
            "Trigonometry",
            "Statistics & Probability",
            "Calculus (Pre-Calc)",
            "Number Theory"
        ]
    }


@app.get("/difficulties")
async def list_difficulties():
    """List available difficulty levels."""
    return {"difficulties": ["easy", "medium", "hard"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
