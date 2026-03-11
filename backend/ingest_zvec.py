"""
Ingest questions from JSONL into zvec vector database.
Uses OpenAI embeddings for vector search.
"""

import zvec
import json
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Paths
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "questions_db"
QUESTIONS_FILE = DATA_DIR / "questions.jsonl"

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Embedding config
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
BATCH_SIZE = 100  # OpenAI allows up to 2048, but smaller batches are safer


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings for a list of texts, handling batching."""
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        print(f"  Embedding batch {i // BATCH_SIZE + 1}/{(len(texts) - 1) // BATCH_SIZE + 1}...")

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch
        )
        all_embeddings.extend([d.embedding for d in response.data])

    return all_embeddings


def create_schema() -> zvec.CollectionSchema:
    """Create the zvec collection schema."""
    return zvec.CollectionSchema(
        name="questions",
        vectors=zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, EMBEDDING_DIM),
        fields=[
            zvec.FieldSchema("subject", zvec.DataType.STRING),
            zvec.FieldSchema("topic", zvec.DataType.STRING),
            zvec.FieldSchema("subtopic", zvec.DataType.STRING),
            zvec.FieldSchema("difficulty", zvec.DataType.STRING),
            zvec.FieldSchema("bloom_level", zvec.DataType.STRING),
            zvec.FieldSchema("verified", zvec.DataType.BOOL),
            zvec.FieldSchema("payload", zvec.DataType.STRING),  # Full question JSON
        ]
    )


def ingest_questions():
    """Load questions from JSONL and ingest into zvec."""

    # Load questions
    print(f"Loading questions from {QUESTIONS_FILE}...")
    questions = [json.loads(line) for line in open(QUESTIONS_FILE)]
    print(f"Loaded {len(questions)} questions")

    # Get embeddings
    print("Generating embeddings...")
    embed_texts = [q["embed_text"] for q in questions]
    embeddings = get_embeddings(embed_texts)

    # Create/open collection
    print(f"Creating zvec collection at {DB_PATH}...")
    schema = create_schema()

    # Remove existing DB if it exists
    if DB_PATH.exists():
        import shutil
        shutil.rmtree(DB_PATH)

    collection = zvec.create_and_open(path=str(DB_PATH), schema=schema)

    # Insert documents in batches
    print("Inserting documents...")
    INSERT_BATCH = 100

    for i in range(0, len(questions), INSERT_BATCH):
        batch_q = questions[i:i + INSERT_BATCH]
        batch_emb = embeddings[i:i + INSERT_BATCH]

        docs = [
            zvec.Doc(
                id=q["id"],
                vectors={"embedding": emb},
                fields={
                    "subject": q["subject"],
                    "topic": q["topic"],
                    "subtopic": q["subtopic"],
                    "difficulty": q["difficulty"],
                    "bloom_level": q["bloom_level"],
                    "verified": q["verified"],
                    "payload": json.dumps(q),
                }
            )
            for q, emb in zip(batch_q, batch_emb)
        ]

        collection.insert(docs)
        print(f"  Inserted batch {i // INSERT_BATCH + 1}/{(len(questions) - 1) // INSERT_BATCH + 1}")

    print(f"Successfully ingested {len(questions)} questions into zvec!")
    return len(questions)


def test_query():
    """Test querying the database."""
    print("\nTesting query...")

    collection = zvec.open(path=str(DB_PATH))

    # Test query
    test_text = "What is the quadratic formula?"
    print(f"Query: '{test_text}'")

    # Get embedding for query
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[test_text]
    )
    query_embedding = response.data[0].embedding

    # Search
    results = collection.query(
        zvec.VectorQuery("embedding", vector=query_embedding),
        topk=3
    )

    print(f"\nTop 3 results:")
    for i, result in enumerate(results, 1):
        # zvec returns Doc objects with fields attribute
        payload = json.loads(result.fields["payload"])
        print(f"{i}. [{payload['difficulty']}] {payload['question_text'][:80]}...")
        print(f"   Score: {result.score:.4f}")

if __name__ == "__main__":
    ingest_questions()
    test_query()
