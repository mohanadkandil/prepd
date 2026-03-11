"""
Generates high school math MCQ questions using the Claude API.
Outputs a JSONL file ready for Qdrant ingestion.
"""

import anthropic
import json
import uuid
from datetime import date
from dotenv import load_dotenv
import os
from pathlib import Path

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

# CONFIG

MATH_TOPICS = {
    "Algebra": [
        "Linear equations", "Quadratic equations", "Inequalities",
        "Systems of equations", "Polynomials", "Factoring", "Functions"
    ],
    "Geometry": [
        "Triangles", "Circles", "Area and perimeter", "Volume",
        "Coordinate geometry", "Transformations", "Proofs"
    ],
    "Trigonometry": [
        "SOH-CAH-TOA", "Unit circle", "Trigonometric identities",
        "Sine and cosine rules", "Graphs of trig functions"
    ],
    "Statistics & Probability": [
        "Mean, median, mode", "Standard deviation", "Probability basics",
        "Conditional probability", "Distributions", "Data interpretation"
    ],
    "Calculus (Pre-Calc)": [
        "Limits", "Derivatives basics", "Integrals basics",
        "Sequences and series", "Exponential functions", "Logarithms"
    ],
    "Number Theory": [
        "Primes and factors", "Ratios and proportions", "Percentages",
        "Surds and indices", "Scientific notation"
    ],
}

DIFFICULTIES = ["easy", "medium", "hard"]

BLOOM_LEVELS = {
    "easy":   ["recall", "comprehension"],
    "medium": ["comprehension", "application"],
    "hard":   ["application", "analysis"],
}

QUESTIONS_PER_COMBO = 10
TODAY = date.today().isoformat()

# Define the tool for structured output 
QUESTIONS_TOOL = {
    "name": "submit_questions",
    "description": "Submit the generated math questions",
    "input_schema": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question_text": {"type": "string"},
                        "options": {
                            "type": "object",
                            "properties": {
                                "A": {"type": "string"},
                                "B": {"type": "string"},
                                "C": {"type": "string"},
                                "D": {"type": "string"}
                            },
                            "required": ["A", "B", "C", "D"]
                        },
                        "correct_answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                        "explanation": {"type": "string"},
                        "estimated_time_seconds": {"type": "integer"}
                    },
                    "required": ["question_text", "options", "correct_answer", "explanation", "estimated_time_seconds"]
                }
            }
        },
        "required": ["questions"]
    }
}

# PROMPT

SYSTEM_PROMPT = """You are an expert high school mathematics teacher creating exam questions."""

def make_user_prompt(subject: str, topic: str, subtopic: str, difficulty: str, bloom: str, n: int) -> str:
    return f"""Generate {n} multiple-choice questions about "{subtopic}" (part of {subject} > {topic}).
- Student level: High school (ages 14-18)
- Difficulty: {difficulty}
- Bloom's taxonomy level: {bloom}

Rules:
- One correct answer, three plausible distractors
- No ambiguous questions
- Explanations must show working, not just the answer
- estimated_time_seconds: easy=45, medium=90, hard=150
"""

# GENERATOR

def generate_batch(client: anthropic.Anthropic, subject: str, topic: str,
                   subtopic: str, difficulty: str, bloom: str) -> list[dict]:
    prompt = make_user_prompt(subject, topic, subtopic, difficulty, bloom, QUESTIONS_PER_COMBO)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt + "\n\nUse the submit_questions tool to return your response."}],
        tools=[QUESTIONS_TOOL],
        tool_choice={"type": "tool", "name": "submit_questions"}
    )

    tool_use = next(block for block in response.content if block.type == "tool_use")
    questions = tool_use.input["questions"]
    
    enriched = []
    for i, q in enumerate(questions):
        enriched.append({
            "id": str(uuid.uuid4()),
            
            # Embeddable text (question + options joined)
            "embed_text": q["question_text"] + " " + " ".join(q["options"].values()),
            
            # Question content
            "question_text": q["question_text"],
            "question_type": "mcq",
            "options": q["options"],
            "correct_answer": q["correct_answer"],
            "explanation": q["explanation"],
            
            # Classification 
            "subject": subject,
            "topic": topic,
            "subtopic": subtopic,
            "difficulty": difficulty,
            "bloom_level": bloom,
            "grade_level": "high_school",
            "curriculum": "general",        # Can be GCSE / SAT / IB etc.
            "language": "en",
            
            # Exam metadata
            "tags": [subject.lower(), topic.lower(), subtopic.lower(), difficulty],
            "estimated_time_seconds": q.get("estimated_time_seconds", 60),
            "source": "ai_generated",
            "verified": False,
            
            # Analytics (populated at runtime)
            "times_used": 0,
            "avg_score_rate": None,
            
            # Timestamps
            "created_at": TODAY,
            "updated_at": TODAY,
        })
    return enriched


def generate_all(output_file: str = "questions.jsonl"):
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))  
    all_questions = []
    total = 0

    for subject, topics in MATH_TOPICS.items():
        for topic in topics:
            for difficulty in DIFFICULTIES:
                bloom = BLOOM_LEVELS[difficulty][0]
                print(f"  Generating: {subject} > {topic} | {difficulty} | {bloom} ...", end=" ")
                try:
                    batch = generate_batch(client, subject, subject, topic, difficulty, bloom)
                    all_questions.extend(batch)
                    total += len(batch)
                    print(f"✓ {len(batch)} questions")
                except Exception as e:
                    print(f"✗ Error: {e}")

    with open(output_file, "w") as f:
        for q in all_questions:
            f.write(json.dumps(q) + "\n")

    print(f"\nDone! {total} questions saved to {output_file}")
    return output_file


if __name__ == "__main__":
    # Output to data folder (relative to project root)
    import pathlib
    output_path = pathlib.Path(__file__).parent.parent / "data" / "questions.jsonl"
    generate_all(str(output_path))
