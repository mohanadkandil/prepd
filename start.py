"""
Startup script for Railway deployment.
Initializes the database if needed and starts the server.
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Check if database exists, if not run ingest
DB_PATH = PROJECT_ROOT / "data" / "questions_db"

if not DB_PATH.exists():
    print("Database not found. Running ingest...")
    from backend.ingest_zvec import ingest_questions
    ingest_questions()
    print("Ingest complete!")
else:
    print(f"Database found at {DB_PATH}")

# Start the server
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port)
