from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
import openai
from openai import OpenAI
import os
import psycopg2
from dotenv import load_dotenv

from rag_functions import(
    augment_query_generated, retrieve_similar_case, generate_response
)

# OpenAI
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

# Database connection details
DB_PARAMS = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "sslmode": "require"
}

# Initialize FastAPI app
app = FastAPI()

# Define the request model
class QueryRequest(BaseModel):
    query: str

@app.get("/")
def root():
    return {"message": "Things are as they should be :)"}

@app.post("/chat/")
def chat(request: QueryRequest):
    """Handles user legal questions, augments query, retrieves case law, and generates AI response."""
    try:
        # Expand the query for better retrieval
        user_query = request.query
        augmented_query = augment_query_generated(user_query)
        final_query = f"{user_query} {augmented_query}"

        # Query PostgreSQL (`pgvector`) for relevant legal documents
        retrieved_case = retrieve_similar_case(final_query)

        if not retrieved_case:
            ai_response = "⚠️ No relevant legal documents found. Consider consulting a lawyer."
            return {"response": ai_response}

        # Generate AI response
        ai_response = generate_response(user_query, retrieved_case)

        return {"response": ai_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



