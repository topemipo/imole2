import openai
import psycopg2
import json

import os
import numpy as np
from psycopg2.extras import execute_values
from openai import OpenAI
import tiktoken
from dotenv import load_dotenv

# === Configure OpenAI API Key ===
load_dotenv()

# OpenAI
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

# === Database Connection Details ===
DB_PARAMS = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "sslmode": "require"
}

# === Connect to PostgreSQL ===
conn = psycopg2.connect(**DB_PARAMS)
cursor = conn.cursor()

# === Load Summaries JSON File ===
json_path = "summaries.json"  # Update path if needed
with open(json_path, "r", encoding="utf-8") as f:
    summaries = json.load(f)

# === Insert Each Case into PostgreSQL ===
for doc_id, details in summaries.items():
    filename = details["filename"]
    summary_text = details["summary"]

    # Generate embedding
    response = openai.embeddings.create(
        input=[summary_text],
        model="text-embedding-ada-002"
    )
    embedding = response.data[0].embedding  # 1536-dimension vector

    # Insert into PostgreSQL
    cursor.execute("""
        INSERT INTO legal_cases (filename, summary, embedding)
        VALUES (%s, %s, %s::vector)
        ON CONFLICT (filename) DO NOTHING;
    """, (filename, summary_text, embedding))

# === Commit Changes & Close Connection ===
conn.commit()
cursor.close()
conn.close()
print("✅ All legal cases inserted successfully!")