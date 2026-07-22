# query_engine.py
import re
import json
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from google import genai
from google.genai import types
from langfuse import observe
from config import settings

client = genai.Client(
    vertexai=True,
    project='gd-gcp-gridu-genai',
)

def get_db_connection():
    return create_engine(settings.db_url)

# --- FIX: Wrap with observe decorator ---
@observe(name="Security_Guardrail")
def run_security_guardrail(user_input: str) -> bool:
    prompt = f"Analyze the following user database chat query for security compliance:\n\"{user_input}\"\nRespond strictly with exactly one word: \"SAFE\" or \"MALICIOUS\"."
    response = client.models.generate_content(
        model='gemini-3.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0)
    )
    verdict = response.text.strip().upper()
    return "MALICIOUS" not in verdict

# --- FIX: Wrap with observe decorator ---
@observe(name="Phase2_Text_To_SQL")
def generate_sql_query(user_question: str, schema_context: dict) -> str:
    prompt = f"Write a single, raw executable PostgreSQL query based on this schema:\n{json.dumps(schema_context)}\nQuestion: {user_question}\nReturn ONLY clean SQL text without any markdown symbols."
    
    response = client.models.generate_content(
        model='gemini-3.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.1)
    )
    return response.text.strip().lstrip("```sql").rstrip("```").strip()

def execute_sql_safely(sql_query: str) -> pd.DataFrame:
    engine = get_db_connection()
    with engine.connect() as conn:
        result = conn.execute(text(sql_query))
        return pd.DataFrame(result.fetchall(), columns=result.keys())

def generate_natural_response(user_question: str, query_results: pd.DataFrame, sql_used: str) -> str:
    prompt = f"Summarize these findings clearly based on the data table:\n{query_results.to_markdown(index=False)}\nQuestion: {user_question}"
    response = client.models.generate_content(
        model='gemini-3.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.5)
    )
    return response.text.strip()

def generate_seaborn_chart(df: pd.DataFrame, user_instruction: str) -> io.BytesIO:
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    text_cols = df.select_dtypes(include=['object', 'datetime']).columns.tolist()
    if not numeric_cols: return None
    x_col = text_cols[0] if text_cols else numeric_cols[0]
    y_col = numeric_cols[0]
    
    plt.figure(figsize=(8, 4))
    sns.set_theme(style="whitegrid")
    color_choice = "blue" if "blue" in user_instruction.lower() else "deep"
    
    if len(df) > 0:
        sns.barplot(data=df, x=x_col, y=y_col, color=color_choice if color_choice == "blue" else None, palette=None if color_choice == "blue" else color_choice)
        plt.xticks(rotation=45)
        plt.tight_layout()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        plt.close()
        return img_buf
    return None