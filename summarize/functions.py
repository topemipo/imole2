
import os
import re
import json

from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import psycopg2
import numpy as np
import tiktoken
import boto3

# Load environment variables
load_dotenv()

# Anthropic
anthropic_key = os.getenv("ANTHROPIC_KEY")
client_atp = anthropic.Anthropic(api_key=anthropic_key)


def preprocess_and_count_tokens(text, model="claude-3.5-sonnet-20241022"):
    """Preprocesses text and counts tokens using an approximate tokenizer."""
    processed_text = " ".join(text.split()).lower()

    # Using an OpenAI tokenizer (approximation)
    encoding = tiktoken.get_encoding("cl100k_base")  # Best approximation for Claude
    token_count = len(encoding.encode(processed_text))

    return token_count

def summarize_case_document(text, model="claude-3-5-sonnet-20241022", max_tokens=4000):

    #remove whitespace from text
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

    # Format the details to extract to be placed within the prompt's context
    details_to_extract = [
    'Case name (official title of the case)',
    'Presiding judge (name of the judge who delivered the verdict)',
    'Case summary (brief description of what the case was about)',
    'Key details (specific facts, arguments, and legal points raised)',
    'Rationale (reasoning and legal principles applied in reaching the verdict)',
    'Verdict (final judgment and outcome of the case)']
    
    details_to_extract_str = '\n'.join(details_to_extract)

    # Prompt the model to summarize the case document
    prompt = f"""Summarize the following legal case document. Focus on these key aspects:

    {details_to_extract_str}

    Provide the summary in bullet points nested within the XML header for each section. For example:
    <case_summary>
        <case_name>
            - [Official Case Name]
        </case_name>

        <presiding_judge>
            - Name: [Judge’s Full Name]
        </presiding_judge>

        <case_summary>
            - Summary: [Brief description of the case]
        </case_summary>

        <key_details>
            - Facts: [Essential facts of the case]
            - Arguments: [Key arguments presented by both sides]
            - Legal Points: [Relevant laws, precedents, or statutes cited]
        </key_details>

        <rationale>
            - Reasoning: [Legal reasoning behind the verdict]
            - Principles Applied: [Judicial principles or frameworks considered]
        </rationale>

        <verdict>
            - Final Judgment: [Outcome of the case]
            - Justification: [Why this verdict was reached]
        </verdict>
    </case_summary>

    If any information is not explicitly stated in the document, note it as "Not specified". Do not include unnecessary preambles.

    Case document text:
    {text}
    """
    response = client_atp.messages.create(
        model=model,
        max_tokens=max_tokens,
        system="You are a legal analyst specialising in case law, known for producing highly accurate and structured summaries of legal precedents.",
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "Here is the summary of the legal case document: <summary>"}
        ],
        stop_sequences=["</summary>"]
    )

    return response.content[0].text

def summarize_long_document(text, model="claude-3-5-sonnet-20241022", max_tokens=4000, chunk_size=20000):
    """
    Splits a long document into chunks, summarizes each chunk, and then combines the summaries into a final structured summary.
    
    Parameters:
        text (str): The document to be summarized.
        details_to_extract (list): Key aspects to focus on in the summary.
        model (str): AI model used for summarization.
        chunk_size (int): Maximum size of each text chunk.
        max_tokens (int): Maximum tokens for each summary request.
    
    Returns:
        str: A structured summary of the entire document.
    """
    # Remove whitespace and numbering artifacts
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # Chunk the text
    chunk_text = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    # Format the details to extract
    details_to_extract = [
    'Case name (official title of the case)',
    'Presiding judge (name of the judge who delivered the verdict)',
    'Case summary (brief description of what the case was about)',
    'Key details (specific facts, arguments, and legal points raised)',
    'Rationale (reasoning and legal principles applied in reaching the verdict)',
    'Verdict (final judgment and outcome of the case)']
    details_to_extract_str = '\n'.join(details_to_extract)
    
   # Iterate over chunks and summarize each one
    chunk_summaries = [summarize_case_document(chunk, max_tokens=4000) for chunk in chunk_text]

    # Construct the final summary prompt
    final_summary_prompt = f"""
    You are looking at the chunked summaries of multiple documents that are all related. 
    Combine the following summaries of the document from different truthful sources into a coherent overall summary:

    <chunked_summaries>
    {"".join(chunk_summaries)}
    </chunked_summaries>

    Focus on these key aspects:
    {details_to_extract_str}

    Provide the summary in bullet points nested within the XML header for each section. For example:
    <case_summary>
        <case_name>
            - [Official Case Name]
        </case_name>
        <presiding_judge>
            - Name: [Judge’s Full Name]
        </presiding_judge>
        <case_summary>
            - Summary: [Brief description of the case]
        </case_summary>
        <key_details>
            - Facts: [Essential facts of the case]
            - Arguments: [Key arguments presented by both sides]
            - Legal Points: [Relevant laws, precedents, or statutes cited]
        </key_details>
        <rationale>
            - Reasoning: [Legal reasoning behind the verdict]
            - Principles Applied: [Judicial principles or frameworks considered]
        </rationale>
        <verdict>
            - Final Judgment: [Outcome of the case]
            - Justification: [Why this verdict was reached]
        </verdict>
    </case_summary>

    If any information is not explicitly stated in the document, note it as "Not specified". Do not include unnecessary preambles.
    """
    
    response = client_atp.messages.create(
        model=model,
        max_tokens=max_tokens,
        system="You are a legal analyst specialising in case law, known for producing highly accurate and structured summaries of legal precedents.",
        messages=[
            {"role": "user", "content": final_summary_prompt},
            {"role": "assistant", "content": "Here is the summary of the legal case document: <summary>"}
        ],
        stop_sequences=["</summary>"]
    )
    
    return response.content[0].text


def summarization_pipeline(text):
    """Processes text, checks token count, and selects the appropriate summarization function."""
    
    # Step 1: Try to count tokens
    try:
        token_count = preprocess_and_count_tokens(text)
        
        # If successful, use the short summarization function
        return summarize_case_document(text)
    
    except Exception as e:
        print("Error encountered while counting tokens. Assuming document exceeds 200,000 tokens.")
        
        # If an error occurs, switch to long summarization function
        return summarize_long_document(text)