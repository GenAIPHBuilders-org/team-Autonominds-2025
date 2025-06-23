import math
import os
import re
import sys
import json
import traceback
import time
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional
import contextlib
import io
import threading

# Third-party imports
import numpy as np
from sklearn.cluster import KMeans
import chromadb
import google.generativeai as genai


# --- 1. Load .env file from project root ---
# This MUST be done BEFORE importing any project modules that might need these env vars
# (like api_client_manager via paper_collector, keyword_critic, etc.)
from dotenv import load_dotenv  # Import load_dotenv here

project_root = Path(__file__).resolve().parent.parent  # Get project root directory
dotenv_path = project_root / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
    print(f"INFO: app.py loaded .env file from: {dotenv_path}")
else:
    print(f"WARNING: app.py did not find .env file at: {dotenv_path}")

sys.path.append(str(project_root))
from embeddings import embed_text

# --- 2. Initialize ChromaDB
CHROMA_PATH = str(Path(__file__).resolve().parent / "chroma_db_store")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)


# --- Helper classes for streaming stdout to the frontend ---
class _SocketIOWriter(io.TextIOBase):
    def __init__(self, emit_func):
        super().__init__()
        self.emit_func = emit_func
        self._buffer = ""

    def write(self, s):
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if line:
                self.emit_func(line)
        return len(s)

    def flush(self):
        if self._buffer.strip():
            self.emit_func(self._buffer.strip())
            self._buffer = ""


class _StdoutTee(io.TextIOBase):
    def __init__(self, original, emit_func):
        super().__init__()
        self.original = original
        self.socket_writer = _SocketIOWriter(emit_func)

    def write(self, s):
        self.original.write(s)
        self.socket_writer.write(s)
        return len(s)

    def flush(self):
        self.original.flush()
        self.socket_writer.flush()


@contextlib.contextmanager
def tee_stdout(emit_func):
    original = sys.stdout
    tee = _StdoutTee(original, emit_func)
    sys.stdout = tee
    try:
        yield
    finally:
        sys.stdout = original
        tee.flush()


# --- 2.1. ChromaDB Helpers
# Helper for chunking text
def chunk_text(text, chunk_size=512, overlap=50):
    """
    A simple text chunker. This function takes a long string of text
    and breaks it into smaller, overlapping pieces. For a hackathon, this is sufficient.
    """
    # 1. Split the text into a list of words.
    words = text.split()

    # 2. Create an empty list where we will store the smaller text pieces (chunks).
    chunks = []

    # 3. Step through the list of words, taking 512 words at a time,
    # but stepping forward by only 462 words (512 - 50) to create an overlap.
    for i in range(0, len(words), chunk_size - overlap):
        # Join the words for the current chunk back into a single string.
        chunk_content = " ".join(words[i : i + chunk_size])
        # Add the completed chunk to our list.
        chunks.append(chunk_content)

    # 4. Return the list of text chunks.
    return chunks


# The main indexing function
def build_rag_index(papers: list, session_id: str):
    """Creates a ChromaDB collection for a user session and indexes paper abstracts."""

    collection_name = f"session_{session_id.replace('-', '_')}"  # Sanitize session ID for collection name

    # Get or create the collection
    collection = chroma_client.get_or_create_collection(name=collection_name)

    documents, metadatas, ids = [], [], []
    chunk_count = 0
    for paper in papers:
        abstract = paper.get("abstract")
        if not abstract:
            continue

        paper_chunks = chunk_text(abstract)
        for i, chunk in enumerate(paper_chunks):
            documents.append(chunk)
            metadatas.append(
                {"doi": paper.get("doi", ""), "title": paper.get("title", "")}
            )
            ids.append(f"{paper.get('doi', '')}_{chunk_count}_{i}")

        chunk_count += 1

    if not documents:
        print(f"No documents to index for session {session_id}")
        return

    # Add all documents to the collection. ChromaDB will handle embedding internally.
    collection.add(documents=documents, metadatas=metadatas, ids=ids)

    print(f"Indexed {len(ids)} chunks into collection '{collection_name}'")


# --- 3. Add Project Root to sys.path for imports ---
# This allows importing your backend modules from the parent directory (project root)
sys.path.append(str(project_root))
print(f"INFO: Project root '{str(project_root)}' added to sys.path.")

# --- 4. Import Flask, SocketIO, and YOUR Up-to-Date Backend Modules ---
from flask import Flask, render_template, jsonify, request, session
from flask_socketio import SocketIO
import uuid

# Import your project's modules (ensure these are the cleaned versions)
# Note: api_client_manager.py will be used internally by these modules.
from paper_collector import (
    generate_topics,
    generate_subthemes,  # New function to import
    generate_keywords_by_subtheme,  # New function to import
    collect_papers,
    generate_domain_terms,
    generate_app_terms,  # Ensure this is imported if used by app.py's run_pipeline
    generate_tech_terms,  # Ensure this is imported if used by app.py's run_pipeline
)
from filter_and_rank import (
    filter_by_doi,
    filter_by_abstract,
    dedupe_by_doi,
    semantic_rank_papers,
    # Add other functions from filter_and_rank if your app.py's run_pipeline will use them e.g. load_candidates_from_json
)
from keyword_critic import critique_list  # For critiquing terms

# from embeddings import embed_text
from embeddings import embed_text
from extract_insights import generate_insights

from vector_store_manager import build_rag_index, query_rag_index

# Initialize Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = (
    "authematic-secret-key"  # Good to keep this configurable or more random for production
)
socketio = SocketIO(app)

# A simple in-memory store for user session data
# Key: user_socket_id, Value: list of final papers
USER_SESSIONS = {}
# Mapping of user socket IDs to threading.Event objects used for
# cancelling a running pipeline for that user.
PIPELINE_CANCEL_EVENTS: Dict[str, threading.Event] = {}

# --- Constants for Term Cleaning (from run_pipeline.py) ---
STOP_TIER1 = {
    "survey",
    "review",
    "framework",
    "architecture",
    "architectures",
    "analysis",
    "analyses",
    "system",
    "systems",
}
STOP_TIER2 = {
    "method",
    "methods",
    "approach",
    "approaches",
    "algorithm",
    "algorithms",
    "technique",
    "techniques",
}


# --- Helper Function for Term Cleaning (from run_pipeline.py) ---
def clean_terms(terms: List[str]) -> List[str]:
    """
    Cleans a list of terms by removing items containing predefined stop words.
    (Same implementation as in your cleaned run_pipeline.py)
    """
    # First pass: remove terms containing any Tier-1 stop word
    tier1_clean = [t for t in terms if set(t.split()).isdisjoint(STOP_TIER1)]
    if tier1_clean:
        return tier1_clean

    tier2_clean = [t for t in terms if set(t.split()).isdisjoint(STOP_TIER2)]
    if tier2_clean:
        return tier2_clean

    return terms


# ==================== NEW CHAT FUNCTIONS ====================


def get_general_rag_response(query: str, collection_name: str):
    """
    Queries the ChromaDB collection to find relevant context and then answers the user's question.
    """

    try:
        # Ensure you handle the case where the collection might not exist
        # This can happen if a user reconnects to a session where the index wasn't built
        # or was cleaned up.
        collection = chroma_client.get_collection(name=collection_name)
    except ValueError:
        # This is a critical error message for the user.
        return "Error: Could not find the research context for this session. Please start a new search by typing 'reset'."

    # 1. Retrieve more documents to give the LLM a richer context.
    # Increasing n_results from 3 to 5 is a good starting point.
    try:
        query_embedding = embed_text(query, use_cache=True).tolist()

        results = collection.query(
            query_embeddings=[
                query_embedding
            ],  # Use query_embeddings instead of query_texts
            n_results=5,
        )
        docs = results["documents"][0]
        metas = results["metadatas"][0]

        if not docs:
            return "I couldn't find any specific information about that in the provided papers. Could you ask about something else?"

    except Exception as e:
        print(
            f"ERROR: ChromaDB query failed for collection {collection_name}. Error: {e}"
        )
        return "Sorry, I encountered an error while searching the documents. Please try another question."

    # 2. Join the retrieved abstracts with their titles into a single context block.
    context_blocks = [
        f"Title: {meta.get('title', 'N/A')}\nAbstract: {doc}"
        for doc, meta in zip(docs, metas)
    ]
    context = "\n\n---\n\n".join(context_blocks)

    # 3. This new prompt reframes the agent's role.
    # It encourages synthesis, direct quotes, and a helpful, academic tone.
    prompt = f"""You are Authematic, an expert AI research companion. Your primary role is to help users understand, analyze, and synthesize information from scientific papers. You must answer based *exclusively* on the provided context. Do not use any prior knowledge.

**Core Directives:**
1.  **Grounding is Paramount:** Every part of your answer must be directly traceable to the provided context. Never invent information, metrics, or conclusions.
2.  **Assume User is an Expert:** The user is likely a researcher. Prioritize technical accuracy and detail. Use professional, scientific language.
3.  **Structure Your Answers:** Begin with a direct, concise answer to the user's question. Then, provide the detailed, structured explanation and supporting evidence from the context.
4.  **Cite Your Sources:** Attribute findings to their sources by differentiating between the papers. For example, use phrases like "The study on [topic A] found that...", "In contrast, the paper discussing [topic B] suggests...", or "Multiple abstracts agree on...".

---
**Task-Specific Instructions:**

Based on the user's question, adopt the most appropriate response mode:

**1. Analyze Mode (Default):**
* When the user asks "what," "how," or "why," provide a deep analysis.
* Extract and explain key methodologies, technical terms, datasets, and quantitative results mentioned in the context.
* Identify the main claims and the evidence presented for them.

**2. Compare/Contrast Mode:**
* If the user asks to compare papers or ideas, explicitly identify points of agreement, disagreement, and divergence.
* Highlight differences in methodology, scope, or conclusions between the sources.
* Synthesize these different perspectives into a coherent overview of the research landscape as presented in the context.

**3. Summarize Mode:**
* If the user asks for a summary, create a dense, information-rich synthesis of the relevant information.
* Do not just list findings; connect them to explain the overall narrative or state of knowledge in the context.

**4. Simplify Mode:**
* If the user explicitly asks for a simple explanation (e.g., "explain this simply," "in plain English"), translate the core concepts into accessible language.
* Use analogies or metaphors if helpful.
* Crucially, add a concluding sentence that flags it as a simplification, e.g., "This is a simplified overview; the source material provides deeper technical details on [mention specific concepts]."

**5. Proactive Assistance:**
* After answering the question, if you notice clear gaps or unanswered questions in the context, briefly mention them. For example, "...the context does not specify the sample size used in this experiment."
* If appropriate, you can suggest a logical follow-up question that the user might want to ask next.

---
**Provided Abstracts From Research Papers:**
{context}
---

**User's Question:** {query}

**Your Expert Answer:**"""

    # 4. Call the LLM to get the final answer
    model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
    try:
        response = model.generate_content(contents=prompt)
        # Add a check for empty or invalid responses from the API
        if response and response.text:
            return response.text
        else:
            return "I was unable to generate a response for that query. Please try rephrasing your question."
    except Exception as e:
        print(f"ERROR: Gemini API call failed in get_general_rag_response. Error: {e}")
        return "Sorry, I'm having trouble connecting to my reasoning engine. Please try again in a moment."


def get_single_paper_response(query: str, paper: dict):
    """
    Answers a user's question based ONLY on the title and abstract of a single paper.
    """
    if not paper or not paper.get("abstract"):
        return "This paper does not have an abstract available to chat with."

    # This prompt is highly constrained to prevent the LLM from using general knowledge.
    prompt = f"""You are an academic assistant. Based ONLY on the provided paper title and abstract, answer the user's question. Keep the usage of external knowledge to a minimum.

Paper Title: {paper.get('title', '')}

Abstract: {paper.get('abstract', '')}

User Question: {query}

Answer:"""

    model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
    response = model.generate_content(contents=prompt)
    return response.text


## cleanup after disconnection
@socketio.on("disconnect")
def handle_disconnect():
    """
    Handles cleaning up user data when they disconnect.
    """
    user_socket_id = request.sid
    if user_socket_id in USER_SESSIONS:
        # Here you could also add logic to clean up the ChromaDB collection if desired
        del USER_SESSIONS[user_socket_id]
    if user_socket_id in PIPELINE_CANCEL_EVENTS:
        PIPELINE_CANCEL_EVENTS.pop(user_socket_id).set()
    print(f"Cleaned up session data for disconnected user: {user_socket_id}")


# ==================== CITATION FUNCTIONS ====================


def format_apa_citation(paper):
    """
    Format a paper dictionary into APA citation style.

    Args:
        paper (dict): Paper dictionary with keys: title, authors, year, doi

    Returns:
        str: Formatted APA citation
    """
    try:
        # Extract basic information
        title = paper.get("title", "Untitled").strip()
        year = paper.get("year", "n.d.")
        doi = paper.get("doi", "")
        authors = paper.get("authors", [])

        # Format authors
        if not authors:
            author_text = "Author, A."
        elif len(authors) == 1:
            author_text = format_author_apa(authors[0])
        elif len(authors) == 2:
            author_text = (
                f"{format_author_apa(authors[0])}, & {format_author_apa(authors[1])}"
            )
        elif len(authors) <= 20:
            formatted_authors = [format_author_apa(author) for author in authors[:-1]]
            author_text = (
                ", ".join(formatted_authors) + f", & {format_author_apa(authors[-1])}"
            )
        else:
            # For more than 20 authors, list first 19, then "...", then last author
            formatted_authors = [format_author_apa(author) for author in authors[:19]]
            author_text = (
                ", ".join(formatted_authors) + f", ... {format_author_apa(authors[-1])}"
            )

        # Clean and format title
        if not title.endswith("."):
            title += "."

        # Format DOI or URL
        if doi and doi.startswith("https://doi.org/"):
            doi_text = doi
        elif doi and not doi.startswith("http"):
            doi_text = f"https://doi.org/{doi}"
        elif doi:
            doi_text = doi
        else:
            doi_text = ""

        # Build citation
        citation_parts = []
        citation_parts.append(f"{author_text} ({year}).")
        citation_parts.append(f"{title}")

        if doi_text:
            citation_parts.append(f"{doi_text}")

        return " ".join(citation_parts)

    except Exception as e:
        print(f"Error formatting citation: {e}")
        return f"Error formatting citation for: {paper.get('title', 'Unknown title')}"


def format_paper_with_citation_and_insights(paper: Dict[str, Any], index: int) -> str:
    """
    Format a single paper result with its APA citation and AI-generated insights.
    """
    # Basic paper info
    result_msg = (
        f"**{index}. {paper.get('title', 'Untitled')}** ({paper.get('year', 'N/A')})\n"
    )

    authors = paper.get("authors", [])
    if authors:
        author_text = ", ".join(authors[:3])  # Show first 3 authors
        if len(authors) > 3:
            author_text += f", et al."  # Indicate more authors
        result_msg += f"   👥 **Authors:** {author_text}\n"

    result_msg += f"   🔗 **DOI:** {paper.get('doi', 'N/A')}\n"
    result_msg += f"   📊 **Relevance Score:** {paper.get('score', 0.0):.4f}\n"

    # APA Citation
    apa_citation = format_apa_citation(paper)  # Assumes format_apa_citation is defined
    result_msg += f"   📚 **APA Citation:** {apa_citation}\n"

    # Add AI Insights
    if paper.get("insights_summary") and paper.get("insights_summary") not in [
        "Summary not available.",
        "Insight generation skipped: Abstract was missing or empty.",
    ]:
        result_msg += f"   💡 **AI Summary:** {paper['insights_summary']}\n"
    if paper.get("insights_relevance") and paper.get("insights_relevance") not in [
        "Relevance assessment not available.",
        "Relevance assessment skipped: Abstract was missing or empty.",
    ]:
        result_msg += f"   🎯 **AI Relevance to '{paper.get('query_title', 'your research')}':** {paper['insights_relevance']}\n"  # Assuming query_title is available or use generic

    return result_msg


def format_author_apa(author_name: str) -> str:
    """
    Formats a single author name into APA style (Last, F. M.).
    This is a more robust version that handles "Last, First" and "First M. Last"
    and even malformed data like "FM, Last".
    """
    if not author_name or not isinstance(author_name, str):
        return "N.A."

    name = author_name.strip()

    # Heuristic-based parsing
    if "," in name:
        parts = name.split(",", 1)
        part1 = parts[0].strip()
        part2 = parts[1].strip()

        # Heuristic: If the part BEFORE the comma is very short (likely initials)
        # then the part AFTER the comma is the last name. e.g., "FK, W."
        if len(part1.replace(".", "")) <= 2:
            last_name = part2
            first_middle_parts = part1.replace(".", " ").split()
        # Otherwise, assume standard "Last, First M." format
        else:
            last_name = part1
            first_middle_parts = part2.replace(".", " ").split()

    # No comma present
    else:
        parts = name.split()
        if len(parts) == 1:
            return parts[0]

        last_token = parts[-1]

        # If the last token looks like initials (e.g., "PJ" or "C"), treat it
        # as the first/middle name and everything before it as the last name.
        # This covers formats like "Xu C" or "Königs EK" from some metadata
        # sources.
        token_stripped = last_token.replace(".", "")
        if (
            len(token_stripped) <= 3
            and token_stripped.isalpha()
            and token_stripped.upper() == token_stripped
        ):
            last_name = " ".join(parts[:-1])
            # Split combined initials into separate characters
            first_middle_parts = list(token_stripped)
        else:
            last_name = last_token
            first_middle_parts = parts[:-1]

    # Generate initials from the identified first/middle parts
    initials = []
    for part in first_middle_parts:
        if part:  # Ensure the part is not an empty string
            initials.append(f"{part[0].upper()}.")

    if not initials:
        return last_name

    return f"{last_name}, {' '.join(initials)}"


def generate_bibliography(papers):
    """
    Generate a complete bibliography from a list of papers.

    Args:
        papers (list): List of paper dictionaries

    Returns:
        str: Formatted bibliography
    """
    if not papers:
        return "No papers to cite."

    citations = []
    for paper in papers:
        citation = format_apa_citation(paper)
        citations.append(citation)

    # Sort alphabetically by first author's last name
    citations.sort()

    bibliography = "**BIBLIOGRAPHY (APA Style)**\n\n"
    for i, citation in enumerate(citations, 1):
        bibliography += f"{citation}\n\n"

    return bibliography


def format_paper_with_citation(paper, index):
    """
    Format a single paper result with its APA citation.

    Args:
        paper (dict): Paper dictionary
        index (int): Paper number in the list

    Returns:
        str: Formatted paper with citation
    """
    # Basic paper info
    result_msg = f"**{index}.** **{paper['title']}** ({paper.get('year', 'N/A')})\n"

    # Authors
    authors = paper.get("authors", [])
    if authors:
        author_text = ", ".join(authors[:3])
        if len(authors) > 3:
            author_text += f" and {len(authors)-3} more"
        result_msg += f"   👥 **Authors:** {author_text}\n"

    # DOI and Score
    result_msg += f"   🔗 **DOI:** {paper.get('doi', 'N/A')}\n"
    result_msg += f"   📊 **Relevance Score:** {paper.get('score', 0):.4f}\n"

    # APA Citation
    apa_citation = format_apa_citation(paper)
    result_msg += f"   📚 **APA Citation:** {apa_citation}"

    return result_msg


# ==================== FLASK ROUTES ====================


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("connect")
def handle_connect():
    print("Client connected")


@socketio.on("disconnect")
def handle_disconnect():
    """
    Handles cleaning up user data when they disconnect.
    """
    user_socket_id = request.sid
    if user_socket_id in USER_SESSIONS:
        del USER_SESSIONS[user_socket_id]
    if user_socket_id in PIPELINE_CANCEL_EVENTS:
        PIPELINE_CANCEL_EVENTS.pop(user_socket_id).set()
    print(f"Cleaned up session data for disconnected user: {user_socket_id}")


# this is for the first inputs
@socketio.on("send_message")
def handle_message(data):
    user_socket_id = request.sid
    user_message = data.get("message", "")
    context = data.get("context", {})
    print(f"Received message: '{user_message}' with context: {context}")

    # Ensure we have a place to store chat history for this user
    user_session = USER_SESSIONS.setdefault(user_socket_id, {})
    chat_history: List[Dict[str, str]] = user_session.setdefault("chat_history", [])

    def emit_to_requesting_user(event_name, data_payload):
        socketio.emit(event_name, data_payload, room=user_socket_id)
        if event_name == "receive_message":
            chat_history.append(
                {
                    "sender": data_payload.get("sender", "bot"),
                    "message": data_payload.get("message", ""),
                }
            )

    # Clear context on reset
    if user_message.lower() == "reset" or user_message.lower() == "restart":
        chat_history.clear()
        cancel_event = PIPELINE_CANCEL_EVENTS.pop(user_socket_id, None)
        if cancel_event:
            cancel_event.set()
            socketio.emit("pipeline_cancelled", {}, room=user_socket_id)
        emit_to_requesting_user(
            "set_context",
            {"title": None, "processing": False, "search_complete": False},
        )
        emit_to_requesting_user(
            "receive_message",
            {
                "sender": "bot",
                "message": "Conversation has been reset. You can start a new search by providing a research title.",
            },
        )
        return

    # Record the user's message after handling reset
    chat_history.append({"sender": "user", "message": user_message})

    # Process the message as a research title if:
    # 1. It starts with "Research:" or "Research title:"
    # 2. OR It's a longer message that doesn't look like a year
    if (
        user_message.lower().startswith("research:")
        or user_message.lower().startswith("research title:")
        or "graph neural network"
        in user_message.lower()  # Example specific to your case
        or (len(user_message) > 20 and not user_message.isdigit())
    ):

        # Extract the title
        if ":" in user_message:
            title = user_message.split(":", 1)[1].strip()
        else:
            title = user_message.strip()

        print(f"Extracted title: '{title}'")
        emit_to_requesting_user(
            "receive_message",
            {
                "sender": "bot",
                "message": f"I'll help you find relevant papers for: '{title}'. What year would you like to use as a cutoff?",
            },
        )
        emit_to_requesting_user(
            "set_context",
            {"title": title, "processing": False, "search_complete": False},
        )
        return

    if context.get("title") and (
        user_message.isdigit()
        or "to" in user_message
        or "year:" in user_message.lower()
    ):
        # Extract year or year range
        year_text = user_message.lower().replace("year:", "").strip()

        year_to_use_str = "2020"  # Default year
        if "to" in year_text:
            parts = year_text.split("to")
            if (
                len(parts) == 2
                and parts[0].strip().isdigit()
                and len(parts[0].strip()) == 4
            ):
                year_to_use_str = parts[0].strip()
        elif year_text.isdigit() and len(year_text) == 4:  # Ensure it's a 4-digit year
            year_to_use_str = year_text
        else:  # If not a clear 4-digit year, prompt again or clarify
            emit_to_requesting_user(
                "receive_message",
                {
                    "sender": "bot",
                    "message": "Please provide a valid 4-digit year for the cutoff (e.g., 2015).",
                },
            )
            return  # Don't start pipeline if year is not clear

        title_to_process = context.get(
            "title"
        )  # Use a different variable name than the outer 'title' if it exists

        # Generate progress message (Corrected: separate calls)
        emit_to_requesting_user(
            "receive_message",
            {
                "sender": "bot",
                "message": f"🚀 Starting paper search for '{title_to_process}' (≥{year_to_use_str}). This will take several minutes. I'll keep you updated on my progress!",
            },
        )

        # Set processing flag (Corrected: separate call, and ensure context uses title_to_process)
        emit_to_requesting_user(
            "set_context",
            {"title": title_to_process, "processing": True, "search_complete": False},
        )

        # Run the pipeline in the background with cancellation support
        cancel_event = threading.Event()
        PIPELINE_CANCEL_EVENTS[user_socket_id] = cancel_event
        socketio.start_background_task(
            run_pipeline,
            title_to_process,
            int(year_to_use_str),
            user_socket_id,
            cancel_event,
        )
        return

    # to be improved--
    # Handle general questions after search is complete
    if context.get("search_complete"):
        # Handle citation-related questions
        if "citation" in user_message.lower() or "cite" in user_message.lower():
            emit_to_requesting_user(
                "receive_message",
                {
                    "sender": "bot",
                    "message": "📚 **Citation Help:**\n\nI've already provided APA citations for all the papers I found. Here are some tips:\n\n• **APA Format**: Author, A. A. (Year). Title of paper. DOI or URL\n• **Copy Citations**: You can copy the APA citations I provided directly into your reference list\n• **Bibliography**: I've also generated a complete bibliography at the end of your results\n\n**Need other formats?** I can help explain MLA, Chicago, or IEEE citation styles if needed!",
                },
            )
        elif "mla" in user_message.lower():
            emit_to_requesting_user(
                "receive_message",
                {
                    "sender": "bot",
                    "message": '📖 **MLA Citation Format:**\n\nMLA format looks like this:\n**Author Last, First. "Title of Paper." *Journal Name*, vol. #, no. #, Year, pp. ##-##. DOI or URL.**\n\nExample:\n*Smith, John. "Machine Learning Applications." AI Research Journal, vol. 15, no. 3, 2023, pp. 45-62. https://doi.org/10.1000/example.*\n\nWould you like me to help you convert any specific citations to MLA format?',
                },
            )
        elif "chicago" in user_message.lower():
            emit_to_requesting_user(
                "receive_message",
                {
                    "sender": "bot",
                    "message": '📑 **Chicago Citation Format:**\n\n**Notes-Bibliography Style:**\nAuthor Last, First. "Title of Paper." Journal Name vol. #, no. # (Year): ##-##. DOI or URL.\n\n**Author-Date Style:**\nAuthor Last, First. Year. "Title of Paper." Journal Name vol. # (no. #): ##-##. DOI or URL.\n\nExample (Author-Date):\n*Smith, John. 2023. "Machine Learning Applications." AI Research Journal 15 (3): 45-62. https://doi.org/10.1000/example.*',
                },
            )
        # Handle general chat questions
        elif user_message.lower() in ["hi", "hello", "hey"]:
            emit_to_requesting_user(
                "receive_message",
                {
                    "sender": "bot",
                    "message": "Hello! I'm Authematic, your literature curation assistant. I can help you find relevant academic papers for your research. How can I help you today?",
                },
            )
        elif "who are you" in user_message.lower():
            emit_to_requesting_user(
                "receive_message",
                {
                    "sender": "bot",
                    "message": "I'm Authematic, an AI-powered literature curation assistant designed to help researchers find relevant academic papers. I use natural language processing and semantic ranking to identify the most relevant papers for your research topics, and I provide properly formatted APA citations for all results!",
                },
            )
        elif (
            "how do you work" in user_message.lower()
            or "what can you do" in user_message.lower()
        ):
            emit_to_requesting_user(
                "receive_message",
                {
                    "sender": "bot",
                    "message": "I help you find academic papers by:\n\n1. Analyzing your research title\n2. Generating relevant academic topics and keywords\n3. Searching multiple sources (arXiv, Semantic Scholar, etc.)\n4. Filtering and ranking papers using SciBERT embeddings\n5. Presenting you with focused and exploratory results\n6. **Providing APA citations** for all papers found\n7. **Generating a complete bibliography** ready for your references\n\nTo start, just type 'Research title: [your research topic]'",
                },
            )
        else:
            # For any other message, prompt for a research title
            emit_to_requesting_user(
                "receive_message",
                {
                    "sender": "bot",
                    "message": "I'm designed to help with academic literature curation and citations. If you'd like to search for papers, please provide a research title by saying 'Research title: [your title]'",
                },
            )
        return

    # Handle other queries when we don't have a complete context
    # If we're processing, just acknowledge
    if context.get("processing"):
        emit_to_requesting_user(
            "receive_message",
            {
                "sender": "bot",
                "message": "I'm still working on your search! Please be patient as I scan through thousands of academic papers. This usually takes 30 minutes. For now, take a break while you wait.",
            },
        )
    # If we have a title but not processing yet, assume this is a year
    elif context.get("title") and not context.get("processing"):
        # Try to extract a year, or use default
        year = "2020"  # Default
        if user_message.isdigit():
            year = user_message

        title = context.get("title")
        emit_to_requesting_user(
            "receive_message",
            {
                "sender": "bot",
                "message": f"🚀 Starting paper search for '{title}' (≥{year}). This will take 2-3 minutes. I'll keep you updated!",
            },
        )
        emit_to_requesting_user(
            "set_context", {"processing": True, "search_complete": False}
        )
        cancel_event = threading.Event()
        PIPELINE_CANCEL_EVENTS[user_socket_id] = cancel_event
        socketio.start_background_task(
            run_pipeline,
            title,
            int(year),
            user_socket_id,
            cancel_event,
        )
    # Otherwise prompt for research title
    else:
        emit_to_requesting_user(
            "receive_message",
            {
                "sender": "bot",
                "message": "I can help you find relevant academic papers with properly formatted citations. Please provide a research title by saying 'Research title: [your title]'",
            },
        )


def _find_referenced_papers(
    query: str, papers: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Finds papers referenced in the query by number (e.g., "paper 1") or author.
    """
    referenced_papers = []

    # 1. Find references by number (e.g., "paper 1", "the second paper")
    numerical_references = re.findall(r"(?:paper|the)\s*(\d+)", query.lower())
    for num_str in numerical_references:
        try:
            paper_index = int(num_str) - 1  # Convert to 0-based index
            if 0 <= paper_index < len(papers):
                referenced_papers.append(papers[paper_index])
        except (ValueError, IndexError):
            continue

    # 2. Find references by author's last name
    for i, paper in enumerate(papers):
        authors = paper.get("authors", [])
        for author_name in authors:
            if not author_name or not isinstance(author_name, str):
                continue  # Skip to the next author if this one is empty or invalid.
            last_name = author_name.split(",")[0].split()[-1].lower()
            if (
                f"{last_name}'s paper" in query.lower()
                or f"({last_name})" in query.lower()
            ):
                referenced_papers.append(paper)
                break

    # Remove duplicates
    unique_refs = []
    seen_dois = set()
    for paper in referenced_papers:
        doi = paper.get("doi")
        if doi and doi not in seen_dois:
            unique_refs.append(paper)
            seen_dois.add(doi)

    return unique_refs


# This is the new handler for the chat sidebar
@socketio.on("sidebar_chat_message")
def handle_sidebar_chat(data):
    user_query = data["query"]
    context = data["context"]
    session_id = request.sid

    user_session_data = USER_SESSIONS.setdefault(session_id, {})
    chat_history = user_session_data.setdefault("chat_history", [])
    final_papers = user_session_data.get("papers", [])
    rag_collection_name = user_session_data.get("rag_collection_name")

    chat_history.append({"sender": "user", "message": user_query})

    # --- Pre-processing step for paper references (This part is correct) ---
    referenced_papers = _find_referenced_papers(user_query, final_papers)

    final_query_for_llm = user_query

    if referenced_papers:
        injected_context = "\n\n--- CONTEXT FROM REFERENCED PAPERS ---\n"
        for i, paper in enumerate(referenced_papers, 1):
            injected_context += (
                f"Referenced Paper {i} (Title: {paper.get('title', 'N/A')}):\n"
            )
            injected_context += (
                f"Abstract: {paper.get('abstract', 'No abstract available.')}\n\n"
            )

        final_query_for_llm = (
            f"{injected_context}" f"--- USER'S QUESTION ---\n" f"{user_query}"
        )

    # --- Routing Logic with the fix ---
    if context["type"] == "general":
        if not rag_collection_name:
            response_text = "The RAG memory for this session could not be built. General chat is disabled."
        else:
            # --- THIS IS THE FIX ---
            # We must pass the 'final_query_for_llm' which may contain injected context.
            response_text = get_general_rag_response(
                final_query_for_llm, rag_collection_name
            )
            # --- END OF FIX ---

    elif context["type"] == "specific" and context.get("doi"):
        active_paper = next(
            (p for p in final_papers if p.get("doi") == context["doi"]), None
        )
        # For specific chat, we correctly use the original query
        response_text = get_single_paper_response(user_query, active_paper)
    else:
        response_text = "An error occurred. Invalid chat context."

    chat_history.append({"sender": "bot", "message": response_text})

    socketio.emit("sidebar_chat_response", {"message": response_text}, room=session_id)


# ===== PIPELINE HELPER FUNCTIONS =====


def _generate_and_refine_keywords(
    title: str, emit_progress: callable
) -> Dict[str, Dict[str, List[str]]]:
    """
    PIPELINE HELPER - PHASE 1:
    Generates and refines topics, subthemes, and keywords based on the research title.

    Args:
        title (str): The user's research title.
        emit_progress (callable): A function to send progress updates to the frontend.

    Returns:
        Dict[str, Dict[str, List[str]]]: A nested dictionary of critiqued keywords.
    """
    emit_progress("📊 **Phase 1: Generating Topics, Subthemes & Keywords...**")

    related_topics: List[str] = generate_topics(title)
    emit_progress(f"-> Generated {len(related_topics)} research topics.")
    if related_topics:
        topics_str = ", ".join(related_topics)
        emit_progress(f"🔸 Topics: {topics_str}")

    subthemes_by_topic: Dict[str, List[str]] = generate_subthemes(
        related_topics, max_subthemes=3
    )
    emit_progress(f"-> Generated subthemes for topics.")

    raw_keywords_nested: Dict[str, Dict[str, List[str]]] = (
        generate_keywords_by_subtheme(subthemes_by_topic, max_terms=5)
    )
    emit_progress(f"-> Generated raw keywords for all subthemes.")

    emit_progress("🤖 Refining keywords with Critic AI...")
    critiqued_keywords_nested: Dict[str, Dict[str, List[str]]] = {}
    for topic_key, subthemes_map in raw_keywords_nested.items():
        refined_subthemes_for_topic: Dict[str, List[str]] = {}
        for subtheme_key, original_keywords in subthemes_map.items():
            critic_label = (
                f"Keywords for topic '{topic_key}' / subtheme '{subtheme_key}'"
            )
            # The second return value (suggestions) is not used in the main pipeline, so we use `_`.
            refined_list, _ = critique_list(critic_label, original_keywords)
            refined_subthemes_for_topic[subtheme_key] = refined_list
        critiqued_keywords_nested[topic_key] = refined_subthemes_for_topic
    emit_progress("✅ Keywords refined and ready for search.")

    return critiqued_keywords_nested


def _collect_and_filter_papers(
    keywords: Dict, cutoff_year: int, emit_progress: callable
) -> List[Dict]:
    """
    PIPELINE HELPER - PHASE 2 & 3 (PART 1):
    Uses the provided keywords to collect papers from multiple sources, then runs
    the initial filtering and deduplication steps.

    Args:
        keywords (Dict): The nested dictionary of keywords from the previous phase.
        cutoff_year (int): The minimum publication year for papers.
        emit_progress (callable): A function to send progress updates to the frontend.

    Returns:
        List[Dict]: A clean list of unique paper candidates with valid DOIs and abstracts.
    """
    # === Phase 2: Paper Collection ===
    emit_progress(
        "📚 **Phase 2: Collecting Academic Papers...** (This is the longest step)"
    )
    papers_to_fetch_per_keyword_source: int = 3
    collected_papers, raw_hit_count, bucket_paper_count = collect_papers(
        keywords_by_topic=keywords,
        cutoff_year=cutoff_year,
        paper_per_keyword=papers_to_fetch_per_keyword_source,
    )
    emit_progress(f"-> Raw search returned {raw_hit_count} entries.")
    emit_progress(
        f"-> Consolidated to {bucket_paper_count} papers before enrichment/filtering."
    )
    emit_progress(
        f"-> {len(collected_papers)} papers remain after enrichment and global filtering."
    )

    # === Phase 3: Initial Filtering ===
    emit_progress("🧹 **Phase 3: Performing Initial Filtering...**")

    # Process in memory instead of saving to/loading from JSON
    papers_after_doi_filter = filter_by_doi(collected_papers)
    papers_after_abstract_filter = filter_by_abstract(papers_after_doi_filter)
    papers_after_initial_filters = dedupe_by_doi(papers_after_abstract_filter)

    emit_progress(
        f"✅ Initial filtering complete. {len(papers_after_initial_filters)} papers remain."
    )

    return papers_after_initial_filters


def _generate_thematic_terms(
    title: str, emit_progress: callable
) -> Tuple[List[str], List[str], List[str]]:
    """
    PIPELINE HELPER - PHASE 3 (PART 2) & 4 (PART 1):
    Generates, critiques, and cleans the thematic terms (Domain, Application, Technique)
    that are used for ranking and categorization.

    Args:
        title (str): The user's research title.
        emit_progress (callable): A function to send progress updates to the frontend.

    Returns:
        Tuple[List[str], List[str], List[str]]: A tuple containing the final lists of
                                                 application, technique, and domain terms.
    """
    emit_progress("🛠️ **Phase 4: Generating Thematic Terms...**")

    # === Domain Terms ===
    raw_domain_terms: List[str] = generate_domain_terms(title, max_terms=15)
    critiqued_domain_terms, domain_suggestions = critique_list(
        f"Domain terms for {title}", raw_domain_terms
    )
    final_domain_terms: List[str] = clean_terms(critiqued_domain_terms)
    if not final_domain_terms and critiqued_domain_terms:
        final_domain_terms = critiqued_domain_terms[:10]  # Fallback
    elif not final_domain_terms:
        final_domain_terms = raw_domain_terms[:10]  # Deeper Fallback
    emit_progress("-> Domain terms generated and cleaned.")

    # === Application Terms ===
    raw_app_terms: List[str] = generate_app_terms(title, max_terms=7)
    critiqued_app_terms, app_suggestions = critique_list(
        f"Application terms for {title}", raw_app_terms
    )
    final_app_terms: List[str] = clean_terms(critiqued_app_terms)
    final_app_terms = [
        t for t in final_app_terms if len(t.split()) > 1
    ]  # Keep only phrases
    title_lower_stripped = title.lower().strip()
    if title_lower_stripped not in final_app_terms:
        final_app_terms.insert(0, title_lower_stripped)
    if not final_app_terms:  # Fallback
        final_app_terms = raw_app_terms[:5]
    emit_progress("-> Application terms generated and cleaned.")

    # === Technique Terms ===
    raw_tech_terms: List[str] = generate_tech_terms(title, max_terms=10)
    critiqued_tech_terms, tech_suggestions = critique_list(
        f"Technique terms for {title}", raw_tech_terms
    )
    final_tech_terms: List[str] = clean_terms(critiqued_tech_terms)
    if not final_tech_terms and critiqued_tech_terms:
        final_tech_terms = critiqued_tech_terms[:3]  # Fallback
    elif not final_tech_terms:
        final_tech_terms = raw_tech_terms[:5]  # Deeper Fallback
    emit_progress("-> Technique terms generated and cleaned.")

    # Log the final terms that will be used for analysis
    emit_progress(f"🔑 Core Application Terms: {final_app_terms}")
    emit_progress(f"🔑 Core Technique Terms: {final_tech_terms}")
    emit_progress(f"🔑 Core Domain Terms: {final_domain_terms}")

    emit_progress("✅ Thematic terms finalized.")

    return (
        final_app_terms,
        final_tech_terms,
        final_domain_terms,
        list(app_suggestions.values()),
        list(tech_suggestions.values()),
        list(domain_suggestions.values()),
    )


def _rank_and_categorize_papers(
    papers: List[Dict],
    title: str,
    app_terms: List[str],
    tech_terms: List[str],
    domain_terms: List[str],
    app_suggestions: List[str],
    tech_suggestions: List[str],
    domain_suggestions: List[str],
    emit_progress: callable,
) -> Tuple[List[Dict], List[Dict]]:
    """
    PIPELINE HELPER - PHASES 4, 5, 6 (CORRECTED)
    Performs term clustering, semantic ranking, score boosting, and final categorization.
    """
    emit_progress("🧬 **Phase 4: Clustering & Preparing Terms...**")

    # --- Corrected: Use expanded term lists for matching ---
    app_terms_for_patterns = list(set(app_terms + app_suggestions))
    tech_terms_for_patterns = list(set(tech_terms + tech_suggestions))
    domain_terms_for_patterns = list(set(domain_terms + domain_suggestions))

    # --- Term Clustering (same as before) ---
    representative_tech_terms = tech_terms_for_patterns
    if tech_terms and len(tech_terms) > 4:
        valid_tech_terms = [term for term in tech_terms if term.strip()]
        if len(valid_tech_terms) > 4:
            tech_term_vectors = [
                embed_text(term, use_cache=True) for term in valid_tech_terms
            ]
            valid_tech_term_vectors = [
                vec for vec in tech_term_vectors if vec is not None
            ]
            terms_actually_embedded = [
                term
                for term, vec in zip(valid_tech_terms, tech_term_vectors)
                if vec is not None
            ]

            if valid_tech_term_vectors and len(valid_tech_term_vectors) > 1:
                emit_progress(
                    "-> Finding representative technique terms via clustering..."
                )
                try:
                    num_clusters = min(5, max(2, len(terms_actually_embedded) // 2))
                    kmeans = KMeans(
                        n_clusters=num_clusters, random_state=0, n_init="auto"
                    )
                    kmeans.fit_predict(np.vstack(valid_tech_term_vectors))
                    # Simplified logic to grab representative terms for the pipeline
                    # In a real system, you might pick terms closest to centroids
                    representative_tech_terms = list(set(terms_actually_embedded))[
                        :10
                    ]  # Cap for performance
                    emit_progress(
                        f"-> Clustered terms into {num_clusters} groups. Using top representatives."
                    )
                except Exception as e_cluster:
                    emit_progress(
                        f"⚠️ Warning: Technique term clustering failed: {e_cluster}"
                    )

    # --- Phase 5: Semantic Ranking & Boosting ---
    emit_progress("⚖️ **Phase 5: Semantic Ranking & Score Boosting...**")
    if not papers:
        emit_progress("⚠️ No papers available for ranking.")
        return [], []

    semantically_ranked_papers = semantic_rank_papers(title, papers, top_n=None)
    emit_progress(
        f"-> Semantic ranking complete for {len(semantically_ranked_papers)} papers."
    )

    # Prepare regex with the CORRECT, expanded term lists
    app_regex = [
        re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        for term in app_terms_for_patterns
        if term
    ]
    tech_regex = [
        re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        for term in representative_tech_terms
        if term
    ]

    for paper in semantically_ranked_papers:
        text = f"{paper.get('title','')} {paper.get('abstract','')}".lower()
        matches_app = any(p.search(text) for p in app_regex)
        matches_tech = any(p.search(text) for p in tech_regex)
        if matches_app and matches_tech:
            paper["score"] = paper.get("score", 0.0) * 1.25
        elif matches_app or matches_tech:
            paper["score"] = paper.get("score", 0.0) * 1.10

    semantically_ranked_papers.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    emit_progress("-> Paper scores boosted based on thematic relevance.")

    emit_progress("📋 **Phase 6: Categorizing Papers...**")

    domain_regex = [
        re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        for term in domain_terms_for_patterns
        if term
    ]

    def check_matches_app(p_dict: Dict) -> bool:
        txt = (p_dict.get("title", "") + " " + p_dict.get("abstract", "")).lower()
        return any(pat.search(txt) for pat in app_regex)

    def check_matches_tech(p_dict: Dict) -> bool:
        txt = (p_dict.get("title", "") + " " + p_dict.get("abstract", "")).lower()
        return any(pat.search(txt) for pat in tech_regex)

    def count_domain_hits(p_dict: Dict) -> int:
        txt = (p_dict.get("title", "") + " " + p_dict.get("abstract", "")).lower()
        return sum(bool(dp.search(txt)) for dp in domain_regex)

    DESIRED_FOCUSED_COUNT = 20
    DESIRED_EXPLORATORY_COUNT = 10
    focused_papers = []

    # --- Corrected: Restore the complete tiered logic ---
    # Tier 1: App AND Tech AND Domain (strictest)
    for p in semantically_ranked_papers:
        if len(focused_papers) >= DESIRED_FOCUSED_COUNT:
            break
        if check_matches_app(p) and check_matches_tech(p) and count_domain_hits(p) >= 1:
            focused_papers.append(p)

    # Tier 2: App AND Tech (fallback)
    if len(focused_papers) < DESIRED_FOCUSED_COUNT:
        for p in semantically_ranked_papers:
            if len(focused_papers) >= DESIRED_FOCUSED_COUNT:
                break
            if (
                p.get("doi") not in {fp.get("doi") for fp in focused_papers}
                and check_matches_app(p)
                and check_matches_tech(p)
            ):
                focused_papers.append(p)

    # Tier 3: Domain AND (App OR Tech) (CRITICAL MISSING FALLBACK)
    if len(focused_papers) < DESIRED_FOCUSED_COUNT:
        for p in semantically_ranked_papers:
            if len(focused_papers) >= DESIRED_FOCUSED_COUNT:
                break
            if (
                p.get("doi") not in {fp.get("doi") for fp in focused_papers}
                and count_domain_hits(p) >= 1
                and (check_matches_app(p) or check_matches_tech(p))
            ):
                focused_papers.append(p)

    final_focused_papers = focused_papers[:DESIRED_FOCUSED_COUNT]
    focused_paper_dois = {p["doi"] for p in final_focused_papers if p.get("doi")}

    # Logic to find exploratory papers
    exploratory_candidates = [
        p for p in semantically_ranked_papers if p.get("doi") not in focused_paper_dois
    ]
    exploratory_papers = []
    for p in exploratory_candidates:
        if len(exploratory_papers) >= DESIRED_EXPLORATORY_COUNT:
            break
        if count_domain_hits(p) >= 2 or (
            count_domain_hits(p) >= 1 and check_matches_tech(p)
        ):
            exploratory_papers.append(p)
    if len(exploratory_papers) < DESIRED_EXPLORATORY_COUNT:
        for p in exploratory_candidates:
            if len(exploratory_papers) >= DESIRED_EXPLORATORY_COUNT:
                break
            if p not in exploratory_papers and count_domain_hits(p) >= 1:
                exploratory_papers.append(p)

    final_exploratory_papers = exploratory_papers[:DESIRED_EXPLORATORY_COUNT]

    emit_progress(
        f"✅ Categorization complete: {len(final_focused_papers)} Focused, {len(final_exploratory_papers)} Exploratory."
    )

    return final_focused_papers, final_exploratory_papers


def _enrich_papers_with_insights(
    papers_list: List[Dict],
    title: str,
    emit_progress: callable,
    category: str = "",
    cancel_event: Optional[threading.Event] = None,
):
    """
    PIPELINE HELPER - PHASE 7:
    Enriches a list of papers with AI-generated summaries and relevance scores.
    This function modifies the paper dictionaries in-place.
    """
    if not papers_list:
        return

    category_prefix = f" for {category}" if category else ""
    emit_progress(
        f"💡 **Phase 7: Generating AI Insights{category_prefix} ({len(papers_list)} papers)...**"
    )
    for i, paper in enumerate(papers_list):
        if cancel_event and cancel_event.is_set():
            emit_progress("Insight generation cancelled.")
            return
        # We can emit less frequently here to avoid cluttering the log
        if (i + 1) % 5 == 0:
            emit_progress(f"  -> Analyzing paper {i+1}/{len(papers_list)}...")

        abstract = paper.get("abstract")
        if abstract and isinstance(abstract, str) and abstract.strip():
            insights = generate_insights(title, abstract)
            paper["insights_summary"] = insights.get("summary", "Not available.")
            paper["insights_relevance"] = insights.get("relevance", "Not available.")
            socketio.sleep(2.1)  # Respect API rate limits
        else:
            paper["insights_summary"] = (
                "Insight generation skipped: Abstract was missing."
            )
            paper["insights_relevance"] = (
                "Relevance assessment skipped: Abstract was missing."
            )
    emit_progress("✅ AI insights generated.")


# == NEW RUN PIPELINE (MODULARIZED, CLEANED) ==


def run_pipeline(
    title: str, cutoff_year: int, user_socket_id: str, cancel_event: threading.Event
):
    """
    Main pipeline orchestrator.
    This function calls the modular helper functions in sequence to perform the
    full literature search and analysis, then prepares the results for the frontend.
    """

    # Define the progress emitter to send updates to the specific user
    def emit_progress(message: str):
        socketio.emit("progress_update", {"message": message}, room=user_socket_id)

    def emit_log(message: str):
        socketio.emit("server_log", {"message": message}, room=user_socket_id)

    def check_cancel() -> bool:
        if cancel_event.is_set():
            emit_progress("⏹ Pipeline cancelled by user.")
            socketio.emit("pipeline_cancelled", {}, room=user_socket_id)
            socketio.emit("set_context", {"processing": False}, room=user_socket_id)
            return True
        return False

    try:
        with tee_stdout(emit_log):
            pipeline_start_time = time.time()
            emit_progress(
                f"🚀 **Starting full literature search for:** '{title}' (papers ≥ {cutoff_year})"
            )
            if check_cancel():
                return

            # Step 1: Generate and Refine Keywords
            keywords = _generate_and_refine_keywords(title, emit_progress)
            if check_cancel():
                return

            # Step 2: Collect & Filter Papers
            papers = _collect_and_filter_papers(keywords, cutoff_year, emit_progress)
            if check_cancel():
                return

            # Step 3: Generate Thematic Terms - now gets 6 return values
            app_terms, tech_terms, domain_terms, app_sug, tech_sug, domain_sug = (
                _generate_thematic_terms(title, emit_progress)
            )
            if check_cancel():
                return

            # Step 4: Rank & Categorize Papers - pass the suggestions
            focused_papers, exploratory_papers = _rank_and_categorize_papers(
                papers,
                title,
                app_terms,
                tech_terms,
                domain_terms,
                app_sug,
                tech_sug,
                domain_sug,
                emit_progress,
            )
            if check_cancel():
                return

            # Tag each paper with its category for the frontend
            for p in focused_papers:
                p["category"] = "focused"
            for p in exploratory_papers:
                p["category"] = "exploratory"

            # Step 5: Enrich both lists with AI-powered insights
            _enrich_papers_with_insights(
                focused_papers,
                title,
                emit_progress,
                category="Focused papers",
                cancel_event=cancel_event,
            )
            if check_cancel():
                return
            _enrich_papers_with_insights(
                exploratory_papers,
                title,
                emit_progress,
                category="Exploratory papers",
                cancel_event=cancel_event,
            )
            if check_cancel():
                return

            # --- Step 6: Prepare Final Results and Build the RAG Index ---

            all_final_papers = focused_papers + exploratory_papers

            emit_progress("🧠 **Finalizing: Building RAG memory...**")
            session_collection_name = f"rag_collection_{uuid.uuid4().hex}"
            index_built = build_rag_index(
                collection_name=session_collection_name, papers=all_final_papers
            )
            if check_cancel():
                return

            # Instead of using the Flask session, we store the data in our global dictionary
            # using the user's unique socket ID as the key.
            USER_SESSIONS[user_socket_id] = {
                "papers": all_final_papers,
                "rag_collection_name": session_collection_name if index_built else None,
                "rag_index_built": index_built,
            }

            if index_built:
                emit_progress(
                    "✅ RAG memory is ready. The 'General Chat' tab is now enabled."
                )
            else:
                emit_progress(
                    "⚠️ RAG memory could not be built. 'General Chat' will be disabled."
                )

            # Pre-format citations on the backend before sending
            for paper in all_final_papers:
                paper["formatted_citation"] = format_apa_citation(paper)
            if check_cancel():
                return

            # --- Step 7: Send the Final, Comprehensive Results to the Frontend ---

            elapsed_time = time.time() - pipeline_start_time
            minutes, seconds = divmod(int(elapsed_time), 60)
            emit_progress(f"🎉 **Pipeline complete in {minutes}m {seconds}s!**")

            # We emit a single 'results_ready' event with all the data.
            # This payload includes the papers and the crucial 'rag_collection_name'
            # that the frontend needs to store for future chat requests.
            intro_msg = (
                "General Chat lets you discuss the entire set of papers. "
                "Each paper tab focuses the conversation on that single paper."
            )

            socketio.emit(
                "set_context",
                {"processing": False, "search_complete": True},
                room=user_socket_id,
            )

            socketio.emit(
                "results_ready",
                {
                    "papers": all_final_papers,
                    "rag_collection_name": session_collection_name,
                    "initial_chat_history": USER_SESSIONS.get(user_socket_id, {}).get(
                        "chat_history", []
                    ),
                    "intro_message": intro_msg,
                },
                room=user_socket_id,
            )

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"ERROR: Unhandled exception in app.py run_pipeline: {error_details}")
        emit_progress(
            f"❌ **An unexpected error occurred:**\n{str(e)}\nPlease try 'reset'."
        )
        # Ensure the frontend knows the process has failed.
        socketio.emit("pipeline_failed", {}, room=user_socket_id)
    finally:
        # Clean up the cancel event for this user if it exists
        PIPELINE_CANCEL_EVENTS.pop(user_socket_id, None)


# Add a simple health check route for testing
@app.route("/health")
def health_check():
    return jsonify(
        {"status": "ok", "gemini_api_key_set": bool(os.environ.get("GEMINI_API_KEY"))}
    )


# Add a simple test route to verify Gemini API is working
@app.route("/test-gemini")
def test_gemini():
    try:
        from google import genai

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        # Simple test prompt
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents="What is 2+2? Give just the number."
        )
        return jsonify({"status": "ok", "gemini_response": response.text.strip()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# Add route to manually test topic generation
@app.route("/test-topics")
def test_topics():
    title = request.args.get("title", "Machine learning in healthcare")
    try:
        topics = generate_topics(title)
        return jsonify({"status": "ok", "title": title, "topics": topics})
    except Exception as e:
        return jsonify(
            {"status": "error", "message": str(e), "traceback": traceback.format_exc()}
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 for local dev
    print("Starting Authematic web application...")
    socketio.run(app, host="0.0.0.0", port=port, debug=True)
