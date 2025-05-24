import math
import os
import re
import sys
import json
import traceback
import time
from pathlib import Path
from typing import Any, List, Dict, Tuple

# Third-party imports
import numpy as np
from sklearn.cluster import KMeans


# --- 1. Load .env file from project root ---
# This MUST be done BEFORE importing any project modules that might need these env vars
# (like api_client_manager via paper_collector, keyword_critic, etc.)
from dotenv import load_dotenv # Import load_dotenv here
project_root = Path(__file__).resolve().parent.parent # Get project root directory
dotenv_path = project_root / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
    print(f"INFO: app.py loaded .env file from: {dotenv_path}")
else:
    print(f"WARNING: app.py did not find .env file at: {dotenv_path}")

# For debugging during merge - check if keys are loaded as expected by api_client_manager
# These are the keys your api_client_manager.py expects.
# print(f"INFO: GEMINI_API_KEY_1 after load_dotenv in app.py: {'SET' if os.getenv('GEMINI_API_KEY_1') else 'NOT SET'}")
# print(f"INFO: GEMINI_API_KEY_2 after load_dotenv in app.py: {'SET' if os.getenv('GEMINI_API_KEY_2') else 'NOT SET'}")
# --- End .env loading ---



# --- 3. Add Project Root to sys.path for imports ---
# This allows importing your backend modules from the parent directory (project root)
sys.path.append(str(project_root))
print(f"INFO: Project root '{str(project_root)}' added to sys.path.")

# --- 4. Import Flask, SocketIO, and YOUR Up-to-Date Backend Modules ---
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO

# Import your project's modules (ensure these are the cleaned versions)
# Note: api_client_manager.py will be used internally by these modules.
from paper_collector import (
    generate_topics,
    generate_subthemes,             # New function to import
    generate_keywords_by_subtheme,  # New function to import
    collect_papers,
    generate_domain_terms,
    generate_app_terms,             # Ensure this is imported if used by app.py's run_pipeline
    generate_tech_terms             # Ensure this is imported if used by app.py's run_pipeline
)
from filter_and_rank import (
    filter_by_doi, 
    filter_by_abstract, 
    dedupe_by_doi, 
    semantic_rank_papers
    # Add other functions from filter_and_rank if your app.py's run_pipeline will use them e.g. load_candidates_from_json
)
from keyword_critic import critique_list # For critiquing terms
# from embeddings import embed_text 
from embeddings import embed_text
from extract_insights import generate_insights

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'authematic-secret-key' # Good to keep this configurable or more random for production
socketio = SocketIO(app)

# --- Constants for Term Cleaning (from run_pipeline.py) ---
STOP_TIER1 = {
    "survey", "review", "framework", "architecture", "architectures",
    "analysis", "analyses", "system", "systems",
}
STOP_TIER2 = {
    "method", "methods", "approach", "approaches",
    "algorithm", "algorithms", "technique", "techniques",
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
        title = paper.get('title', 'Untitled').strip()
        year = paper.get('year', 'n.d.')
        doi = paper.get('doi', '')
        authors = paper.get('authors', [])
        
        # Format authors
        if not authors:
            author_text = "Author, A."
        elif len(authors) == 1:
            author_text = format_author_apa(authors[0])
        elif len(authors) == 2:
            author_text = f"{format_author_apa(authors[0])}, & {format_author_apa(authors[1])}"
        elif len(authors) <= 20:
            formatted_authors = [format_author_apa(author) for author in authors[:-1]]
            author_text = ", ".join(formatted_authors) + f", & {format_author_apa(authors[-1])}"
        else:
            # For more than 20 authors, list first 19, then "...", then last author
            formatted_authors = [format_author_apa(author) for author in authors[:19]]
            author_text = ", ".join(formatted_authors) + f", ... {format_author_apa(authors[-1])}"
        
        # Clean and format title
        if not title.endswith('.'):
            title += '.'
        
        # Format DOI or URL
        if doi and doi.startswith('https://doi.org/'):
            doi_text = doi
        elif doi and not doi.startswith('http'):
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
    result_msg = f"**{index}. {paper.get('title', 'Untitled')}** ({paper.get('year', 'N/A')})\n"
    
    authors = paper.get('authors', [])
    if authors:
        author_text = ', '.join(authors[:3]) # Show first 3 authors
        if len(authors) > 3:
            author_text += f", et al." # Indicate more authors
        result_msg += f"   👥 **Authors:** {author_text}\n"
    
    result_msg += f"   🔗 **DOI:** {paper.get('doi', 'N/A')}\n"
    result_msg += f"   📊 **Relevance Score:** {paper.get('score', 0.0):.4f}\n"
    
    # APA Citation
    apa_citation = format_apa_citation(paper) # Assumes format_apa_citation is defined
    result_msg += f"   📚 **APA Citation:** {apa_citation}\n"

    # Add AI Insights
    if paper.get('insights_summary') and paper.get('insights_summary') not in ["Summary not available.", "Insight generation skipped: Abstract was missing or empty."]:
        result_msg += f"   💡 **AI Summary:** {paper['insights_summary']}\n"
    if paper.get('insights_relevance') and paper.get('insights_relevance') not in ["Relevance assessment not available.", "Relevance assessment skipped: Abstract was missing or empty."]:
        result_msg += f"   🎯 **AI Relevance to '{paper.get('query_title', 'your research')}':** {paper['insights_relevance']}\n" # Assuming query_title is available or use generic

    return result_msg

def format_author_apa(author_name):
    """
    Format a single author name in APA style (Last, F. M.)
    
    Args:
        author_name (str): Full author name
    
    Returns:
        str: APA formatted author name
    """
    if not author_name or not isinstance(author_name, str):
        return "Author, A."
    
    # Clean the name
    name = author_name.strip()
    
    # Handle different name formats
    if ',' in name:
        # Already in "Last, First" format
        parts = name.split(',', 1)
        last_name = parts[0].strip()
        first_part = parts[1].strip() if len(parts) > 1 else ""
    else:
        # Assume "First Last" or "First Middle Last" format
        name_parts = name.split()
        if len(name_parts) == 1:
            return f"{name_parts[0]}, A."
        else:
            last_name = name_parts[-1]
            first_part = " ".join(name_parts[:-1])
    
    # Format first names and initials
    if first_part:
        # Extract initials from first/middle names
        first_names = first_part.split()
        initials = []
        for fname in first_names:
            if fname and len(fname) > 0:
                initials.append(f"{fname[0].upper()}.")
        
        if initials:
            return f"{last_name}, {' '.join(initials)}"
        else:
            return f"{last_name}, A."
    else:
        return f"{last_name}, A."

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
    authors = paper.get('authors', [])
    if authors:
        author_text = ', '.join(authors[:3])
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

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('send_message')

def handle_message(data):
    user_socket_id = request.sid
    user_message = data.get('message', '')
    context = data.get('context', {})
    print(f"Received message: '{user_message}' with context: {context}")
    
    def emit_to_requesting_user(event_name, data_payload):
        socketio.emit(event_name, data_payload, room=user_socket_id)

    # Clear context on reset
    if user_message.lower() == 'reset' or user_message.lower() == 'restart':
        emit_to_requesting_user('set_context', {'title': None, 'processing': False, 'search_complete': False})
        emit_to_requesting_user('receive_message', {
            'sender': 'bot',
            'message': "Conversation has been reset. You can start a new search by providing a research title."
        })
        return
    
    # Clear context on reset
    if user_message.lower() == 'reset' or user_message.lower() == 'restart':
        emit_to_requesting_user('set_context', {'title': None, 'processing': False, 'search_complete': False})
        emit_to_requesting_user('receive_message', {
            'sender': 'bot',
            'message': "Conversation has been reset. You can start a new search by providing a research title."
        })
        return
    
    # Process the message as a research title if:
    # 1. It starts with "Research:" or "Research title:"
    # 2. OR It's a longer message that doesn't look like a year
    if (user_message.lower().startswith("research:") or 
        user_message.lower().startswith("research title:") or
        "graph neural network" in user_message.lower() or  # Example specific to your case
        (len(user_message) > 20 and not user_message.isdigit())):
        
        # Extract the title
        if ":" in user_message:
            title = user_message.split(":", 1)[1].strip()
        else:
            title = user_message.strip()
            
        print(f"Extracted title: '{title}'")
        emit_to_requesting_user('receive_message', {
            'sender': 'bot', 
            'message': f"I'll help you find relevant papers for: '{title}'. What year would you like to use as a cutoff?"
        })
        emit_to_requesting_user('set_context', {'title': title, 'processing': False, 'search_complete': False})
        return
    
    if context.get('title') and (user_message.isdigit() or 
                             "to" in user_message or 
                             "year:" in user_message.lower()):
        # Extract year or year range
        year_text = user_message.lower().replace("year:", "").strip()
        
        year_to_use_str = "2020"  # Default year
        if "to" in year_text:
            parts = year_text.split("to")
            if len(parts) == 2 and parts[0].strip().isdigit() and len(parts[0].strip()) == 4 :
                year_to_use_str = parts[0].strip()
        elif year_text.isdigit() and len(year_text) == 4: # Ensure it's a 4-digit year
            year_to_use_str = year_text
        else: # If not a clear 4-digit year, prompt again or clarify
            emit_to_requesting_user('receive_message', {
                'sender': 'bot',
                'message': "Please provide a valid 4-digit year for the cutoff (e.g., 2015)."
            })
            return # Don't start pipeline if year is not clear
            
        title_to_process = context.get('title') # Use a different variable name than the outer 'title' if it exists
        
        # Generate progress message (Corrected: separate calls)
        emit_to_requesting_user('receive_message', {
            'sender': 'bot',
            'message': f"🚀 Starting paper search for '{title_to_process}' (≥{year_to_use_str}). This will take several minutes. I'll keep you updated on my progress!"
        })
        
        # Set processing flag (Corrected: separate call, and ensure context uses title_to_process)
        emit_to_requesting_user('set_context', {'title': title_to_process, 'processing': True, 'search_complete': False})
        
        # Run the pipeline in the background - this part is correct
        socketio.start_background_task(run_pipeline, title_to_process, int(year_to_use_str), user_socket_id)
        return
    
    # Handle general questions after search is complete
    if context.get('search_complete'):
        # Handle citation-related questions
        if 'citation' in user_message.lower() or 'cite' in user_message.lower():
            emit_to_requesting_user('receive_message', {
                'sender': 'bot',
                'message': "📚 **Citation Help:**\n\nI've already provided APA citations for all the papers I found. Here are some tips:\n\n• **APA Format**: Author, A. A. (Year). Title of paper. DOI or URL\n• **Copy Citations**: You can copy the APA citations I provided directly into your reference list\n• **Bibliography**: I've also generated a complete bibliography at the end of your results\n\n**Need other formats?** I can help explain MLA, Chicago, or IEEE citation styles if needed!"
            })
        elif 'mla' in user_message.lower():
            emit_to_requesting_user('receive_message', {
                'sender': 'bot',
                'message': "📖 **MLA Citation Format:**\n\nMLA format looks like this:\n**Author Last, First. \"Title of Paper.\" *Journal Name*, vol. #, no. #, Year, pp. ##-##. DOI or URL.**\n\nExample:\n*Smith, John. \"Machine Learning Applications.\" AI Research Journal, vol. 15, no. 3, 2023, pp. 45-62. https://doi.org/10.1000/example.*\n\nWould you like me to help you convert any specific citations to MLA format?"
            })
        elif 'chicago' in user_message.lower():
            emit_to_requesting_user('receive_message', {
                'sender': 'bot',
                'message': "📑 **Chicago Citation Format:**\n\n**Notes-Bibliography Style:**\nAuthor Last, First. \"Title of Paper.\" Journal Name vol. #, no. # (Year): ##-##. DOI or URL.\n\n**Author-Date Style:**\nAuthor Last, First. Year. \"Title of Paper.\" Journal Name vol. # (no. #): ##-##. DOI or URL.\n\nExample (Author-Date):\n*Smith, John. 2023. \"Machine Learning Applications.\" AI Research Journal 15 (3): 45-62. https://doi.org/10.1000/example.*"
            })
        # Handle general chat questions
        elif user_message.lower() in ['hi', 'hello', 'hey']:
            emit_to_requesting_user('receive_message', {
                'sender': 'bot',
                'message': "Hello! I'm Authematic, your literature curation assistant. I can help you find relevant academic papers for your research. How can I help you today?"
            })
        elif 'who are you' in user_message.lower():
            emit_to_requesting_user('receive_message', {
                'sender': 'bot',
                'message': "I'm Authematic, an AI-powered literature curation assistant designed to help researchers find relevant academic papers. I use natural language processing and semantic ranking to identify the most relevant papers for your research topics, and I provide properly formatted APA citations for all results!"
            })
        elif 'how do you work' in user_message.lower() or 'what can you do' in user_message.lower():
            emit_to_requesting_user('receive_message', {
                'sender': 'bot',
                'message': "I help you find academic papers by:\n\n1. Analyzing your research title\n2. Generating relevant academic topics and keywords\n3. Searching multiple sources (arXiv, Semantic Scholar, etc.)\n4. Filtering and ranking papers using SciBERT embeddings\n5. Presenting you with focused and exploratory results\n6. **Providing APA citations** for all papers found\n7. **Generating a complete bibliography** ready for your references\n\nTo start, just type 'Research title: [your research topic]'"
            })
        else:
            # For any other message, prompt for a research title
            emit_to_requesting_user('receive_message', {
                'sender': 'bot',
                'message': "I'm designed to help with academic literature curation and citations. If you'd like to search for papers, please provide a research title by saying 'Research title: [your title]'"
            })
        return
    
    # Handle other queries when we don't have a complete context
    # If we're processing, just acknowledge
    if context.get('processing'):
        emit_to_requesting_user('receive_message', {
            'sender': 'bot',
            'message': "I'm still working on your search! Please be patient as I scan through thousands of academic papers. This usually takes 2-3 minutes."
        })
    # If we have a title but not processing yet, assume this is a year
    elif context.get('title') and not context.get('processing'):
        # Try to extract a year, or use default
        year = "2020"  # Default
        if user_message.isdigit():
            year = user_message
            
        title = context.get('title')
        emit_to_requesting_user('receive_message', {
            'sender': 'bot',
            'message': f"🚀 Starting paper search for '{title}' (≥{year}). This will take 2-3 minutes. I'll keep you updated!"
        })
        emit_to_requesting_user('set_context', {'processing': True, 'search_complete': False})
        socketio.start_background_task(run_pipeline, title, int(year), user_socket_id)
    # Otherwise prompt for research title
    else:
        emit_to_requesting_user('receive_message', {
            'sender': 'bot',
            'message': "I can help you find relevant academic papers with properly formatted citations. Please provide a research title by saying 'Research title: [your title]'"
        })


def run_pipeline(title: str, cutoff_year: int, user_socket_id: str):
    """
    Complete pipeline logic, adapted for the web app with SocketIO progress updates
    to a specific user.
    """
    # Helper to emit messages specifically to the user who made the request
    def emit_to_user(event_name: str, data_payload: Dict[str, Any]):
        socketio.emit(event_name, data_payload, room=user_socket_id)

    try:
        pipeline_start_time = time.time()
        emit_to_user('receive_message', {'sender': 'bot', 'message': f"🚀 Starting full literature search for: '{title}' (papers ≥ {cutoff_year}). This may take several minutes..."})

        # === Phase 1: Topic, Subtheme, and Keyword Generation & Critique ===
        emit_to_user('receive_message', {'sender': 'bot', 'message': "📊 **Phase 1/7: Generating Topics, Subthemes & Keywords...**"})
        
        related_topics: List[str] = generate_topics(title)
        emit_to_user('receive_message', {'sender': 'bot', 'message': f"✅ Generated {len(related_topics)} research topics."})

        subthemes_by_topic: Dict[str, List[str]] = generate_subthemes(related_topics, max_subthemes=3)
        emit_to_user('receive_message', {'sender': 'bot', 'message': f"✅ Generated subthemes for topics."})

        raw_keywords_nested: Dict[str, Dict[str, List[str]]] = generate_keywords_by_subtheme(subthemes_by_topic, max_terms=5)
        emit_to_user('receive_message', {'sender': 'bot', 'message': f"✅ Generated raw keywords for subthemes."})
        
        emit_to_user('receive_message', {'sender': 'bot', 'message': "🤖 Refining keywords with Critic AI..."})
        critiqued_keywords_nested: Dict[str, Dict[str, List[str]]] = {}
        for topic_key, subthemes_map in raw_keywords_nested.items():
            refined_subthemes_for_topic: Dict[str, List[str]] = {}
            for subtheme_key, original_keywords in subthemes_map.items():
                critic_label = f"Keywords for topic '{topic_key}' / subtheme '{subtheme_key}'"
                refined_list, _ = critique_list(critic_label, original_keywords) # Assuming suggestions_map is not used here
                refined_subthemes_for_topic[subtheme_key] = refined_list
            critiqued_keywords_nested[topic_key] = refined_subthemes_for_topic
        emit_to_user('receive_message', {'sender': 'bot', 'message': "✅ Keywords refined."})

        # === Phase 2: Paper Collection ===
        emit_to_user('receive_message', {'sender': 'bot', 'message': "📚 **Phase 2/7: Collecting Academic Papers...** (This is the longest step)"})
        papers_to_fetch_per_keyword_source: int = 3
        collected_papers: List[Dict] = collect_papers(
            keywords_by_topic=critiqued_keywords_nested,
            cutoff_year=cutoff_year,
            paper_per_keyword=papers_to_fetch_per_keyword_source
            # min_papers_per_bucket and max_fetch_attempts will use defaults from collect_papers definition
        )
        emit_to_user('receive_message', {'sender': 'bot', 'message': f"✅ Initial collection found {len(collected_papers)} paper candidates."})

        # === Phase 3: Initial Filtering & Domain Term Generation ===
        emit_to_user('receive_message', {'sender': 'bot', 'message': "🧹 **Phase 3/7: Initial Filtering & Domain Term Generation...**"})
        
        # Instead of saving to/loading from JSON, process in memory for the web app
        papers_after_doi_filter = filter_by_doi(collected_papers)
        papers_after_abstract_filter = filter_by_abstract(papers_after_doi_filter)
        papers_after_initial_filters = dedupe_by_doi(papers_after_abstract_filter)
        emit_to_user('receive_message', {'sender': 'bot', 'message': f"✅ Initial filtering complete. {len(papers_after_initial_filters)} papers remain."})

        raw_domain_terms: List[str] = generate_domain_terms(title, max_terms=15)
        critiqued_domain_terms, domain_term_suggestions = critique_list(f"Domain terms for {title}", raw_domain_terms)
        emit_to_user('receive_message', {'sender': 'bot', 'message': f"✅ Domain terms generated and refined."})
        # Optional: emit domain_term_suggestions if useful for UI
        # emit_to_user('receive_message', {'sender': 'bot', 'message': f"ℹ️ Domain term suggestions: {domain_term_suggestions}"})

        # === Phase 4: Application/Technique Terms, Cleaning & Clustering ===
        emit_to_user('receive_message', {'sender': 'bot', 'message': "🛠️ **Phase 4/7: App/Tech Terms, Cleaning & Clustering...**"})
        raw_app_terms: List[str] = generate_app_terms(title, max_terms=7)
        raw_tech_terms: List[str] = generate_tech_terms(title, max_terms=10)
        
        critiqued_app_terms, app_term_suggestions = critique_list(f"Application terms for {title}", raw_app_terms)
        critiqued_tech_terms, tech_term_suggestions = critique_list(f"Technique terms for {title}", raw_tech_terms)

        # Finalize terms using the clean_terms helper function (ensure it's defined in app.py or imported)
        final_app_terms: List[str] = clean_terms(critiqued_app_terms)
        final_app_terms = [t for t in final_app_terms if len(t.split()) > 1]
        title_lower_stripped = title.lower().strip()
        if title_lower_stripped not in final_app_terms:
            final_app_terms.insert(0, title_lower_stripped)
        if not final_app_terms:
            emit_to_user('receive_message', {'sender': 'bot', 'message': "⚠️ App terms empty after cleaning; using raw as fallback."})
            final_app_terms = raw_app_terms[:5]

        final_tech_terms: List[str] = clean_terms(critiqued_tech_terms)
        if not final_tech_terms and critiqued_tech_terms:
            emit_to_user('receive_message', {'sender': 'bot', 'message': "⚠️ Tech terms empty after cleaning; using critiqued as fallback."})
            final_tech_terms = critiqued_tech_terms[:3]
        elif not final_tech_terms:
            emit_to_user('receive_message', {'sender': 'bot', 'message': "⚠️ Tech terms empty; using raw as fallback."})
            final_tech_terms = raw_tech_terms[:5]
        
        final_domain_terms: List[str] = clean_terms(critiqued_domain_terms) # Use the critiqued_domain_terms from Phase 3
        if not final_domain_terms and critiqued_domain_terms:
            emit_to_user('receive_message', {'sender': 'bot', 'message': "⚠️ Domain terms empty after cleaning; using critiqued as fallback."})
            final_domain_terms = critiqued_domain_terms[:10]
        elif not final_domain_terms:
            emit_to_user('receive_message', {'sender': 'bot', 'message': "⚠️ Domain terms empty; using raw as fallback."})
            final_domain_terms = raw_domain_terms[:10] # Fallback to Phase 3 raw if needed

        # Prepare term lists for pattern matching
        app_terms_for_patterns: List[str] = final_app_terms + list(app_term_suggestions.values())
        tech_terms_for_patterns: List[str] = final_tech_terms + list(tech_term_suggestions.values()) # Base for patterns
        domain_terms_for_patterns: List[str] = final_domain_terms + list(domain_term_suggestions.values())
        
        emit_to_user('receive_message', {'sender': 'bot', 'message': f"🔑 Final Core Application Terms: {final_app_terms}"})
        emit_to_user('receive_message', {'sender': 'bot', 'message': f"🔑 Final Core Technique Terms: {final_tech_terms}"})
        emit_to_user('receive_message', {'sender': 'bot', 'message': f"🔑 Final Core Domain Terms: {final_domain_terms}"})

        # --- Technique Term Clustering ---
        emit_to_user('receive_message', {'sender': 'bot', 'message': "🧬 Processing technique terms for potential clustering..."})
        representative_tech_terms: List[str] = final_tech_terms # Default
        
        if final_tech_terms and len(final_tech_terms) > 4:
            # Filter out empty strings from final_tech_terms before embedding
            valid_final_tech_terms = [term for term in final_tech_terms if term.strip()]
            if len(valid_final_tech_terms) > 4 : # Check again after filtering empty strings
                tech_term_vectors: List[np.ndarray] = [embed_text(term, use_cache=True) for term in valid_final_tech_terms]
                # Filter out None vectors in case embed_text can return None or fails for some terms
                valid_tech_term_vectors = [vec for vec in tech_term_vectors if vec is not None]
                # Align valid_final_tech_terms with valid_tech_term_vectors
                terms_actually_embedded = [term for term, vec in zip(valid_final_tech_terms, tech_term_vectors) if vec is not None]

                if valid_tech_term_vectors and len(valid_tech_term_vectors) > 1: # Need at least 2 vectors for clustering
                    emit_to_user('receive_message', {'sender': 'bot', 'message': "🔬 Clustering technique terms to find representative set..."})
                    try:
                        tech_vectors_stacked: np.ndarray = np.vstack(valid_tech_term_vectors)
                        max_clusters: int = 5
                        num_clusters: int = min(max_clusters, max(2, len(terms_actually_embedded) // 2))
                        if num_clusters < 2 and len(terms_actually_embedded) >=2 : # Ensure num_clusters is at least 2 if possible
                            num_clusters = 2 
                        elif len(terms_actually_embedded) < 2: # Cannot cluster less than 2 samples
                             raise ValueError("Not enough samples to form clusters after embedding.")


                        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto')
                        cluster_labels: np.ndarray = kmeans.fit_predict(tech_vectors_stacked)

                        clustered_tech_terms_map: Dict[int, List[str]] = {}
                        for term, label in zip(terms_actually_embedded, cluster_labels):
                            clustered_tech_terms_map.setdefault(int(label), []).append(term)

                        cluster_messages = ["🔑 **Technique Clusters Formed:**"]
                        for label_id, terms_in_cluster in clustered_tech_terms_map.items():
                            cluster_messages.append(f"    Cluster {label_id}: {', '.join(terms_in_cluster)}")
                        emit_to_user('receive_message', {'sender': 'bot', 'message': "\n".join(cluster_messages)})

                        selected_representative_terms_list: List[str] = []
                        for label_id, terms_in_cluster in clustered_tech_terms_map.items():
                            num_to_keep_from_cluster: int = max(1, math.ceil(math.sqrt(len(terms_in_cluster))))
                            if len(terms_in_cluster) <= num_to_keep_from_cluster:
                                selected_representative_terms_list.extend(terms_in_cluster)
                            else:
                                term_indices_in_cluster: List[int] = []
                                for t_cluster in terms_in_cluster:
                                    try:
                                        term_indices_in_cluster.append(terms_actually_embedded.index(t_cluster))
                                    except ValueError:
                                        # This term was in terms_in_cluster but somehow not in terms_actually_embedded
                                        print(f"Warning: Term '{t_cluster}' from cluster not found in 'terms_actually_embedded' during distance calculation.")
                                        continue
                                
                                centroid_for_cluster: np.ndarray = kmeans.cluster_centers_[label_id]
                                distances_to_centroid: List[Tuple[str, float]] = []
                                for term, original_idx in zip(terms_in_cluster, term_indices_in_cluster):
                                    if original_idx < len(valid_tech_term_vectors):
                                        distances_to_centroid.append(
                                            (term, float(np.linalg.norm(valid_tech_term_vectors[original_idx] - centroid_for_cluster)))
                                        )
                                distances_to_centroid.sort(key=lambda x: x[1])
                                selected_representative_terms_list.extend([t_dist[0] for t_dist in distances_to_centroid[:num_to_keep_from_cluster]])
                        
                        if selected_representative_terms_list:
                            representative_tech_terms = sorted(list(set(selected_representative_terms_list)))
                        
                        emit_to_user('receive_message', {'sender': 'bot', 'message': f"🔑 Representative Technique Terms after clustering: {representative_tech_terms}"})
                    
                    except Exception as e_cluster:
                        emit_to_user('receive_message', {'sender': 'bot', 'message': f"⚠️ Error during technique term clustering: {e_cluster}. Using unclustered terms."})
                        representative_tech_terms = final_tech_terms # Fallback
                else:
                    emit_to_user('receive_message', {'sender': 'bot', 'message': "ℹ️ Technique clustering skipped (not enough valid vectors generated)."})
            else: # Not enough terms after filtering empty strings
                 emit_to_user('receive_message', {'sender': 'bot', 'message': "ℹ️ Technique clustering skipped (not enough non-empty terms)."})
        else:
            emit_to_user('receive_message', {'sender': 'bot', 'message': "ℹ️ Technique clustering skipped (reason: ≤ 4 terms initially)."})
        
        final_tech_terms_for_patterns = representative_tech_terms if representative_tech_terms else tech_terms_for_patterns

        # === Phase 5: Semantic Ranking & Boosting ===
        emit_to_user('receive_message', {'sender': 'bot', 'message': "⚖️ **Phase 5/7: Semantic Ranking & Score Boosting...**"})
        if not papers_after_initial_filters:
            emit_to_user('receive_message', {'sender': 'bot', 'message': "⚠️ No papers available for semantic ranking."})
            semantically_ranked_papers = []
        else:
            semantically_ranked_papers: List[Dict] = semantic_rank_papers(
                query=title, 
                papers=papers_after_initial_filters, 
                top_n=None # Rank all papers initially
            )
            emit_to_user('receive_message', {'sender': 'bot', 'message': f"✅ Semantic ranking complete for {len(semantically_ranked_papers)} papers."})

            app_regex_patterns = [re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE) for term in app_terms_for_patterns if term]
            tech_regex_patterns = [re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE) for term in final_tech_terms_for_patterns if term]
            
            for paper in semantically_ranked_papers:
                text_to_search = f"{paper.get('title','')} {paper.get('abstract','')}".lower()
                matches_an_app_term = any(pattern.search(text_to_search) for pattern in app_regex_patterns)
                matches_a_tech_term = any(pattern.search(text_to_search) for pattern in tech_regex_patterns)
                current_score = paper.get("score", 0.0)
                if matches_an_app_term and matches_a_tech_term:
                    paper["score"] = current_score * 1.25
                elif matches_an_app_term or matches_a_tech_term:
                    paper["score"] = current_score * 1.10
            
            semantically_ranked_papers.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            emit_to_user('receive_message', {'sender': 'bot', 'message': "✅ Paper scores boosted and re-sorted."})
        
        # === Phase 6: Categorization & Final Results ===
        emit_to_user('receive_message', {'sender': 'bot', 'message': "📋 **Phase 6/7: Categorizing & Preparing Final Results...**"})
        
        domain_regex_patterns = [re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE) for term in domain_terms_for_patterns if term]

        def check_matches_app(p_dict: Dict) -> bool:
            txt = (p_dict.get("title", "") + " " + p_dict.get("abstract", "")).lower()
            return any(pat.search(txt) for pat in app_regex_patterns) if app_regex_patterns else False
        def check_matches_tech(p_dict: Dict) -> bool:
            txt = (p_dict.get("title", "") + " " + p_dict.get("abstract", "")).lower()
            return any(pat.search(txt) for pat in tech_regex_patterns) if tech_regex_patterns else False
        def count_domain_hits(p_dict: Dict) -> int:
            txt = (p_dict.get("title", "") + " " + p_dict.get("abstract", "")).lower()
            return sum(bool(dp.search(txt)) for dp in domain_regex_patterns) if domain_regex_patterns else 0

        DESIRED_FOCUSED_COUNT: int = 20
        DESIRED_EXPLORATORY_COUNT: int = 10
        focused_papers: List[Dict] = []
        
        for p in semantically_ranked_papers:
            if len(focused_papers) >= DESIRED_FOCUSED_COUNT: break
            if check_matches_app(p) and check_matches_tech(p) and count_domain_hits(p) >= 1:
                focused_papers.append(p)
        if len(focused_papers) < DESIRED_FOCUSED_COUNT:
            for p in semantically_ranked_papers:
                if len(focused_papers) >= DESIRED_FOCUSED_COUNT: break
                if p in focused_papers: continue
                if check_matches_app(p) and check_matches_tech(p):
                    focused_papers.append(p)
        if len(focused_papers) < DESIRED_FOCUSED_COUNT:
            for p in semantically_ranked_papers:
                if len(focused_papers) >= DESIRED_FOCUSED_COUNT: break
                if p in focused_papers: continue
                if count_domain_hits(p) >= 1 and (check_matches_app(p) or check_matches_tech(p)):
                    focused_papers.append(p)
        final_focused_papers = focused_papers[:DESIRED_FOCUSED_COUNT]
        focused_paper_dois = {p_item["doi"] for p_item in final_focused_papers if p_item.get("doi")}

        exploratory_candidates = [p for p in semantically_ranked_papers if p.get("doi") not in focused_paper_dois]
        exploratory_papers: List[Dict] = []
        for p in exploratory_candidates:
            if len(exploratory_papers) >= DESIRED_EXPLORATORY_COUNT: break
            if count_domain_hits(p) >= 2 or (count_domain_hits(p) >= 1 and check_matches_tech(p)):
                exploratory_papers.append(p)
        if len(exploratory_papers) < DESIRED_EXPLORATORY_COUNT:
            for p in exploratory_candidates:
                if len(exploratory_papers) >= DESIRED_EXPLORATORY_COUNT: break
                if p in exploratory_papers: continue
                if count_domain_hits(p) >= 1:
                    exploratory_papers.append(p)
        final_exploratory_papers = exploratory_papers[:DESIRED_EXPLORATORY_COUNT]
        
        # --- Results Presentation ---
        elapsed_time = time.time() - pipeline_start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        emit_to_user('receive_message', {'sender': 'bot', 'message': f"🎉 **Search completed in {minutes}m {seconds}s!**"})

        if final_focused_papers:
            emit_to_user('receive_message', {'sender': 'bot', 'message': f"🎯 **TOP {len(final_focused_papers)} FOCUSED RESULTS**:"})
            for i, paper in enumerate(final_focused_papers, 1):
                result_msg = format_paper_with_citation(paper, i)
                emit_to_user('receive_message', {'sender': 'bot', 'message': result_msg})
                socketio.sleep(0.1) 
        else:
            emit_to_user('receive_message', {'sender': 'bot', 'message': "ℹ️ No papers met the criteria for 'Focused Results'."})

        if final_exploratory_papers:
            emit_to_user('receive_message', {'sender': 'bot', 'message': f"🔍 **TOP {len(final_exploratory_papers)} EXPLORATORY RESULTS**:"})
            for i, paper in enumerate(final_exploratory_papers, 1):
                result_msg = format_paper_with_citation(paper, i)
                emit_to_user('receive_message', {'sender': 'bot', 'message': result_msg})
                socketio.sleep(0.1)
        
        all_results_for_bib = final_focused_papers + final_exploratory_papers
        if all_results_for_bib:
            emit_to_user('receive_message', {'sender': 'bot', 'message': "📚 **COMPLETE BIBLIOGRAPHY** (APA Style):"})
            bibliography = generate_bibliography(all_results_for_bib)
            emit_to_user('receive_message', {'sender': 'bot', 'message': bibliography})
        
        emit_to_user('set_context', {'processing': False, 'search_complete': True, 'title': title})
        emit_to_user('receive_message', {'sender': 'bot', 'message': "✅ **All done!** What would you like to do next? (e.g., 'cite MLA', 'new search: [title]', 'reset')"})
    
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"ERROR: Unhandled exception in app.py run_pipeline: {error_details}")
        emit_to_user('receive_message', {
            'sender': 'bot',
            'message': f"❌ **An unexpected error occurred in the pipeline:**\n\n{str(e)}\n\nPlease try 'reset' or a different research title."
        })
        emit_to_user('set_context', {'processing': False, 'search_complete': False, 'title': title}) # Pass current title to context

        # === Phase 7: Generate AI Insights for Top Papers ===
        emit_to_user('receive_message', {'sender': 'bot', 'message': "💡 **Phase 7/7: Generating AI-powered insights for top papers...** (This may take a moment)"})
        
        # Process focused papers
        if final_focused_papers:
            emit_to_user('receive_message', {'sender': 'bot', 'message': f"🧠 Analyzing {len(final_focused_papers)} focused papers..."})
            for i, paper_dict in enumerate(final_focused_papers):
                paper_abstract = paper_dict.get('abstract')
                if paper_abstract and isinstance(paper_abstract, str) and paper_abstract.strip():
                    # Display progress to the user for each paper or in batches
                    if (i + 1) % 5 == 0 or i == 0 : # Update every 5 papers or for the first one
                        emit_to_user('receive_message', {'sender': 'bot', 'message': f"  ↪ Generating insights for focused paper {i+1}/{len(final_focused_papers)}..."})
                    
                    insights_data = generate_insights(title, paper_abstract) # 'title' is the main research title
                    paper_dict['insights_summary'] = insights_data.get('summary', "Summary not available.")
                    paper_dict['insights_relevance'] = insights_data.get('relevance', "Relevance assessment not available.")
                    socketio.sleep(2.1) # Sleep to respect Gemini API rate limits (get_next_api_client helps, but good to be cautious)
                else:
                    paper_dict['insights_summary'] = "Insight generation skipped: Abstract was missing or empty."
                    paper_dict['insights_relevance'] = "Relevance assessment skipped: Abstract was missing or empty."
        
        # Process exploratory papers
        if final_exploratory_papers:
            emit_to_user('receive_message', {'sender': 'bot', 'message': f"🧠 Analyzing {len(final_exploratory_papers)} exploratory papers..."})
            for i, paper_dict in enumerate(final_exploratory_papers):
                paper_abstract = paper_dict.get('abstract')
                if paper_abstract and isinstance(paper_abstract, str) and paper_abstract.strip():
                    if (i + 1) % 5 == 0 or i == 0:
                        emit_to_user('receive_message', {'sender': 'bot', 'message': f"  ↪ Generating insights for exploratory paper {i+1}/{len(final_exploratory_papers)}..."})

                    insights_data = generate_insights(title, paper_abstract)
                    paper_dict['insights_summary'] = insights_data.get('summary', "Summary not available.")
                    paper_dict['insights_relevance'] = insights_data.get('relevance', "Relevance assessment not available.")
                    socketio.sleep(2.1) # Sleep to respect Gemini API rate limits
                else:
                    paper_dict['insights_summary'] = "Insight generation skipped: Abstract was missing or empty."
                    paper_dict['insights_relevance'] = "Relevance assessment skipped: Abstract was missing or empty."
        
        emit_to_user('receive_message', {'sender': 'bot', 'message': "✅ AI insights generated for top papers."})

        # --- Results Presentation (Phase now becomes 8 or continues from existing numbering) ---
        # The 'final_focused_papers' and 'final_exploratory_papers' lists now contain the insights.
        # Your existing result presentation loop will pick these up if you modify the formatting.
        # For example, in format_paper_with_citation, you can add these new fields.

        elapsed_time = time.time() - pipeline_start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        emit_to_user('receive_message', {'sender': 'bot', 'message': f"🎉 **Search and Analysis completed in {minutes}m {seconds}s!**"})

        # ... (The rest of your results presentation logic for focused_papers, exploratory_papers, bibliography) ...
        # You will need to modify format_paper_with_citation to display these insights.
        
        if final_focused_papers:
            emit_to_user('receive_message', {'sender': 'bot', 'message': f"🎯 **TOP {len(final_focused_papers)} FOCUSED RESULTS (with Insights):**"})
            for i, paper in enumerate(final_focused_papers, 1):
                # Modify format_paper_with_citation to include insights_summary and insights_relevance
                result_msg = format_paper_with_citation_and_insights(paper, i) # New/Modified formatting function
                emit_to_user('receive_message', {'sender': 'bot', 'message': result_msg})
                socketio.sleep(0.1) 
        else:
            emit_to_user('receive_message', {'sender': 'bot', 'message': "ℹ️ No papers met the criteria for 'Focused Results'."})

        if final_exploratory_papers:
            emit_to_user('receive_message', {'sender': 'bot', 'message': f"🔍 **TOP {len(final_exploratory_papers)} EXPLORATORY RESULTS (with Insights):**"})
            for i, paper in enumerate(final_exploratory_papers, 1):
                # Modify format_paper_with_citation to include insights_summary and insights_relevance
                result_msg = format_paper_with_citation_and_insights(paper, i) # New/Modified formatting function
                emit_to_user('receive_message', {'sender': 'bot', 'message': result_msg})
                socketio.sleep(0.1)
        
        all_results_for_bib = final_focused_papers + final_exploratory_papers
        if all_results_for_bib:
            emit_to_user('receive_message', {'sender': 'bot', 'message': "📚 **COMPLETE BIBLIOGRAPHY** (APA Style):"})
            bibliography = generate_bibliography(all_results_for_bib)
            emit_to_user('receive_message', {'sender': 'bot', 'message': bibliography})
        
        emit_to_user('set_context', {'processing': False, 'search_complete': True, 'title': title})
        emit_to_user('receive_message', {'sender': 'bot', 'message': "✅ **All done!** Insights included. What would you like to do next? (e.g., 'cite MLA', 'new search: [title]', 'reset')"})
    
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"ERROR: Unhandled exception in app.py run_pipeline: {error_details}")
        emit_to_user('receive_message', {
            'sender': 'bot',
            'message': f"❌ **An unexpected error occurred in the pipeline:**\n\n{str(e)}\n\nPlease try 'reset' or a different research title."
        })
        # Ensure context is reset correctly, passing the title that was being processed
        emit_to_user('set_context', {'processing': False, 'search_complete': False, 'title': title})

# Add a simple health check route for testing
@app.route('/health')
def health_check():
    return jsonify({
        "status": "ok",
        "gemini_api_key_set": bool(os.environ.get("GEMINI_API_KEY"))
    })

# Add a simple test route to verify Gemini API is working
@app.route('/test-gemini')
def test_gemini():
    try:
        from google import genai
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        # Simple test prompt
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="What is 2+2? Give just the number."
        )
        return jsonify({
            "status": "ok",
            "gemini_response": response.text.strip()
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

# Add route to manually test topic generation
@app.route('/test-topics')
def test_topics():
    title = request.args.get('title', 'Machine learning in healthcare')
    try:
        topics = generate_topics(title)
        return jsonify({
            "status": "ok",
            "title": title,
            "topics": topics
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        })

if __name__ == '__main__':
    # Create templates and static directories if they don't exist
    templates_dir = Path("templates")
    static_dir = Path("static/css")
    static_js_dir = Path("static/js")
    
    if not templates_dir.exists():
        templates_dir.mkdir(parents=True)
        print(f"Created {templates_dir} directory")
        
    if not static_dir.exists():
        static_dir.mkdir(parents=True)
        print(f"Created {static_dir} directory")

    if not static_js_dir.exists():
        static_js_dir.mkdir(parents=True)
        print(f"Created {static_js_dir} directory")
        
    # Check if the template file exists
    template_file = templates_dir / "index.html"
    if not template_file.exists():
        print(f"Warning: Template file {template_file} not found!")
        print("Creating a basic template file...")
        with open(template_file, 'w') as f:
            f.write(r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Authematic - Literature Curation</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container">
        <header class="py-3 mb-4 border-bottom">
            <div class="d-flex align-items-center">
                <span class="fs-4">Authematic</span>
                <span class="ms-2 text-muted">Literature Curation Assistant</span>
            </div>
        </header>
        
        <div class="row">
            <div class="col-md-12">
                <div class="card mb-4">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">Authematic Chat Interface</h5>
                        <button id="reset-button" class="btn btn-outline-secondary btn-sm">Reset</button>
                    </div>
                    <div class="card-body">
                        <div id="chat-messages" class="chat-container mb-3">
                            <div class="message bot">
                                <div class="message-content">
                                    Welcome to Authematic! I can help you find relevant academic papers with proper APA citations. 
                                    Please provide your research title by saying "Research title: [your title]"
                                </div>
                            </div>
                        </div>
                        <div class="input-group">
                            <input type="text" id="user-input" class="form-control" placeholder="Type your message...">
                            <button id="send-button" class="btn btn-primary">Send</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <footer class="py-3 mt-4 text-center text-muted">
            <p>© 2025 Authematic</p>
        </footer>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.socket.io/4.6.0/socket.io.min.js"></script>
    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
    <script>
        // Initialize chat context with localStorage for persistence
        let chatContext = JSON.parse(localStorage.getItem('chatContext') || '{}');
        
        document.addEventListener('DOMContentLoaded', function() {
            const chatMessages = document.getElementById('chat-messages');
            const userInput = document.getElementById('user-input');
            const sendButton = document.getElementById('send-button');
            const resetButton = document.getElementById('reset-button');
            
            // Initialize Socket.IO
            const socket = io();
            
            // Handle connection
            socket.on('connect', function() {
                console.log('Connected to server');
            });
            
            // Handle disconnection
            socket.on('disconnect', function() {
                console.log('Disconnected from server');
                appendMessage('bot', 'Connection lost. Please refresh the page.');
            });
            
            // Handle received messages
            socket.on('receive_message', function(data) {
                appendMessage(data.sender, data.message);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            });
            
            // Handle context updates
            socket.on('set_context', function(data) {
                chatContext = {...chatContext, ...data};
                console.log('Context updated:', chatContext);
                localStorage.setItem('chatContext', JSON.stringify(chatContext));
            });
            
            // Send message on button click
            sendButton.addEventListener('click', function() {
                sendMessage();
            });
            
            // Reset button functionality
            resetButton.addEventListener('click', function() {
                clearChatContext();
            });
            
            // Send message on Enter key
            userInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
            
            // Function to send message
            function sendMessage() {
                const message = userInput.value.trim();
                if (message) {
                    // Display user message
                    appendMessage('user', message);
                    
                    // Send to server with context
                    socket.emit('send_message', {
                        message: message,
                        context: chatContext
                    });
                    
                    // Clear input
                    userInput.value = '';
                    
                    // Show typing indicator
                    showTypingIndicator();
                }
            }
            
            // Function to append message to chat
            function appendMessage(sender, message) {
                // Remove typing indicator if exists
                const typingIndicator = document.querySelector('.typing-indicator');
                if (typingIndicator) {
                    typingIndicator.remove();
                }
                
                // Create message element
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}`;
                
                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                
                // Support for simple markdown-like formatting
                message = message
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                    .replace(/\n/g, '<br>');
                    
                contentDiv.innerHTML = message;
                messageDiv.appendChild(contentDiv);
                
                // Add to chat
                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
            // Function to show typing indicator
            function showTypingIndicator() {
                const indicator = document.createElement('div');
                indicator.className = 'typing-indicator';
                indicator.innerHTML = '<span></span><span></span><span></span>';
                chatMessages.appendChild(indicator);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
            // Function to clear chat context
            function clearChatContext() {
                chatContext = {};
                localStorage.removeItem('chatContext');
                
                // Clear chat messages except welcome
                while (chatMessages.childNodes.length > 1) {
                    chatMessages.removeChild(chatMessages.lastChild);
                }
                
                // Add reset message
                appendMessage('bot', 'Conversation has been reset. You can start a new search by providing a research title.');
            }
        });
    </script>
</body>
</html>
            """)
    
    # Check if the CSS file exists
    css_file = static_dir / "style.css"
    if not css_file.exists():
        print(f"Warning: CSS file {css_file} not found!")
        print("Creating a basic CSS file...")
        with open(css_file, 'w') as f:
            f.write("""
.chat-container {
    height: 500px;
    overflow-y: auto;
    padding: 15px;
    background-color: #f8f9fa;
    border-radius: 5px;
}

.message {
    margin-bottom: 15px;
    max-width: 80%;
    clear: both;
}

.message.user {
    float: right;
}

.message.bot {
    float: left;
}

.message-content {
    padding: 10px 15px;
    border-radius: 15px;
    display: inline-block;
    word-wrap: break-word;
}

.user .message-content {
    background-color: #007bff;
    color: white;
}

.bot .message-content {
    background-color: #e9ecef;
    color: #212529;
}

/* For Markdown-like formatting */
.bot .message-content strong,
.bot .message-content b {
    font-weight: bold;
}

.bot .message-content em,
.bot .message-content i {
    font-style: italic;
}

.typing-indicator {
    display: inline-block;
    padding: 10px 15px;
    background-color: #e9ecef;
    border-radius: 15px;
    margin-bottom: 15px;
}

.typing-indicator span {
    height: 10px;
    width: 10px;
    background-color: #666;
    border-radius: 50%;
    display: inline-block;
    margin: 0 2px;
    animation: bounce 1.5s infinite;
}

.typing-indicator span:nth-child(2) {
    animation-delay: 0.2s;
}

.typing-indicator span:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); }
    30% { transform: translateY(-5px); }
}

/* Progress animations */
.progress-message {
    animation: fadeIn 0.5s ease-in;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Citation formatting */
.citation-text {
    font-family: 'Courier New', monospace;
    font-size: 0.9em;
    background-color: #f8f9fa;
    padding: 5px;
    border-left: 3px solid #007bff;
    margin: 5px 0;
}

/* Bibliography section */
.bibliography-section {
    background-color: #fff;
    border: 1px solid #dee2e6;
    border-radius: 5px;
    padding: 15px;
    margin: 10px 0;
}
            """)
    
    # Run the app
    print("Starting Authematic web application...")
    socketio.run(app, debug=True)
