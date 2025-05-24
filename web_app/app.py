import os
import sys
import json
import traceback
import time
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO

# Print debug info
print("Starting app.py")
print(f"Current directory: {os.getcwd()}")

# Set API key directly before any imports that need it
os.environ["GEMINI_API_KEY"] = "AIzaSyCN6hmeeB5sxliWoYL7OVxHxTSvgbnGCko"
print(f"GEMINI_API_KEY after setting: {os.environ.get('GEMINI_API_KEY')}")

# Add parent directory to path to import from parent modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your existing modules
from paper_collector import generate_topics, generate_keywords, collect_papers, generate_domain_terms
from filter_and_rank import filter_by_doi, filter_by_abstract, dedupe_by_doi, semantic_rank_papers

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'authematic-secret-key'
socketio = SocketIO(app)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

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
    user_message = data.get('message', '')
    context = data.get('context', {})
    print(f"Received message: '{user_message}' with context: {context}")
    
    # Clear context on reset
    if user_message.lower() == 'reset' or user_message.lower() == 'restart':
        socketio.emit('set_context', {'title': None, 'processing': False, 'search_complete': False})
        socketio.emit('receive_message', {
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
        socketio.emit('receive_message', {
            'sender': 'bot', 
            'message': f"I'll help you find relevant papers for: '{title}'. What year would you like to use as a cutoff?"
        })
        socketio.emit('set_context', {'title': title, 'processing': False, 'search_complete': False})
        return
    
    # If we have a title and this looks like a year input
    if context.get('title') and (user_message.isdigit() or 
                             "to" in user_message or 
                             "year:" in user_message.lower()):
        # Extract year or year range
        year_text = user_message.lower().replace("year:", "").strip()
        
        # Handle year range (e.g., "2020 to 2025")
        if "to" in year_text:
            parts = year_text.split("to")
            if len(parts) == 2 and parts[0].strip().isdigit():
                year = parts[0].strip()  # Use the start year
            else:
                year = "2020"  # Default
        elif year_text.isdigit():
            year = year_text
        else:
            year = "2020"  # Default
            
        title = context.get('title')
        
        # Generate progress message
        socketio.emit('receive_message', {
            'sender': 'bot',
            'message': f"🚀 Starting paper search for '{title}' (≥{year}). This will take 2-3 minutes. I'll keep you updated on my progress!"
        })
        
        # Set processing flag to prevent duplicate requests
        socketio.emit('set_context', {'processing': True, 'search_complete': False})
        
        # Run the pipeline in the background
        socketio.start_background_task(run_pipeline, title, int(year))
        return
    
    # Handle general questions after search is complete
    if context.get('search_complete'):
        # Handle citation-related questions
        if 'citation' in user_message.lower() or 'cite' in user_message.lower():
            socketio.emit('receive_message', {
                'sender': 'bot',
                'message': "📚 **Citation Help:**\n\nI've already provided APA citations for all the papers I found. Here are some tips:\n\n• **APA Format**: Author, A. A. (Year). Title of paper. DOI or URL\n• **Copy Citations**: You can copy the APA citations I provided directly into your reference list\n• **Bibliography**: I've also generated a complete bibliography at the end of your results\n\n**Need other formats?** I can help explain MLA, Chicago, or IEEE citation styles if needed!"
            })
        elif 'mla' in user_message.lower():
            socketio.emit('receive_message', {
                'sender': 'bot',
                'message': "📖 **MLA Citation Format:**\n\nMLA format looks like this:\n**Author Last, First. \"Title of Paper.\" *Journal Name*, vol. #, no. #, Year, pp. ##-##. DOI or URL.**\n\nExample:\n*Smith, John. \"Machine Learning Applications.\" AI Research Journal, vol. 15, no. 3, 2023, pp. 45-62. https://doi.org/10.1000/example.*\n\nWould you like me to help you convert any specific citations to MLA format?"
            })
        elif 'chicago' in user_message.lower():
            socketio.emit('receive_message', {
                'sender': 'bot',
                'message': "📑 **Chicago Citation Format:**\n\n**Notes-Bibliography Style:**\nAuthor Last, First. \"Title of Paper.\" Journal Name vol. #, no. # (Year): ##-##. DOI or URL.\n\n**Author-Date Style:**\nAuthor Last, First. Year. \"Title of Paper.\" Journal Name vol. # (no. #): ##-##. DOI or URL.\n\nExample (Author-Date):\n*Smith, John. 2023. \"Machine Learning Applications.\" AI Research Journal 15 (3): 45-62. https://doi.org/10.1000/example.*"
            })
        # Handle general chat questions
        elif user_message.lower() in ['hi', 'hello', 'hey']:
            socketio.emit('receive_message', {
                'sender': 'bot',
                'message': "Hello! I'm Authematic, your literature curation assistant. I can help you find relevant academic papers for your research. How can I help you today?"
            })
        elif 'who are you' in user_message.lower():
            socketio.emit('receive_message', {
                'sender': 'bot',
                'message': "I'm Authematic, an AI-powered literature curation assistant designed to help researchers find relevant academic papers. I use natural language processing and semantic ranking to identify the most relevant papers for your research topics, and I provide properly formatted APA citations for all results!"
            })
        elif 'how do you work' in user_message.lower() or 'what can you do' in user_message.lower():
            socketio.emit('receive_message', {
                'sender': 'bot',
                'message': "I help you find academic papers by:\n\n1. Analyzing your research title\n2. Generating relevant academic topics and keywords\n3. Searching multiple sources (arXiv, Semantic Scholar, etc.)\n4. Filtering and ranking papers using SciBERT embeddings\n5. Presenting you with focused and exploratory results\n6. **Providing APA citations** for all papers found\n7. **Generating a complete bibliography** ready for your references\n\nTo start, just type 'Research title: [your research topic]'"
            })
        else:
            # For any other message, prompt for a research title
            socketio.emit('receive_message', {
                'sender': 'bot',
                'message': "I'm designed to help with academic literature curation and citations. If you'd like to search for papers, please provide a research title by saying 'Research title: [your title]'"
            })
        return
    
    # Handle other queries when we don't have a complete context
    # If we're processing, just acknowledge
    if context.get('processing'):
        socketio.emit('receive_message', {
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
        socketio.emit('receive_message', {
            'sender': 'bot',
            'message': f"🚀 Starting paper search for '{title}' (≥{year}). This will take 2-3 minutes. I'll keep you updated!"
        })
        socketio.emit('set_context', {'processing': True, 'search_complete': False})
        socketio.start_background_task(run_pipeline, title, int(year))
    # Otherwise prompt for research title
    else:
        socketio.emit('receive_message', {
            'sender': 'bot',
            'message': "I can help you find relevant academic papers with properly formatted citations. Please provide a research title by saying 'Research title: [your title]'"
        })

def run_pipeline(title, cutoff_year):
    try:
        start_time = time.time()
        
        # Step 1: Topic and Keyword Generation
        socketio.emit('receive_message', {'sender': 'bot', 'message': "📊 **Step 1/5:** Analyzing your research title and generating relevant topics..."})
        
        # Generate topics
        related_topics = generate_topics(title)
        socketio.emit('receive_message', {'sender': 'bot', 'message': f"✅ Generated **{len(related_topics)} research topics:**\n• {chr(10).join(['• ' + topic for topic in related_topics])}"})
        
        # Generate keywords
        socketio.emit('receive_message', {'sender': 'bot', 'message': "🔍 Creating search keywords for each topic..."})
        keywords_by_topic = generate_keywords(related_topics)
        
        total_keywords = sum(len(keywords) for keywords in keywords_by_topic.values())
        socketio.emit('receive_message', {'sender': 'bot', 'message': f"✅ Generated **{total_keywords} search keywords** across all topics"})
        
        # Generate domain terms
        domain_terms = generate_domain_terms(title, max_terms=10)
        socketio.emit('receive_message', {'sender': 'bot', 'message': f"✅ Identified **{len(domain_terms)} domain-specific terms** for filtering"})
        
        # Step 2: Paper Collection
        socketio.emit('receive_message', {'sender': 'bot', 'message': "📚 **Step 2/5:** Searching academic databases..."})
        socketio.emit('receive_message', {'sender': 'bot', 'message': "🔍 **Searching arXiv** (preprint server)..."})
        
        # Show keyword search progress
        total_searches = len(related_topics)
        for i, (topic, keywords) in enumerate(keywords_by_topic.items(), 1):
            socketio.emit('receive_message', {
                'sender': 'bot', 
                'message': f"🔍 **Searching topic {i}/{total_searches}:** {topic[:50]}{'...' if len(topic) > 50 else ''}"
            })
            socketio.emit('receive_message', {
                'sender': 'bot', 
                'message': f"   → Using keywords: {', '.join(keywords[:3])}{'...' if len(keywords) > 3 else ''}"
            })
            time.sleep(0.5)  # Small delay to show progress
        
        # Collect papers
        papers = collect_papers(keywords_by_topic, cutoff_year)
        socketio.emit('receive_message', {'sender': 'bot', 'message': f"✅ **Found {len(papers)} papers** from all sources!"})
        
        # Step 3: Filtering
        socketio.emit('receive_message', {'sender': 'bot', 'message': f"🔍 **Step 3/5:** Filtering and cleaning {len(papers)} papers..."})
        
        # DOI filtering
        papers_before = len(papers)
        papers = filter_by_doi(papers)
        socketio.emit('receive_message', {'sender': 'bot', 'message': f"✅ DOI validation: {papers_before} → **{len(papers)} papers** (removed {papers_before - len(papers)} without DOI)"})
        
        # Abstract filtering  
        papers_before = len(papers)
        papers = filter_by_abstract(papers)
        socketio.emit('receive_message', {'sender': 'bot', 'message': f"✅ Abstract validation: {papers_before} → **{len(papers)} papers** (removed {papers_before - len(papers)} without abstract)"})
        
        # Deduplication
        papers_before = len(papers)
        papers = dedupe_by_doi(papers)
        socketio.emit('receive_message', {'sender': 'bot', 'message': f"✅ Duplicate removal: {papers_before} → **{len(papers)} unique papers** (removed {papers_before - len(papers)} duplicates)"})
        
        # Step 4: Semantic Ranking
        socketio.emit('receive_message', {'sender': 'bot', 'message': "⚖️ **Step 4/5:** Analyzing relevance using AI embeddings..."})
        socketio.emit('receive_message', {'sender': 'bot', 'message': "🧠 Loading SciBERT model for semantic analysis..."})
        
        ranked = semantic_rank_papers(title, papers, top_n=30)
        socketio.emit('receive_message', {'sender': 'bot', 'message': f"✅ **Ranked all {len(papers)} papers** by relevance to your research!"})
        
        # Step 5: Results Preparation
        socketio.emit('receive_message', {'sender': 'bot', 'message': "📋 **Step 5/5:** Preparing your curated results with APA citations..."})
        
        # Calculate timing
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        
        # Make sure we have enough results
        result_count = min(len(ranked), 30)
        focused_count = min(20, result_count)
        exploratory_count = max(0, min(10, result_count - focused_count))
        
        focused_results = ranked[:focused_count]
        exploratory_results = ranked[focused_count:focused_count+exploratory_count]
        
        # Success message with timing
        socketio.emit('receive_message', {
            'sender': 'bot', 
            'message': f"🎉 **Search completed in {minutes}m {seconds}s!** Found **{len(papers)}** relevant papers, curated into two lists:"
        })
        
        # Send focused results with APA citations
        socketio.emit('receive_message', {
            'sender': 'bot', 
            'message': f"🎯 **TOP {len(focused_results)} FOCUSED RESULTS** (Best matches for your research):"
        })
        
        for i, paper in enumerate(focused_results, 1):
            result_msg = format_paper_with_citation(paper, i)
            socketio.emit('receive_message', {'sender': 'bot', 'message': result_msg})
            time.sleep(0.2)  # Small delay between results for readability
        
        # Send exploratory results with APA citations if we have any
        if exploratory_results:
            socketio.emit('receive_message', {
                'sender': 'bot', 
                'message': f"🔍 **TOP {len(exploratory_results)} EXPLORATORY RESULTS** (Broader related research):"
            })
            
            for i, paper in enumerate(exploratory_results, 1):
                result_msg = format_paper_with_citation(paper, i)
                socketio.emit('receive_message', {'sender': 'bot', 'message': result_msg})
                time.sleep(0.2)  # Small delay between results
        
        # Generate complete bibliography
        all_results = focused_results + exploratory_results
        if all_results:
            socketio.emit('receive_message', {
                'sender': 'bot',
                'message': "📚 **COMPLETE BIBLIOGRAPHY** (Copy-paste ready for your references):"
            })
            
            bibliography = generate_bibliography(all_results)
            socketio.emit('receive_message', {'sender': 'bot', 'message': bibliography})
        
        # Clear processing flag and set search complete flag
        socketio.emit('set_context', {'processing': False, 'search_complete': True})
        
        # Final message
        socketio.emit('receive_message', {
            'sender': 'bot',
            'message': "✅ **Search complete!** 🎉\n\nI've found the most relevant papers for your research with properly formatted APA citations. Would you like to:\n• Search for another topic: 'Research title: [new title]'\n• Ask me questions about the results\n• Get help with other citation formats (MLA, Chicago, etc.)"
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in pipeline: {error_details}")
        socketio.emit('receive_message', {
            'sender': 'bot',
            'message': f"❌ **An error occurred during the search:**\n\n{str(e)}\n\nPlease try again with a different research title or check your internet connection."
        })
        # Clear processing flag on error and reset search_complete
        socketio.emit('set_context', {'processing': False, 'search_complete': False})

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
            f.write("""
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