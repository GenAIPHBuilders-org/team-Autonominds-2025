# web_collector.py
# Modified version of paper_collector.py for the web app

# Import all the functions we need from the original module
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set the API key before importing the original module
os.environ["GEMINI_API_KEY"] = "AIzaSyCN6hmeeB5sxliWoYL7OVxHxTSvgbnGCko"

# Now import functions from the original module
from paper_collector import (
    generate_topics, 
    generate_keywords, 
    collect_papers, 
    generate_domain_terms,
    generate_app_terms,
    generate_tech_terms
)

# You can add web-specific modifications or wrapper functions here if needed