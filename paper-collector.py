# Libraries and Packages
import os
import time
import random
from dotenv import load_dotenv
from google import genai
import requests
import feedparser

# Load API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# Constant Variables
paper_per_keyword = 3

### TWO-LAYERED KEYWORD EXTRACTION ###

# Layer 1 | Title-to-Topic Mapping
def generate_topics(title):
    prompt = f"""You are an expert academic research assistant.
Given a research title, generate a list of 3-5 most relevant and meaningful academic topics associated with it.
Your output should include a mix of core, adjacent, and emerging topics that could appear in academic journals.
Prioritize conceptual breadth, domain relevance, and interdisciplinary connections.

Research Title: {title}

IMPORTANT RULES:
Do not include any explanation, preamble, or commentary.
Output the topics as a comma-separated list, with no bullet points, no numbering, and no newline characters.
Each topic should be short (max 5 words) and represent a real academic field or concept.
Do not include anything like “Sure!” or “Here are the topics:”. Only return the list.
"""
    
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return response.text.strip().replace('\n', '').split(", ")

# Layer 2 | Topic-to-Keyword Mapping
def generate_keywords(related_topics):
    keywords_by_topic = {}
    
    for topic in related_topics:
        prompt = f"""You are an expert researcher and domain authority in the field of {topic}. Your task is to generate a list of detailed, search-optimized academic keywords and search phrases that a researcher would use to find scholarly literature specifically related to this topic.

These keywords must vary in scope and specificity — including synonyms, technical terms, and real-world search phrases. Prioritize both precision and coverage, ensuring that the result is suitable for use in academic databases.

Topic: {topic}

IMPORTANT RULES:

Do not include any explanation, preamble, or commentary.

Output the keywords as a comma-separated list, with no bullet points, no numbering, and no newline characters.

Use a mix of short and phrase-based keywords (3–5 words max per phrase).

Do not include any punctuation other than the separating commas.

Return exactly 5 to 10 keywords."""
        
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        keywords = response.text.strip().replace('\n', '').split(", ")
        keywords_by_topic[topic] = keywords
        
    return keywords_by_topic

### LITERATURE RETRIEVAL ###

def search_arxiv(keyword):
    try:
        results = []
        
        query = f"http://export.arxiv.org/api/query?search_query=all:{keyword.replace(" ", "%20")}&start=0&max_results=3"
        feed = feedparser.parse(query)
        
        for entry in feed.entries:
            results.append({
                "title": entry.title,
                "authors": [author.name for author in entry.authors],
                "abstract": entry.summary,
                "url": entry.link,
                "year": entry.published[:4]
            })
            
        return results
    except Exception as e:
        print(f"arXiv error: {e}")
        
        return []

def search_semantic_scholar(keyword):
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {"query": keyword, "limit": 3, "fields": "title,authors,abstract,url,year"}
        
        response = requests.get(url, params=params)
        
        return response.json().get("data", [])
    except Exception as e:
        print(f"Semantic Scholar error: {e}")
        return []

def collect_papers(keywords_by_topic):
    papers = []
    
    for topic, keywords in keywords_by_topic.items():
        print(f"\n🔍 Searching for topic: {topic}")
        
        for keyword in keywords:
            print(f"    → Using keyword: {keyword}")
            papers += search_arxiv(keyword)
            # time.sleep(random.uniform(0.5, 1.5))
            papers += search_semantic_scholar(keyword)
            # time.sleep(random.uniform(0.5, 1.5))
            
    return papers

### MAIN ###

if __name__ == "__main__":
    title = input("Enter research title: ")
    
    related_topics = generate_topics(title)
    keywords_by_topic = generate_keywords(related_topics)
    
    print("\n🔑 Generated Keywords:")
    for topic, keywords in keywords_by_topic.items():
        print(f"\n{topic}: {keywords}")
        
    papers = collect_papers(keywords_by_topic)
    
    print(f"\n📄 Found {len(papers)} total papers.\n")
    for i, paper in enumerate(papers):        
        print(f"{i+1}. {paper.get('title', 'No title')}")
        print(f"    URL: {paper.get('url', '')}")
        print(f"    Year: {paper.get('year', 'N/A')}")
        print(f"    Abstract: {(paper.get('abstract') or '')[:300]}...\n")
