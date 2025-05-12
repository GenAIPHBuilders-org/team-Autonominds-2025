# Libraries and Packages
import os
import time
import random
from typing import List
from dotenv import load_dotenv
from google import genai
import requests
import feedparser
import json
from pathlib import Path

METHODOLOGY_STOP_TERMS = {
    "survey", "review", "framework", "architecture", "architectures",
    "method", "methods", "approach", "approaches",
    "system", "systems", "analysis", "analyses",
    "algorithm", "algorithms", "technique", "techniques"
}

# Load API Key
load_dotenv()
print("CWD:", os.getcwd())
print("ENV file:", os.path.abspath(".env"), "→ exists?", os.path.exists(".env"))
print("GEMINI_API_KEY:", os.getenv("GEMINI_API_KEY"))
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not set. Check your .env or environment variables.")
client = genai.Client(api_key=api_key)

# Constant Variables
paper_per_keyword = 4
max_fetch_attempts = 3    # how many pages/offsets to try per keyword

### TWO-LAYERED KEYWORD EXTRACTION ###

# Layer 1 | Title-to-Topic Mapping
def generate_topics(title) -> List[str]:
    prompt = f"""You are an expert academic research assistant.

Given a research paper title, generate EXACTLY 4 academic topic labels that best frame the “Related Work” section.  You must include:
  • 2 Core topics—directly at the heart of the title’s domain  
  • 1 Adjacent topic—close sibling areas that often cross-pollinate  
  • 1 Emerging topic—a nascent or hot area on the horizon  

For each topic, output as:

<Label> (<Category>): a 6–10 word phrase.

Do **not** include any explanation, preamble, or commentary.  
Respond **with just** the four topics, either:
  - **One per line**, **OR**  
  - **A single comma-separated list**  

Avoid bullets, numbering, or extra text.  
Research Title: {title}
"""
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    text = response.text.strip()
    
    # 1) Try splitting on newlines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    # 2) If that yields exactly one line, assume comma-separated
    if len(lines) == 1:
        lines = [t.strip() for t in lines[0].split(",") if t.strip()]
    
    # 3) Validate count
    if len(lines) != 4:
        raise RuntimeError(f"Expected 4 topics, got {len(lines)}: {lines}")
    
    return lines

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

def search_arxiv(keyword, start_index=0):
    try:
        results = []
        query = (
            f"http://export.arxiv.org/api/query?"
            f"search_query=all:{keyword.replace(' ', '%20')}"
            f"&start={start_index}&max_results={paper_per_keyword}"
        )
        feed = feedparser.parse(query)
        
        for entry in feed.entries:
            # Try to pull a DOI if arXiv has one; otherwise use the arXiv ID
            doi = getattr(entry, 'arxiv_doi', None)
            prefix = "https://doi.org/"
            
            if not doi:
                # entry.id looks like "http://arxiv.org/abs/YYMM.NNNNN"
                prefix = "arXiv:"
                doi = entry.id.rsplit('/', 1)[-1]

            results.append({
                "title": entry.title,
                "authors": [author.name for author in entry.authors],
                "abstract": entry.summary,
                "doi": prefix + doi,
                "year": int(entry.published[:4]),
                "source": "arXiv"
            })
            
        return results
    except Exception as e:
        print(f"arXiv error: {e}")
        
        return []

def search_semantic_scholar(keyword, offset=0):
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        # Request the `doi` field explicitly instead of `url`
        params = {
            "query": keyword,
            "limit": paper_per_keyword,
            "offset": offset,
            "fields": "title,authors,abstract,doi,year"
        }
        
        response = requests.get(url, params=params)
        
        data = response.json().get("data", [])
        # Ensure each record maps into our common schema
        results = []
        for rec in data:
            yr = rec.get("year")
            # rec["doi"] may be None or empty if not provided
            results.append({
                "title": rec.get("title"),
                "authors": [a.get("name") for a in rec.get("authors", [])],
                "abstract": rec.get("abstract"),
                "doi": "https://doi.org/" + (rec.get("doi") or ""),
                "year": int(yr) if isinstance(yr, int) or yr.isdigit() else None,
                "source": "SemanticScholar"
            })
        return results
    except Exception as e:
        print(f"Semantic Scholar error: {e}")
        return []

def collect_papers(keywords_by_topic, cutoff_year: int):
    """
    For each (topic → keywords), gather up to `paper_per_keyword` papers PER keyword PER source,
    filtering out any with year < cutoff_year, retrying up to max_fetch_attempts.
    """
    papers = []
    
    for topic, keywords in keywords_by_topic.items():
        print(f"\n🔍 Searching for topic: {topic}")
        
        for keyword in keywords:
            print(f"    → Using keyword: {keyword}")
            
            # --- arXiv with retry & cutoff ---
            attempt = 0
            while attempt < max_fetch_attempts:
                start = attempt * paper_per_keyword
                candidates = search_arxiv(keyword, start_index=start)
                valid = [p for p in candidates if p.get("year", 0) >= cutoff_year]
                papers.extend(valid)
                # stop early if we've got enough valid papers
                if len(valid) >= paper_per_keyword:
                    break
                attempt += 1

            # --- Semantic Scholar with retry & cutoff ---
            attempt = 0
            while attempt < max_fetch_attempts:
                offset = attempt * paper_per_keyword
                candidates = search_semantic_scholar(keyword, offset=offset)
                valid = [p for p in candidates if p.get("year", 0) >= cutoff_year]
                papers.extend(valid)
                if len(valid) >= paper_per_keyword:
                    break
                attempt += 1

    return papers            
            
def generate_domain_terms(title: str, max_terms: int = 10) -> List[str]:
    """
    Ask the Gemini client to list the top domain-specific keywords/phrases
    for a paper titled `title`. Returns a cleaned list of terms.
    """
    prompt = (
    f"""You are an expert academic research assistant. 
Given a research paper title, generate exactly 15 domain-specific keywords or short key phrases that any relevant paper’s title or abstract should contain at least one of. 
These terms will be used to filter out generic or off-topic literature.
**Do not include broad methodology terms** such as surveys, reviews, architectural patterns, or generic frameworks. 
Only return terms that reflect the core subject matter and application domain (including synonyms and jargon).
Respond with a single comma-separated list (no numbering, bullets, or commentary), all in lowercase.

Research Title: {title}
"""
)
    # Use the same API as generate_topics / generate_keywords
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    text = response.text.strip()
    # Split on commas, strip whitespace, lowercase
    terms = [t.strip().lower() for t in text.split(",")]
    # Deduplicate while preserving order
    seen = set()
    cleaned = []
    for term in terms:
        if term and term not in seen:
            cleaned.append(term)
            seen.add(term)
    
    # 1. Remove any term containing a stop-term as a standalone word
    filtered = []
    for term in cleaned:
        words = set(term.split())
        if words.isdisjoint(METHODOLOGY_STOP_TERMS):
            filtered.append(term)
    
    # 2. If you ended up with more than needed, truncate; if fewer, consider logging a warning
    return filtered[:max_terms]

def generate_app_terms(title: str, max_terms: int = 5) -> List[str]:
     """
     Extract application-centric phrases (domain + object) from the title.
     """
     prompt = (
         f"You are an expert academic research assistant.\n"
         f"Given a research paper title, list the {max_terms} most important "
         f"application phrases (domain and object) from that title. "
         f"Do NOT include any methodology or technique words.\n"
         f"Respond with a comma-separated list of lowercased phrases, no commentary.\n\n"
         f"Research Title: {title}"
     )
     resp = client.models.generate_content(
         model="gemini-2.0-flash",
         contents=prompt
     )
     raw = resp.text.strip().lower().split(",")
     # clean & dedupe
     out, seen = [], set()
     for t in raw:
         s = t.strip()
         if s and s not in seen:
             seen.add(s); out.append(s)
         if len(out) >= max_terms:
             break
     return out

def generate_tech_terms(title: str, max_terms: int = 5) -> List[str]:
     """
     Extract technique-centric explainability phrases relevant to the title.
     """
     prompt = (
         f"You are an expert academic research assistant.\n"
         f"Given a research paper title, list the {max_terms} most important "
         f"technique or methodology phrases (e.g. explainability, interpretability) "
         f"relevant to that title. Do NOT include domain or object words.\n"
         f"Respond with a comma-separated list of lowercased phrases, no commentary.\n\n"
         f"Research Title: {title}"
     )
     resp = client.models.generate_content(
         model="gemini-2.0-flash",
         contents=prompt
     )
     raw = resp.text.strip().lower().split(",")
     out, seen = [], set()
     for t in raw:
         s = t.strip()
         if s and s not in seen:
             seen.add(s); out.append(s)
         if len(out) >= max_terms:
             break
     return out

### MAIN ###

if __name__ == "__main__":
    title = input("Enter research title: ")
    
    related_topics = generate_topics(title)
    keywords_by_topic = generate_keywords(related_topics)
    
    print("\n🔑 Generated Keywords:")
    for topic, keywords in keywords_by_topic.items():
        print(f"\n{topic}: {keywords}")
        
    papers = collect_papers(keywords_by_topic)
    
    out_path = Path("raw_candidates.json")
    # Wrap with the query title
    payload = {
        "query_title": title,
        "papers": papers
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved {len(papers)} papers to {out_path.resolve()}")
    
    print(f"\n📄 Found {len(papers)} total papers.\n")
    for i, paper in enumerate(papers):        
        print(f"{i+1}. {paper.get('title', 'No title')}")
        print(f"    DOI: {paper.get('doi', '')}")
        print(f"    Year: {paper.get('year', 'N/A')}")
        print(f"    Abstract: {(paper.get('abstract') or '')[:300]}...\n")
