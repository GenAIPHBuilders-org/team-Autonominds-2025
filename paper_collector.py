# Libraries and Packages
import os
from typing import Dict, List
from dotenv import load_dotenv
from google import genai
import time
import requests
import re
import feedparser
import json
from pathlib import Path
from xml.etree import ElementTree as ET
import json
from google.genai import Client, errors as genai_errors
from google.api_core import exceptions as google_exceptions

from filter_and_rank import dedupe_by_doi, filter_by_abstract, filter_by_doi


METHODOLOGY_STOP_TERMS = {
    "survey", "review", "framework", "architecture", "architectures",
    "method", "methods", "approach", "approaches",
    "system", "systems", "analysis", "analyses",
    "algorithm", "algorithms", "technique", "techniques"
}

# Load API Key
from api_client_manager import get_next_api_client


#Enrichment helpers

def fetch_pubmed_abstract(pmid: str) -> str:
    """Use EFetch to retrieve the abstract text for a PubMed ID."""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": pmid, "retmode": "xml"}
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        return ""
    root = ET.fromstring(resp.text)
    texts = [ab.text.strip() for ab in root.findall(".//AbstractText") if ab.text]
    return " ".join(texts)

def enrich_crossref(doi: str) -> dict:
    """Fetch full metadata for a DOI from Crossref’s work endpoint."""
    raw = doi.replace("https://doi.org/", "")
    url = f"https://api.crossref.org/works/{raw}"
    resp = requests.get(url, headers={"Accept": "application/json"})
    if resp.status_code != 200:
        return {}
    msg = resp.json().get("message", {})
    authors = [
        " ".join(filter(None, [a.get("given"), a.get("family")]))
        for a in msg.get("author", [])
    ]
    return {
        "abstract": msg.get("abstract", ""),
        "authors": authors
    }

# Constant Variables
paper_per_keyword = 4
max_fetch_attempts = 1    # how many pages/offsets to try per keyword

### TWO-LAYERED KEYWORD EXTRACTION ###

# Layer 1 | Title-to-Topic Mapping
def generate_topics(title) -> List[str]:
    prompt = f"""You are an expert academic research assistant.

Given a research paper title, generate EXACTLY 4 distinct academic topic labels that best frame the “Related Work” section.  You must include:
  • 2 Core topics—directly at the heart of the title’s domain  
  • 1 Adjacent topic—close sibling areas that often cross-pollinate  
  • 1 Emerging topic—a nascent or hot area on the horizon  

For each topic, output as:

<Label> (<Category>): <a specific 6–10 word phrase description> 

Do **not** include any explanation, preamble, or commentary.  
Respond **with just** the four topics, either:
  - **One per line**, **OR**  
  - **A single comma-separated list**  

Avoid bullets, numbering, or extra text.  
Research Title: {title}
"""
    
    active_client = get_next_api_client() # Get the next client
    response = active_client.models.generate_content( # Use active_client
        model="gemini-2.0-flash",
        contents=prompt
)
    
    # 1) Try splitting on newlines
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
        
        active_client = get_next_api_client() # Get the next client
        response = active_client.models.generate_content( # Use active_client
            model="gemini-2.0-flash",
            contents=prompt
    )
        keywords = response.text.strip().replace('\n', '').split(", ")
        keywords_by_topic[topic] = keywords
        
    return keywords_by_topic

# ─── TWO-STAGE KEYWORD EXTRACTION ───────────────────────────────────────

def generate_subthemes(
    related_topics: List[str],
    max_subthemes: int = 5
) -> Dict[str, List[str]]:
    """
    For each of the 4 high-level topics, ask Gemini to spit out a small list
    of sub-themes (niches) within that topic.
    """
    subthemes_by_topic: Dict[str, List[str]] = {}
    for topic in related_topics:
        prompt = f"""You are an expert academic research assistant.
Given the academic topic "{topic}", list exactly {max_subthemes} key sub-themes
(research niches or angles) that often appear under this topic.
Respond with a single comma-separated list, no bullets or commentary."""

        active_client = get_next_api_client() # Get the next client
        resp = active_client.models.generate_content( # Use active_client
            model="gemini-2.0-flash",
            contents=prompt
    )
        
        text = resp.text.strip()
        # try newline → comma fallback
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) == 1:
            subs = [s.strip() for s in lines[0].split(",") if s.strip()]
        else:
            subs = lines
        subthemes_by_topic[topic] = subs[:max_subthemes]
    return subthemes_by_topic


def generate_keywords_by_subtheme(
        subthemes_by_topic: Dict[str, List[str]],
        max_terms: int = 5,
        output_path: str = "keywords_by_subtheme.json"
) -> Dict[str, Dict[str, List[str]]]:
    """
    For every Topic ▶ Sub-theme pair ask Gemini for up-to-date
    keywords, robustly match Gemini’s JSON back to our expected
    structure, and overwrite *output_path* each run.
    """
    import json, os

    # fresh file every run
    if os.path.exists(output_path):
        os.remove(output_path)

    # ---------- helper for fuzzy key matching ----------
    def best_match(key: str, candidates: list[str]) -> str | None:
        """Return the candidate whose words overlap most with *key*
        (case-insensitive). Priority: exact > substring > Jaccard ≥ 0.4."""
        k_low   = key.lower()
        k_tok   = set(k_low.split())

        # 1) exact
        for c in candidates:
            if k_low == c.lower():
                return c
        # 2) substring
        for c in candidates:
            if k_low in c.lower() or c.lower() in k_low:
                return c
        # 3) token-overlap
        scores = []
        for c in candidates:
            t = set(c.lower().split())
            j = len(k_tok & t) / max(1, len(k_tok | t))
            scores.append((j, c))
        scores.sort(reverse=True)
        return scores[0][1] if scores and scores[0][0] >= 0.4 else None
    # ---------------------------------------------------

    clean_data: Dict[str, Dict[str, List[str]]] = {}

    for raw_topic, subs in subthemes_by_topic.items():
        # ---- Build per-topic prompt ----------------------------------
        prompt_lines = [
            "You are an expert academic research assistant.",
            f"For each sub-theme of the research topic below, give up to {max_terms} "
            "high-specificity search keywords **that combine the sub-theme’s focus "
            "with the parent topic’s context**.  Each keyword should be 2-5 words, "
            "precise, and something likely to appear in title or abstract.",
            " The key words should differ semantically from its siblings (avoid near-synonyms), and ",
            "be suitable for searching scholarly titles/abstracts.",
            "Respond with **only** a JSON object:",
            '{"<Topic>": {"<Sub-theme>": ["kw1","kw2",…] , … } }',
            "",
            "Topic and its Sub-themes:",
            f"- {raw_topic}:"
        ]
        for s in subs:
            prompt_lines.append(f"  • {s}")
        prompt = "\n".join(prompt_lines)
        # --------------------------------------------------------------

        try:
            active_client = get_next_api_client() # Get the next client
            resp = active_client.models.generate_content( # Use active_client
                model="gemini-2.0-flash",
                contents=prompt
        )
        except google_exceptions.GoogleAPIError as e:
            raise RuntimeError(
                f"Gemini API error for topic “{raw_topic}”: {e}"
            ) from e

        raw = resp.text.strip()
        print(f"\n[DEBUG] Gemini raw response for topic “{raw_topic}”:\n{raw}\n")

        # -------- Parse JSON (tolerate ```json fences) -----------------
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(raw[start:end])
            else:
                raise RuntimeError(f"Could not parse JSON for topic “{raw_topic}”")

        # --------- Match Gemini’s topic key ---------------------------
        topic_key = best_match(raw_topic.split(" (")[0], list(data.keys()))
        if not topic_key or not isinstance(data[topic_key], dict):
            raise RuntimeError(f"Topic “{raw_topic}” missing in Gemini JSON")

        topic_dict = data[topic_key]
        clean_data[raw_topic] = {}

        # --------- Match each sub-theme key ---------------------------
        for sub in subs:
            sub_key = best_match(sub, list(topic_dict.keys()))
            if not sub_key:
                print(f"⚠️  No JSON entry for sub-theme “{sub}” "
                      f"under topic “{raw_topic}”; leaving empty.")
                clean_data[raw_topic][sub] = []
                continue

            kw_list = topic_dict[sub_key]
            if not isinstance(kw_list, list):
                raise RuntimeError(f"Keywords for “{sub_key}” must be a list")
            clean_data[raw_topic][sub] = kw_list[:max_terms]

    # ------------- Persist & return ----------------------------------
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=2)

    return clean_data



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

def search_crossref(keyword, offset=0):
    try:
        url = "https://api.crossref.org/works"
        params = {
            "query": keyword,
            "rows": paper_per_keyword,
            "offset": offset,
            "sort": "relevance",
            "order": "desc"
        }
        
        # Add a polite email to improve rate limits as per Crossref guidelines
        headers = {
            "User-Agent": "LiteratureRetrievalApp/1.0 (mailto:vhienanutafeu@gmail.com)"
        }
        
        try:
            response = requests.get(url,
                                    params=params,
                                    headers=headers,
                                    timeout=10)      # 10s timeout
        except requests.exceptions.RequestException as e:
            print(f"Crossref request failed: {e}")
            return []
        
        if response.status_code != 200:
            print(f"Crossref error: Status code {response.status_code}")
            return []
        
        items = response.json().get("message", {}).get("items", [])
        results = []
        
        for item in items:
            # Parse year
            published = item.get("published-print") or item.get("published-online")
            year = 0
            if published and "date-parts" in published:
                date_parts = published["date-parts"][0]
                if date_parts:
                    year = date_parts[0]

            # Parse authors (may be empty)
            authors = []
            for author in item.get("author", []):
                parts = []
                if author.get("given"):
                    parts.append(author["given"])
                if author.get("family"):
                    parts.append(author["family"])
                if parts:
                    authors.append(" ".join(parts))

            # Format DOI
            doi = item.get("DOI", "")
            if doi and not doi.startswith("https://doi.org/"):
                doi = "https://doi.org/" + doi

            # Base fields
            abstract = item.get("abstract", "") or ""
            # Enrich missing abstract or authors via detailed work lookup
            if not abstract or not authors:
                info = enrich_crossref(doi)
                abstract = abstract or info.get("abstract", "")
                authors  = authors  or info.get("authors", [])

            results.append({
                "title": item.get("title", [""])[0] if item.get("title") else "",
                "authors": authors,
                "abstract": abstract,
                "doi": doi,
                "year": year,
                "source": "Crossref"
            })
            
        return results
    
    except Exception as e:
        print(f"Crossref error: {e}")
        return []

def search_pubmed(keyword, offset=0):
    try:
        # First, search for paper IDs
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": keyword,
            "retmode": "json",
            "retstart": offset,
            "retmax": paper_per_keyword,
            "sort": "relevance"
        }
        
        search_response = requests.get(search_url, params=search_params)
        if search_response.status_code != 200:
            print(f"PubMed search error: Status code {search_response.status_code}")
            return []
            
        search_data = search_response.json()
        pmids = search_data.get("esearchresult", {}).get("idlist", [])
        
        if not pmids:
            return []
        
        # Then fetch details for those IDs
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json"
        }
        
        fetch_response = requests.get(fetch_url, params=fetch_params)
        if fetch_response.status_code != 200:
            print(f"PubMed fetch error: Status code {fetch_response.status_code}")
            return []
            
        fetch_data = fetch_response.json()
        results = []
        
        for pmid in pmids:
            article = fetch_data.get("result", {}).get(pmid, {})
            if not article:
                continue

            # Extract publication year
            pub_date = article.get("pubdate", "")
            year = 0
            if pub_date:
                match = re.search(r'\b(19|20)\d{2}\b', pub_date)
                if match:
                    year = int(match.group(0))

            # Extract authors (may be empty)
            authors = [author["name"] for author in article.get("authors", []) if author.get("name")]

            # Construct DOI or fallback URL
            doi = article.get("elocationid", "")
            if doi.startswith("doi:"):
                doi = "https://doi.org/" + doi[4:]
            else:
                doi = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

            # Fetch the real abstract via EFetch
            abstract = fetch_pubmed_abstract(pmid)

            # Pull the paper’s title out of the ESummary response instead
            paper_title = article.get("title", "").strip()
            
            results.append({
                "title": paper_title,
                "authors": authors,
                "abstract": abstract,
                "doi": doi,
                "year": year,
                "source": "PubMed"
            })
            
        return results
    except Exception as e:
        print(f"PubMed error: {e}")
        return []

def enrich_with_semantic_scholar(papers):
        for p in papers:
            if (not p.get("abstract") or not p.get("authors")) and p.get("doi"):
                recs = search_semantic_scholar(p["doi"], offset=0)
                if recs:
                    rec = recs[0]
                    p["abstract"] = p["abstract"] or rec.get("abstract", "")
                    p["authors"]  = p["authors"]  or rec.get("authors", [])

# You can tune these globally or pass them in
TOTAL_POSSIBLE_PER_KEYWORD = paper_per_keyword * 4  # =16
# to guarantee you run every keyword, set your threshold higher than
# the maximum you could ever collect from just one keyword:
MIN_PER_BUCKET = (TOTAL_POSSIBLE_PER_KEYWORD * 3) + 1  # =65
MAX_FETCH_ATTEMPTS  = 3

def collect_papers(
    keywords_by_topic: Dict[str, Dict[str, List[str]]],
    cutoff_year: int,
    paper_per_keyword: int,
    max_fetch_attempts: int = MAX_FETCH_ATTEMPTS,
    min_per_bucket: int = MIN_PER_BUCKET
) -> List[dict]:
    """
    For each bucket (sub-theme or tech cluster), fetch until you have at least
    `min_per_bucket` papers: try each keyword once, then retry deeper if needed.
    Finally merge all buckets, enrich, filter & dedupe globally.
    """
    
    # — track how many raw hits we fetched (before any filtering)
    raw_fetches = 0

    # 1) Initialize empty lists *per sub-theme* under each topic
    papers_by_bucket: Dict[str, List[dict]] = {}
    for topic, submap in keywords_by_topic.items():
        for subtheme in submap:
            papers_by_bucket[f"{topic} ▶ {subtheme}"] = []

    # 2) First-pass then retries per bucket
    for full_label, collected in papers_by_bucket.items():
        topic, subtheme = full_label.split(" ▶ ", 1)
        # 1️⃣ print the topic
        print(f"\n🔍 Searching for topic: {topic}")
        # 2️⃣ print the sub-theme
        print(f"  ▶ Sub-theme: {subtheme}")
        # 3️⃣ now run through each keyword
        keywords = keywords_by_topic[topic][subtheme]

        # --- First pass: keyword=0 offset ---
        for kw in keywords:
            offset = 0
            if len(collected) >= min_per_bucket:
                print(f"Papers in this bucket is {min_per_bucket}")
                break
            
            print(f"    → Using keyword: {kw}")

            for source_fn in (
                    search_arxiv,
                    search_semantic_scholar,
                    search_pubmed,
                    search_crossref
            ):
                try:
                    # arXiv uses start_index instead of offset
                        if source_fn is search_arxiv:
                            results = source_fn(kw, start_index=offset)
                        else:
                            results = source_fn(kw, offset=offset)

                        raw_fetches += len(results)
                        print(f"      • {source_fn.__name__}: {len(results)} raw hits")
                except Exception:
                    continue

                # -------- filtering / dedupe as before ----------
                valid = dedupe_by_doi(results)
                seen  = {p["doi"] for p in collected}
                new_valid = [p for p in valid if p["doi"] not in seen]

                needed = min_per_bucket - len(collected)
                collected.extend(new_valid[:needed])

                # ✅ 1) Stop the whole bucket if we’re done
                if len(collected) >= min_per_bucket:
                    break                    # ← break out of source loop
            # ──────────────────────────────────────────────────────────────────
            # ⬅︎ after the 4-source loop finishes, add this guard
            if len(collected) >= min_per_bucket:
                break                        # ← break out of keyword loop

        # --- Retry pass: deeper offsets ---
        for attempt in range(1, max_fetch_attempts):
            if len(collected) >= min_per_bucket:
                break

            offset = attempt * paper_per_keyword
            for kw in keywords:
                print(f"    → Retrying keyword: {kw}  (offset {offset})")
                
                if len(collected) >= min_per_bucket:
                    break

                for source_fn in (
                        search_arxiv,
                        search_semantic_scholar,
                        search_pubmed,
                        search_crossref,
                ):
                    try:
                        # arXiv uses start_index instead of offset
                        if source_fn is search_arxiv:
                            results = source_fn(kw, start_index=offset)
                        else:
                            results = source_fn(kw, offset=offset)

                        raw_fetches += len(results)
                        print(f"      • {source_fn.__name__}: {len(results)} raw hits")
                    except Exception as e:
                        # (optional) log e.__class__.__name__ for diagnostics
                        continue

                    # ── FILTER & DEDUPE ───────────────────────────────────────────
                    valid = [p for p in results if p.get("year", 0) >= cutoff_year]
                    valid = filter_by_doi(valid)
                    valid = filter_by_abstract(valid)
                    valid = dedupe_by_doi(valid)           # within this slice

                    # keep only DOIs that are not yet in this bucket
                    seen_doi   = {p["doi"] for p in collected}
                    new_valid  = [p for p in valid if p["doi"] not in seen_doi]

                    # add up to the number still needed
                    needed = min_per_bucket - len(collected)
                    collected.extend(new_valid[:needed])

                    # ✅ if bucket quota reached, stop fetching any further sources / offsets
                    if len(collected) >= min_per_bucket:
                        break  # exits the for-source loop

                # after trying the four sources, bail out of this keyword if quota met
                if len(collected) >= min_per_bucket:
                    break  # exits the for-keyword loop

        papers_by_bucket[full_label] = collected

    # 3) Flatten all buckets into one list
    all_papers = []
    for bucket_list in papers_by_bucket.values():
        all_papers.extend(bucket_list)

    # 4) Final enrichment, filtering, dedupe
    enrich_with_semantic_scholar(all_papers)
    all_papers = filter_by_doi(all_papers)
    all_papers = filter_by_abstract(all_papers)
    all_papers = dedupe_by_doi(all_papers)

    print(f"🔄 Total raw hits fetched (pre-filter): {raw_fetches}")
    print(f"✅ Collected {len(all_papers)} papers across {len(papers_by_bucket)} buckets.")
    return all_papers

            
def generate_domain_terms(title: str, max_terms: int = 10) -> List[str]:
    """
    Ask the Gemini client to list the top domain-specific keywords/phrases
    for a paper titled `title`. Returns a cleaned list of terms.
    """
    prompt = f"""
You are an expert academic research assistant.

Your task: Generate exactly 15 **domain-specific keywords or short phrases** that any relevant paper’s title or abstract should contain at least one of.  
• These terms form your filter: they must capture the core subject matter or application area (including key subdomains and jargon).  
• Do NOT include names of methods, algorithms, surveys, architectures, or generic frameworks.  
• Return exactly 15 comma-separated, lowercase phrases, no numbering, no extra text.

Example for topic “Medical Image Segmentation”:  
medical imaging, image segmentation, radiology, anatomical structures, lesion detection,
pixel-level classification, semantic segmentation, instance segmentation, multimodal fusion,
computer-aided diagnosis, quantitative imaging biomarkers, deep image analysis,
segmentation uncertainty, atlas-based methods, volume rendering

Topic: {title}
"""
    # Use the same API as generate_topics / generate_keywords
    active_client = get_next_api_client() # Get the next client
    response = active_client.models.generate_content( # Use active_client
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

def generate_app_terms(title: str, max_terms: int = 7) -> List[str]:
     """
     Extract application-centric phrases (domain + object) from the title.
     """
     prompt = f"""
You are an expert academic research assistant.

Your task: Given a research paper title, extract exactly {max_terms} **application phrases**—that is, the concrete objects, contexts or tasks being studied (the “what” and “where”).  
• Only output noun or noun-phrase labels that describe the application domain.  
• Do NOT include any method names, algorithm families, evaluation terms, or metrics.
• They must describe the problem space, objects, or domain context (e.g. body part, disease, modality, task)  
• They should **NOT** contain methodology or algorithm words.
• Include at least ONE application term present in the title itself.
• Allow single word synonyms if you deem them as domain critical.
• Return exactly {max_terms} phrases, in a single comma-separated list, all lowercase, no numbering or extra words.

Example for title “Graph Neural Network Architectures for Molecular Property Prediction”:  
molecular property prediction, molecular graphs, drug discovery, materials informatics, cheminformatics

Research Title: {title}
"""
     active_client = get_next_api_client() # Get the next client
     resp = active_client.models.generate_content( # Use active_client
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

def generate_tech_terms(title: str, max_terms: int = 10) -> List[str]:
     """
     Extract technique-centric explainability phrases relevant to the title.
     """
     prompt = f"""
You are an expert academic research assistant.

Your task: From a research paper title, extract exactly {max_terms} **technique or methodology phrases**—the “how” of the work.  
• If relevant and applicable, only include specific algorithm, architecture, or analysis method names (e.g. “saliency maps”, “grad-cam”, “message passing neural networks”), rather than generic words for these.  
• Do NOT include domain words, datasets, or application contexts.
•  Include exactly one technical or methodology term present in the title itself.  
• Return exactly {max_terms} items as a comma-separated, lowercase list with no extra commentary.

Example for title “Explainable AI Techniques for Medical Image Segmentation”:  
saliency maps, grad-cam, shap values, surrogate models, counterfactual explanations

Research Title: {title}
"""
     active_client = get_next_api_client() # Get the next client
     resp = active_client.models.generate_content( # Use active_client
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

    start_time = time.time()

    related_topics = generate_topics(title)
    keywords_by_topic = generate_keywords(related_topics)
    
    print("\n🔑 Generated Keywords:")
    for topic, keywords in keywords_by_topic.items():
        print(f"\n{topic}: {keywords}")
    
    papers = collect_papers(keywords_by_topic)
    
    end_time = time.time()
    
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
        print(f"    Source: {paper.get('source', '')}") # for testing purposes
        print(f"    DOI: {paper.get('doi', '')}")
        print(f"    Year: {paper.get('year', 'N/A')}")
        print(f"    Abstract: {(paper.get('abstract') or '')[:300]}...\n")
        
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    print(f"\n\nPaper collection took {minutes} min {seconds} sec.")
