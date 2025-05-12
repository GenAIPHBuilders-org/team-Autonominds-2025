import logging
import re
from typing import List, Dict, Set, Tuple, Any

# (Optional) persistence for caching your fitted vectorizer
import joblib

# scikit-learn for TF-IDF and similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#loading the JSON file
import json
from pathlib import Path

#load for sciBERT
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from embeddings import embed_text


def load_candidates_from_json(path: str) -> Tuple[str, List[Dict[str, Any]], List[str]]:
    file = Path(path)
    if not file.exists():
        logging.error(f"Candidate file not found: {path}")
        return "", [], []

    with file.open("r", encoding="utf-8") as f:
        container = json.load(f)

    title = container.get("query_title", "")
    papers = container.get("papers", [])
    domain_terms = container.get("domain_terms", [])

    # validate papers as before...
    valid_papers = [p for p in papers if all(k in p for k in ("doi","title","abstract","year","authors"))]
    logging.info(f"Loaded {len(valid_papers)} valid candidates from JSON (out of {len(papers)})")

    return title, valid_papers, domain_terms

def filter_by_doi(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove any entries without a valid 'doi' key or with an empty DOI string.
    """
    before = len(papers)
    filtered = [p for p in papers if p.get("doi") and isinstance(p["doi"], str) and p["doi"].strip()]
    after = len(filtered)
    logging.info(f"DOI filter: {before} → {after} (removed {before - after})")
    return filtered

def filter_by_abstract(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove any entries without a valid 'abstract' key or with an empty abstract string.
    """
    before = len(papers)
    filtered = [
        p for p in papers
        if p.get("abstract") and isinstance(p["abstract"], str) and p["abstract"].strip()
    ]
    after = len(filtered)
    logging.info(f"Abstract filter: {before} → {after} (removed {before - after})")
    return filtered

def dedupe_by_doi(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate papers by their DOI, keeping only the first occurrence.
    """
    before = len(papers)
    seen: set[str] = set()
    unique: List[Dict[str, Any]] = []
    for p in papers:
        doi = p["doi"].strip().lower()
        if doi not in seen:
            unique.append(p)
            seen.add(doi)
    after = len(unique)
    logging.info(f"Deduplication: {before} → {after} (removed {before - after} duplicates)")
    return unique

def _test_doi_filter():
    sample = [
        {"doi": "10.123/abc", "title": "A", "abstract": "x", "year": 2021, "authors": []},
        {"title": "No DOI",      "abstract": "y", "year": 2022, "authors": []},
        {"doi": "",              "title": "Empty", "abstract": "z", "year": 2020, "authors": []},
    ]
    res = filter_by_doi(sample)
    assert len(res) == 1 and res[0]["doi"] == "10.123/abc"
    print("✔️ DOI filter test passed")
    
def _test_abstract_filter():
    sample = [
        {"doi": "1", "abstract": "Valid abstract", "title": "A", "year": 2021, "authors": []},
        {"doi": "2", "abstract": "",               "title": "B", "year": 2022, "authors": []},
        {"doi": "3",                           "title": "C", "year": 2020, "authors": []},  # missing key
    ]
    res = filter_by_abstract(sample)
    assert len(res) == 1 and res[0]["doi"] == "1"
    print("✔️ Abstract filter test passed")
    
def _test_dedupe():
    sample = [
        {"doi": "A", "title": "One", "abstract": "x", "year": 2021, "authors": []},
        {"doi": "a", "title": "One Dup", "abstract": "y", "year": 2022, "authors": []},
        {"doi": "B", "title": "Two",      "abstract": "z", "year": 2020, "authors": []},
    ]
    res = dedupe_by_doi(sample)
    assert len(res) == 2
    assert any(p["doi"].lower() == "a" for p in res)
    assert any(p["doi"] == "B" for p in res)
    print("✔️ Deduplication test passed")
    
#ranking

def rank_papers(
    query: str,
    papers: List[Dict[str, Any]],
    top_n: int = 10,
    boost_domain: bool = True,
    boost_term: str = "molecular",
    boost_factor: float = 0.15
) -> List[Dict[str, Any]]:
    """
    Rank papers by TF-IDF + bigrams similarity to query, with optional boosting for domain relevance.
    Returns top_n papers each annotated with a normalized 'score'.
    """
    if not papers:
        return []

    # 1. Build texts
    texts = [f"{p['title']} {p['abstract']}" for p in papers]

    # 2. TF-IDF vectorization with unigrams + bigrams
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        max_df=0.8,
        min_df=1,
        ngram_range=(1, 2)
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    # 3. Vectorize query
    query_vec = vectorizer.transform([query])

    # 4. Cosine similarity
    raw_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # 5. Boost domain relevance if applicable
    boosted_scores = []
    for score, text in zip(raw_scores, texts):
        if boost_domain and boost_term.lower() in text.lower():
            score *= (1 + boost_factor)
        boosted_scores.append(score)

    # 6. Normalize scores to [0, 1]
    max_score = max(boosted_scores) if boosted_scores else 1
    min_score = min(boosted_scores) if boosted_scores else 0
    range_score = max_score - min_score or 1
    normalized = [(s - min_score) / range_score for s in boosted_scores]

    # 7. Annotate and sort
    for p, s in zip(papers, normalized):
        p['score'] = float(s)

    ranked = sorted(papers, key=lambda x: x['score'], reverse=True)
    return ranked[:top_n]

def _test_rank_papers():
    # Minimal smoke test
    sample = [
        {"title": "Deep learning in robotics", "abstract": "We apply deep nets", "doi":"d1", "year":2020, "authors":[]},
        {"title": "Classical control",           "abstract": "PID controllers",     "doi":"d2", "year":2018, "authors":[]},
        {"title": "Reinforcement learning",      "abstract": "RL algorithms",       "doi":"d3", "year":2019, "authors":[]},
    ]
    ranked = rank_papers("deep reinforcement robotics", sample, top_n=2)
    assert len(ranked) == 2
    assert ranked[0]["doi"] == "d1"
    print("✔️ Rank papers test passed")
    
def semantic_rank_papers(
    query: str,
    papers: List[Dict[str, Any]],
    top_n: int = 10,
    use_cache: bool = True
) -> List[Dict[str, Any]]:
    """
    Rank papers by cosine similarity on SciBERT embeddings.
    """
    # 1. Compute or load embeddings for each paper
    doc_vecs = []
    for p in papers:
        text = f"{p['title']} {p['abstract']}"
        vec = embed_text(text, use_cache=use_cache)
        doc_vecs.append(vec)

    doc_matrix = np.vstack(doc_vecs)          # shape: (num_papers, 768)

    # 2. Embed the query
    q_vec = embed_text(query, use_cache=use_cache)

    # 3. Compute cosine similarities
    # Add small epsilon to denominators to avoid zero-division
    norms = np.linalg.norm(doc_matrix, axis=1) * (np.linalg.norm(q_vec) + 1e-9)
    sims = (doc_matrix @ q_vec) / norms       # shape: (num_papers,)

    # 4. Attach and sort
    for p, s in zip(papers, sims):
        p["score"] = float(s)

    ranked = sorted(papers, key=lambda p: p["score"], reverse=True)
    return ranked[:top_n]

def _test_semantic_rank():
    # Minimal sanity check: two very different docs
    sample = [
        {"title": "Molecular property prediction for drugs", "abstract": "We test on ADMET.", "doi":"d1","year":2020,"authors":[]},
        {"title": "Graph signal processing in networks",      "abstract": "Sampling on graphs.",     "doi":"d2","year":2020,"authors":[]},
    ]
    ranked = semantic_rank_papers("molecular property prediction", sample, top_n=2)
    assert ranked[0]["doi"] == "d1"
    print("✔️ Semantic rank test passed")

def filter_by_domain(
    papers: List[Dict[str, Any]],
    domain_terms: Set[str]
) -> List[Dict[str, Any]]:
    """
    Keep only papers whose title or abstract contains at least one term
    from `domain_terms`.
    """
    before = len(papers)
    patterns = [
        re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        for term in domain_terms
    ]

    filtered = [
        p for p in papers
        if any(pat.search(f"{p.get('title','')} {p.get('abstract','')}") for pat in patterns)
    ]

    logging.info(f"Domain filter: {before} → {len(filtered)} (removed {before - len(filtered)})")
    return filtered

def filter_by_core(
    papers: List[Dict[str, Any]],
    core_terms: Set[str]
) -> List[Dict[str, Any]]:
    """
    Keep only papers whose title or abstract contains at least one core term.
    """
    before = len(papers)
    patterns = [
        re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        for term in core_terms
    ]

    filtered = [
        p for p in papers
        if any(pat.search(f"{p.get('title','')} {p.get('abstract','')}") for pat in patterns)
    ]
    logging.info(f"Core filter: {before} → {len(filtered)} (removed {before - len(filtered)})")
    return filtered

from collections import Counter
def infer_boost_terms(
    ranked_papers: List[Dict[str, Any]],
    domain_terms: List[str],
    top_k: int = 20,
    multiplier_threshold: float = 1.5
) -> List[str]:
    """
    Automatically pick which domain_terms to boost, based on their
    relative frequency in the top_k papers vs. the rest.
    """
    top_texts    = [f"{p['title']} {p['abstract']}".lower() for p in ranked_papers[:top_k]]
    bottom_texts = [f"{p['title']} {p['abstract']}".lower() for p in ranked_papers[top_k:]]
    top_counts    = Counter(term for term in domain_terms for text in top_texts    if term in text)
    bottom_counts = Counter(term for term in domain_terms for text in bottom_texts if term in text)
    n_top    = len(top_texts)
    n_bottom = len(bottom_texts) or 1
    boost_terms = []
    for term in domain_terms:
        lift = (top_counts[term]/n_top) / ((bottom_counts[term]/n_bottom) + 1e-9)
        if lift >= multiplier_threshold:
            boost_terms.append(term)
    return boost_terms

## MAIN ##
if __name__ == "__main__":
    # 1. Load raw candidates
    candidates = load_candidates_from_json("raw_candidates.json")
    if not candidates:
        raise SystemExit("❌ No valid candidate papers loaded—check your JSON file.")

    _test_doi_filter()
    _test_abstract_filter()
    _test_dedupe()
    
    # 2. Filter by DOI
    papers = filter_by_doi(candidates)
    if not papers:
        raise SystemExit("❌ All candidates dropped by DOI filter—nothing to rank.")
    
    # 2a. Filter by Abstract
    papers = filter_by_abstract(papers)
    if not papers:
        raise SystemExit("❌ All candidates dropped by abstract filter—nothing to rank.")

    # 3. rank by TF-IDF
    _test_rank_papers()

    # Placeholder for next steps:
    # papers = filter_by_year(papers)
    # papers = dedupe_by_doi(papers)
    # ...