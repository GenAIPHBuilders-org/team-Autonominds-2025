import json
import sys
from pathlib import Path
from typing import Dict, List
import time
from embeddings import embed_text
from sklearn.cluster import KMeans
import numpy as np
from dotenv import load_dotenv
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)


# Import modules
from paper_collector import generate_topics, generate_keywords, collect_papers, generate_subthemes, generate_keywords_by_subtheme, generate_domain_terms
from filter_and_rank import (
    load_candidates_from_json,
    filter_by_doi,
    filter_by_abstract,
    dedupe_by_doi,
    semantic_rank_papers,
    infer_boost_terms
)
#critique modules
from keyword_critic import critique_list, critique_map

# List of stop terms
METHODOLOGY_STOP_TERMS = {
    "survey", "review", "framework", "architecture", "architectures",
    "method", "methods", "approach", "approaches",
    "system", "systems", "analysis", "analyses",
    "algorithm", "algorithms", "technique", "techniques"
}
# Tier-1: absolutely prohibited words
STOP_TIER1 = {
    "survey", "review", "framework", "architecture", "architectures",
    "analysis", "analyses", "system", "systems",
}
# Tier-2: soft-ban words—only ban them if Tier-1 is present
STOP_TIER2 = {
    "method", "methods", "approach", "approaches",
    "algorithm", "algorithms", "technique", "techniques",
}

MAX_TECH_BOOSTS = 5

#cleaner helper

def clean_terms(terms: List[str]) -> List[str]:
    """
    Remove any term containing a Tier-1 word.
    If that yields none, remove only Tier-2 words.
    Finally, if still empty, return the raw list.
    """
    # First pass: strip Tier-1
    tier1_clean = [t for t in terms if set(t.split()).isdisjoint(STOP_TIER1)]
    if tier1_clean:
        return tier1_clean
    # Second pass: strip Tier-2
    tier2_clean = [t for t in terms if set(t.split()).isdisjoint(STOP_TIER2)]
    if tier2_clean:
        return tier2_clean
    # Fallback to raw if everything got removed
    return terms

def prompt_cutoff_year() -> int:
    while True:
        yr = input("Enter publication year cutoff (e.g. 2015): ").strip()
        if yr.isdigit():
            return int(yr)
        print("⚠️  Invalid year. Please enter a four-digit year, e.g. 2015.")

def main():
    # 1. Accept title as CLI arg or prompt
    if len(sys.argv) > 1:
        title = " ".join(sys.argv[1:])
    else:
        title = input("Enter research title: ")

    # 1.2. Cutoff year
    cutoff_year = prompt_cutoff_year()

    # 1.3. Citation style (for future formatting)
    citation_style = input("Enter citation style (e.g. APA): ").strip() or "APA"
    
    # ─────────────────────────────────────────────────────
    # Start timing from here (all inputs collected)
    start_time = time.time()
    # ─────────────────────────────────────────────────────
    
    # 2. Collect & dump raw JSON
    related_topics     = generate_topics(title)
    subthemes_by_topic = generate_subthemes(related_topics, max_subthemes=3)
    raw_keywords = generate_keywords_by_subtheme(subthemes_by_topic, max_terms=5)

    # apply critique_list to each inner list, preserve topic→subtheme→[kw…] shape
    all_candidates = []
    for subs_map_in_raw in raw_keywords.values(): # Corrected iteration
        for kw_list_val in subs_map_in_raw.values(): # Corrected iteration
            all_candidates.extend(kw_list_val)


    # now critique each sub-theme list against that pool
    keywords_by_topic_critiqued = {} # Renaming for clarity
    for topic, subs_map in raw_keywords.items(): # subs_map is {"Subtheme": [original_kw_list]}
        refined_subthemes_map = {}
        for subtheme, original_kw_list_for_subtheme in subs_map.items():
            # Create a proper label for the critic
            critic_label = f"Keywords for topic '{topic}' under subtheme '{subtheme}'"

            # Critique the specific list of keywords for THIS subtheme
            refined_list, suggestions_map = critique_list(
                critic_label,                   # The string label
                original_kw_list_for_subtheme   # The list of keywords to be critiqued
            )
            refined_subthemes_map[subtheme] = refined_list
        keywords_by_topic_critiqued[topic] = refined_subthemes_map

    # After this, keywords_by_topic_critiqued should have the correct
    # nested structure: {"Topic": {"Subtheme": [refined_kws_for_subtheme]}}
    print("✅ Sub-theme keywords refined via critic.")
    if len(related_topics) != 4:
        print(f"⚠️ Warning: expected 4 topics, got {len(related_topics)}")
    
    paper_per_keyword = 3
    
    # `subthemes_by_topic` is {topic: [sub1, sub2, sub3], …}
    # `keywords_by_topic` is still {topic: [kw1, kw2, …], …}
            
    # --- BEGIN DEBUG ---
    print("\n--- DEBUGGING: Right before calling collect_papers ---")
    # Replace 'keywords_by_subtheme_data_for_collection' with the actual variable name you use
    actual_data_passed = keywords_by_topic_critiqued # Or whatever your variable is named

    print(f"Variable name being passed to collect_papers: keywords_by_subtheme (in run_pipeline.py)")
    print(f"Type of this variable: {type(actual_data_passed)}")

    if isinstance(actual_data_passed, dict) and actual_data_passed:
        first_topic_key = list(actual_data_passed.keys())[0]
        print(f"First key (topic): '{first_topic_key}'")
        value_for_first_topic = actual_data_passed[first_topic_key]
        print(f"Type of value for first topic: {type(value_for_first_topic)}")
        print(f"Value for first topic: {value_for_first_topic}")

        if isinstance(value_for_first_topic, dict) and value_for_first_topic:
            first_subtheme_key = list(value_for_first_topic.keys())[0]
            print(f"  First subtheme key under this topic: '{first_subtheme_key}'")
            keywords_for_first_subtheme = value_for_first_topic[first_subtheme_key]
            print(f"  Type of keywords for this subtheme: {type(keywords_for_first_subtheme)}")
            print(f"  Keywords for this subtheme: {keywords_for_first_subtheme}")
    print("--- END DEBUG ---\n")

    # Generate domain terms, critique them.
    domain_terms_raw = generate_domain_terms(title, max_terms=10)
    domain_terms, domain_suggestions = critique_list("Domain terms", domain_terms_raw)

    # now collect with quotas PER SUB-THEME
    papers = collect_papers(
        keywords_by_topic_critiqued,
        cutoff_year=cutoff_year,
        paper_per_keyword=paper_per_keyword
    )
    
    out_path = Path("raw_candidates.json")
    # Remove old file if it exists
    if out_path.exists():
        out_path.unlink()
        print(f"🗑  Deleted old {out_path.name}")

    # Dump new results
    payload = {
        "query_title": title,
        "domain_terms": domain_terms,
        "papers": papers
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"✅ Collected and saved {len(papers)} papers (≥ {cutoff_year}).")

    # 3. Load & filter
    query_title, candidates, domain_terms = load_candidates_from_json(str(out_path))
    papers = filter_by_doi(candidates)
    papers = filter_by_abstract(papers)
    papers = dedupe_by_doi(papers)

    print(f"\n🚀 {len(papers)} papers remain after DOI, abstract & dedup filters.")

    # 7. Semantic ranking of the full pool
    from filter_and_rank import semantic_rank_papers, filter_by_domain, filter_by_core
    from paper_collector import generate_app_terms, generate_tech_terms 

   # generate the raw lists
    app_terms_raw    = generate_app_terms(title, max_terms=5)
    tech_terms_raw   = generate_tech_terms(title, max_terms=8)
    

    # run them through the critic
    app_critiqued, app_suggestions       = critique_list("Application terms", app_terms_raw)
    tech_critiqued, tech_suggestions     = critique_list("Technique terms", tech_terms_raw)

    # (optional) print out what got rejected & suggested replacements
    print(f"📝 App terms suggestions: {app_suggestions}")
    print(f"📝 Tech terms suggestions: {tech_suggestions}")
    print(f"📝 Domain terms suggestions: {domain_suggestions}")

    # 1) Cleaned APP terms: only multi-word phrases
    app_terms = clean_terms(app_critiqued)
    app_terms = [t for t in app_terms if len(t.split()) > 1]

    # 1b) Anchor to the exact title phrase
    full_phrase = title.lower().strip()
    if full_phrase not in app_terms:
        app_terms.insert(0, full_phrase)

    # 2) Cleaned TECH terms
    tech_terms = clean_terms(tech_critiqued)

    # 2b) If all pruning removed them, fall back to LLM’s own top suggestions
    if not tech_terms:
        tech_terms = tech_critiqued[:3]  # take the first 3 phrases the LLM gave us

    print(f"🔑 Final application terms: {app_terms}")
    print(f"🔑 Final technique terms:   {tech_terms}")
    print(f"🔑 Final domain terms: {domain_terms}")
    
    # If either list is empty, fallback to original raw lists
    if not app_terms:
        print("⚠️ App terms empty; using raw app extraction")
        app_terms = generate_app_terms(title, max_terms=5)
        print(f"🔑 Application core terms: {app_terms}")
    if not tech_terms:
        print("⚠️ Tech terms empty; using raw tech extraction")
        tech_terms = generate_tech_terms(title, max_terms=5)
        print(f"🔑 Technique core terms:   {tech_terms}")
    
    
    # ─── Embed & cluster technique terms ──────────────────────────────────
    tech_vecs = [embed_text(t, use_cache=True) for t in tech_terms]

    if tech_vecs and len(tech_terms) > 4:
        import math, numpy as np
        max_k       = 5
        n_clusters  = min(max_k, max(2, len(tech_terms)//2))
        kmeans      = KMeans(n_clusters=n_clusters, random_state=0)
        labels      = kmeans.fit_predict(np.vstack(tech_vecs))

        cluster_map: dict[int, list[str]] = {}
        for term, lbl in zip(tech_terms, labels):
            cluster_map.setdefault(lbl, []).append(term)

        rep_terms: list[str] = []

        for lbl, terms in cluster_map.items():
            # how many to keep from this cluster?
            keep_k = max(1, math.ceil(math.sqrt(len(terms))))      # ≤3 when len(terms) ≤9
            if len(terms) <= keep_k:
                rep_terms.extend(terms)
                continue

            # distance to centroid → pick the closest `keep_k`
            idxs   = [tech_terms.index(t) for t in terms]
            centre = kmeans.cluster_centers_[lbl]
            dists  = [(t, np.linalg.norm(tech_vecs[i] - centre)) for t, i in zip(terms, idxs)]
            dists.sort(key=lambda x: x[1])
            rep_terms.extend([t for t, _ in dists[:keep_k]])

        tech_terms = rep_terms
        print("🔑 Technique clusters:")
        for lbl, terms in cluster_map.items():
            print(f"    cluster {lbl}: {terms}")
        print(f"🔑 Representative technique terms (≤√n per cluster): {tech_terms}")
    else:
        print("🔑 Technique clustering skipped (≤ 4 terms)")


    # 7c. Compile regex patterns
    import re
    app_patterns  = [re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in app_terms]
    tech_patterns = [re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in tech_terms]
    domain_patterns = [re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in domain_terms]

    # 3. Rank all papers semantically
    print("Using SciBERT to rank the collected papers...")
    ranked_all = semantic_rank_papers(query_title, papers, top_n=len(papers))
    
    # 3a. Core ∧ Tech boost
    print("Applying Core ∧ Tech boost…")

    # Compile regexes for core (app) and tech terms
    import re
    core_patterns = [
        re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE)
        for t in app_terms
    ]
    tech_patterns = [
        re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE)
        for t in tech_terms
    ]

    # Apply a 25% boost only when a paper matches BOTH a core and a tech term
    for p in ranked_all:
        text = f"{p['title']} {p['abstract']}".lower()
        if any(cp.search(text) for cp in core_patterns) and any(tp.search(text) for tp in tech_patterns):
            p["score"] *= 1.25
            continue
        elif any(cp.search(text) for cp in core_patterns) and any(tp.search(text) for tp in tech_patterns):
            #if either, 10% boost
            p["score"] *= 1.10

    # Re-sort after boosting
    ranked_all.sort(key=lambda x: x["score"], reverse=True)


    # helper closures for readability
    def matches_app(p):
        txt = (p["title"] + " " + p["abstract"]).lower()
        return any(pat.search(txt) for pat in app_patterns)

    def matches_tech(p):
        txt = (p["title"] + " " + p["abstract"]).lower()
        return any(pat.search(txt) for pat in tech_patterns)

    def domain_hits(p):
        txt = (p["title"] + " " + p["abstract"]).lower()
        return sum(bool(dp.search(txt)) for dp in domain_patterns)

    print("Categorizing top hits to focused and exploratory domains…")

    DESIRED_FOCUSED = 20

    # ─── FOCUSED ─────────────────────────────────────────────────────────────
    focused = []

    # Tier 1: must hit at least one app, one tech, AND one domain term
    for p in ranked_all:
        if matches_app(p) and matches_tech(p) and domain_hits(p) >= 1:
            focused.append(p)
        if len(focused) >= DESIRED_FOCUSED:
            break

    # Tier 2 (fallback 1): if still short, relax to app AND tech only
    if len(focused) < DESIRED_FOCUSED:
        for p in ranked_all:
            if p in focused: 
                continue
            if matches_app(p) and matches_tech(p):
                focused.append(p)
            if len(focused) >= DESIRED_FOCUSED:
                break

    # Tier 3 (fallback 2): if still short, require domain AND (app OR tech)
    if len(focused) < DESIRED_FOCUSED:
        for p in ranked_all:
            if p in focused: 
                continue
            if domain_hits(p) >= 1 and (matches_app(p) or matches_tech(p)):
                focused.append(p)
            if len(focused) >= DESIRED_FOCUSED:
                break

    focused_top20 = focused[:DESIRED_FOCUSED]
    focused_dois   = {p["doi"] for p in focused_top20}

    # ─── EXPLORATORY ─────────────────────────────────────────────────────────
    noncore = [p for p in ranked_all if p["doi"] not in focused_dois]
    exploratory = []

    # Tier 1: require either ≥2 domain hits OR (≥1 domain AND ≥1 tech)
    for p in noncore:
        if domain_hits(p) >= 2 or (domain_hits(p) >= 1 and matches_tech(p)):
            exploratory.append(p)
        if len(exploratory) >= 10:
            break

    # Tier 2 (fallback): if still short, require ≥1 domain hit
    if len(exploratory) < 10:
        needed = 10 - len(exploratory)
        for p in noncore:
            if p in exploratory: 
                continue
            if domain_hits(p) >= 1:
                exploratory.append(p)
            if len(exploratory) >= 10:
                break

    exploratory_top10 = exploratory[:10]



    # 7. Print the two blocks
    print(f"\n🏆 Focused Top {len(focused_top20)} (application & domain match):")
    for i, p in enumerate(focused_top20, 1):
        print(f"{i}. {p['title']} ({p['year']}) — score {p['score']:.4f} — DOI: {p['doi']}")

    print(f"\n🔍 Exploratory Top {len(exploratory_top10)} (domain-only match):")
    for i, p in enumerate(exploratory_top10, 1):
        print(f"{i}. {p['title']} ({p['year']}) — score {p['score']:.4f} — DOI: {p['doi']}")

    # Log elapsed time up to ranking completion
    elapsed = time.time() - start_time
    print(f"\n⏱️  Pipeline elapsed time (inputs → ranking): {elapsed:.2f} seconds\n")

if __name__ == "__main__":
    main()
