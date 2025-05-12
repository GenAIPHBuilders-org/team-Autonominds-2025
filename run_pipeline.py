import json
import sys
from pathlib import Path
from typing import List

# Import your modules
from paper_collector import generate_topics, generate_keywords, collect_papers
from filter_and_rank import (
    load_candidates_from_json,
    filter_by_doi,
    filter_by_abstract,
    dedupe_by_doi,
    semantic_rank_papers,
    infer_boost_terms
    # future imports: filter_by_year, dedupe_by_doi, rank_papers
)

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
        
    # 1a. Generate domain terms via Gemini
    from paper_collector import generate_domain_terms
    domain_terms = generate_domain_terms(title, max_terms=10)
    print(f"🔑 Domain terms: {domain_terms}")

    # 1.2. Cutoff year
    cutoff_year = prompt_cutoff_year()

    # 1.3. Citation style (for future formatting)
    citation_style = input("Enter citation style (e.g. APA): ").strip() or "APA"
    
    # 2. Collect & dump raw JSON
    related_topics = generate_topics(title)
    keywords_by_topic = generate_keywords(related_topics)
    if len(related_topics) != 4:
        print(f"⚠️ Warning: expected 4 topics, got {len(related_topics)}")
    papers = collect_papers(keywords_by_topic, cutoff_year)

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

    # 7a. Two-phase LLM extraction
    app_raw  = generate_app_terms(title, max_terms=5)
    tech_raw = generate_tech_terms(title, max_terms=5)

    # 1) Cleaned APP terms: only multi-word phrases
    app_terms = clean_terms(app_raw)
    app_terms = [t for t in app_terms if len(t.split()) > 1]

    # 1b) Anchor to the exact title phrase
    full_phrase = title.lower().strip()
    if full_phrase not in app_terms:
        app_terms.insert(0, full_phrase)

    # 2) Cleaned TECH terms
    tech_terms = clean_terms(tech_raw)

    # 2b) If all pruning removed them, fall back to LLM’s own top suggestions
    if not tech_terms:
        tech_terms = tech_raw[:3]  # take the first 3 phrases the LLM gave us

    print(f"🔑 Final application terms: {app_terms}")
    print(f"🔑 Final technique terms:   {tech_terms}")

    print(f"🔑 Final application terms: {app_terms}")
    print(f"🔑 Final technique terms:   {tech_terms}")
    
    # If either list is empty, fallback to original raw lists
    if not app_terms:
        print("⚠️ App terms empty; using raw app extraction")
        app_terms = generate_app_terms(title, max_terms=5)
    if not tech_terms:
        print("⚠️ Tech terms empty; using raw tech extraction")
        tech_terms = generate_tech_terms(title, max_terms=5)
    print(f"🔑 Application core terms: {app_terms}")
    print(f"🔑 Technique core terms:   {tech_terms}")

    # 7c. Compile regex patterns
    import re
    app_patterns  = [re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in app_terms]
    tech_patterns = [re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in tech_terms]
    domain_patterns = [re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in domain_terms]

    # 3. Rank all papers semantically
    print("Using SciBERT to rank the collected papers...")
    ranked_all = semantic_rank_papers(query_title, papers, top_n=len(papers))
    # 3a. determine boost terms
    print("Determining which terms to boost...")
    
    # 1. Infer which terms most distinguish the initial Top-K
    inferred_terms = infer_boost_terms(
        ranked_all, domain_terms, top_k=20, multiplier_threshold=1.5
    )
    print(f"🔍 Inferred lift terms: {inferred_terms}")

    # 2. First try to boost on technique phrases
    boost_terms = [t for t in inferred_terms if t in tech_terms]

    # 3. If that’s empty, fall back to the LLM’s own technique suggestions
    if not boost_terms:
        print("⚡ No inferred tech_terms; falling back to LLM's top tech_raw")
        boost_terms = tech_raw[:3]

    # 4. If still empty, use the top 3 inferred lift terms
    if not boost_terms:
        print("⚡ Still no boost_terms; using top inferred lift terms")
        boost_terms = inferred_terms[:3]

    print(f"⚡ Final boost terms: {boost_terms}")

    # 5. Compile regexes for those boost terms
    import re
    boost_patterns = [
        re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE)
        for t in boost_terms
    ]

    # 6. Apply the boost multiplier
    for p in ranked_all:
        txt = f"{p['title']} {p['abstract']}".lower()
        if any(bp.search(txt) for bp in boost_patterns):
            p["score"] *= 1.10

    # 7. Re-sort by updated scores
    ranked_all.sort(key=lambda x: x["score"], reverse=True)

    # 4. Dual-match Focused filter: must match both core and domain
    print("Categorizing top hits to focused and exploratory domains...")
    focused = []
    for p in ranked_all:
        text = f"{p['title']} {p['abstract']}".lower()
        if any(pat.search(text) for pat in app_patterns) and any(pat.search(text) for pat in tech_patterns):
            focused.append(p)

    # 5. Fallback if too few focused hits
    DESIRED_FOCUSED = 20
    if len(focused) < DESIRED_FOCUSED:
        print(f"⚠️ Only {len(focused)} app+tech matches; filling with app+tech fallback")
        seen = {p["doi"] for p in focused}
        needed = DESIRED_FOCUSED - len(focused)
        fallback = []
        for p in ranked_all:
            if p["doi"] in seen:
                continue
            txt = (p["title"] + " " + p["abstract"]).lower()
            # require BOTH gates
            if any(ap.search(txt) for ap in app_patterns) and any(tp.search(txt) for tp in tech_patterns):
                fallback.append(p)
            if len(fallback) == needed:
                break
        focused += fallback

    focused_top20 = focused[:DESIRED_FOCUSED]
    focused_dois   = {p["doi"] for p in focused_top20}


    # 6. Exploratory: papers not in Focused but matching any domain term
    noncore = [p for p in ranked_all if p["doi"] not in focused_dois]
    exploratory = [
        p for p in noncore
        if any(dp.search(f"{p['title']} {p['abstract']}") for dp in domain_patterns)
    ]
    exploratory_top10 = exploratory[:10]

    # 7. Print the two blocks
    print(f"\n🏆 Focused Top {len(focused_top20)} (application & domain match):")
    for i, p in enumerate(focused_top20, 1):
        print(f"{i}. {p['title']} ({p['year']}) — score {p['score']:.4f} — DOI: {p['doi']}")

    print(f"\n🔍 Exploratory Top {len(exploratory_top10)} (domain-only match):")
    for i, p in enumerate(exploratory_top10, 1):
        print(f"{i}. {p['title']} ({p['year']}) — score {p['score']:.4f} — DOI: {p['doi']}")

    # print(f"\n🏆 Top {top_n} papers by relevance:")
    # for idx, p in enumerate(ranked, start=1):
    #     score = p.get("score", 0.0)
    #     print(f"{idx}. {p['title']} ({p['year']}) — score: {score:.4f} — DOI: {p['doi']}")

if __name__ == "__main__":
    main()