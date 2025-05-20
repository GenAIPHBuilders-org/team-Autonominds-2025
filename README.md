# Authematic: Agentic Literature Curation Pipeline

Authematic is a lightweight, agent-driven Python pipeline that takes a research paper title as input and produces two curated lists of the most relevant academic papers:

1. **Focused Top 20**: Papers that match both your application domain and your chosen technique keywords.  
2. **Exploratory Top 10**: Broader domain-only matches that add methodological or interdisciplinary insights.

**Features to be implemented:** 

1. Increase API sources of papers
2. Summary of the top papers + relevance to the study
3. Highlight literature Gap
4. Auto citation based on citation style inputted

Under the hood, it uses LLM calls (Gemini), arXiv & Semantic Scholar scraping, SciBERT embeddings, and a small amount of TF-IDF + heuristic boosting to maximize both precision and coverage.

---

## 🚀 High-Level Pipeline

1. **User Input**  
   - Title of your research topic  
   - Publication year cutoff (e.g. 2018)  
   - (Future) Citation style  

2. **Topic & Keyword Generation** (`paper_collector.py`)  
   - **4 Related-Work Topics** via Gemini: 2 Core / 1 Adjacent / 1 Emerging  
   - **5–10 Search Phrases per Topic** via Gemini  
   - **Domain Filter Terms** via Gemini  

3. **Paper Collection** (`paper_collector.py`)  
   - For each search phrase:  
     - Retrieve up to 4 papers from **arXiv**  
     - Retrieve up to 4 papers from **Semantic Scholar**  
     - Enforce year cutoff as you fetch  

4. **Raw Dump & Reload**  
   - Save all raw candidates to `raw_candidates.json`  
   - Reload for deterministic filtering & ranking  

5. **Basic Filters** (`filter_and_rank.py`)  
   - **DOI filter**: remove entries with missing/empty DOI  
   - **Abstract filter**: remove entries without an abstract  
   - **Deduplication** by DOI  

6. **Term Buckets Extraction** (`run_pipeline.py`)  
   - **App terms**: multi-word phrases + exact full-title anchor  
   - **Tech terms**: cleaned LLM outputs, guaranteed non-empty by fallback  

7. **Semantic Ranking**  
   - Embed every paper & your title with **SciBERT** (`embeddings.py`)  
   - Rank by cosine similarity  

8. **Heuristic Boosting**  
   - Infer which domain terms are most over-represented in the Top 20 vs rest  
   - Boost scores on any paper containing those terms  

9. **Final Split**  
   - **Focused**: papers matching **both** an app-term regex & a tech-term regex (plus strict fallback if <20)  
   - **Exploratory**: next-best papers matching **any** domain-term regex  

10. **Console Output**  
    - Prints the two lists with title, year, score, and DOI  

---

## 📂 File Breakdown

### 1. `paper_collector.py`

- **LLM prompts** (Gemini) for:  
  - `generate_topics(title)` → 4 Related-Work labels  
  - `generate_keywords(topics)` → 5–10 search phrases each  
  - `generate_domain_terms(title)` → up to 10 domain filter keywords  
  - `generate_app_terms(title)` → 5 application phrases  
  - `generate_tech_terms(title)` → 5 technique phrases  
- **Fetch functions**:  
  - `search_arxiv(keyword, start_index)` via arXiv API + RSS  
  - `search_semantic_scholar(keyword, offset)` via Semantic Scholar REST  
- **Collect orchestration**:  
  - `collect_papers(keywords_by_topic, cutoff_year)`  

---

### 2. `filter_and_rank.py`

- **Load & validate** raw JSON (`load_candidates_from_json`)  
- **Filters**:  
  - `filter_by_doi`  
  - `filter_by_abstract`  
  - `dedupe_by_doi`  
- **Ranking**:  
  - `semantic_rank_papers(query, papers)` using SciBERT embeddings  
  - (TF-IDF + bigram rank available via `rank_papers`)  
- **Boost term inference**:  
  - `infer_boost_terms(ranked_papers, domain_terms)`  

---

### 3. `embeddings.py`

- Wraps HuggingFace’s **`allenai/scibert_scivocab_uncased`** to produce 768-dim embeddings  
- Caches in-memory to speed up repeated calls  

---

### 4. `run_pipeline.py`

- **CLI / prompt** for title, cutoff year, citation style  
- Invokes all of the above in sequence  
- Builds & cleans **app_terms** + **tech_terms**  
- Runs semantic ranking, boosting, focused/exploratory split  
- Prints final Top 20 & Top 10  

---

## 🔧 Installation & Setup

1. **Clone the repo**  
   ```bash
   git clone [https://github.com/your-org/authematic.git](https://github.com/GenAIPHBuilders-org/team-Autonominds-2025)
   cd authematic
2. **Create & activate** a Python 3.10+ virtual environment
  python -m venv venv
  source venv/bin/activate     # Linux / macOS
  venv\Scripts\activate.bat    # Windows

3. **Install dependencies**
   pip install \
  python-dotenv \
  google-genai \
  requests \
  feedparser \
  transformers \
  torch \
  numpy \
  scikit-learn \
  joblib

4. **Run the pipeline**
python run_pipeline.py 
