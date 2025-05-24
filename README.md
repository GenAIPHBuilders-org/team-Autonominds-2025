# Authematic: (Automatic + Thematic)

Authematic is a lightweight, agent-driven Python pipeline that takes a research paper title as input and produces two curated lists of the most relevant academic papers:

1. **Focused Top 20**: Papers that match both your application domain and your chosen technique keywords.  
2. **Exploratory Top 10**: Broader domain-only matches that add methodological or interdisciplinary insights.

**Features:** 

1. Interactive web interface with chat-based interactions
2. Multi-source academic paper retrieval (arXiv, Semantic Scholar)
3. Semantic ranking using SciBERT embeddings
4. Intelligent filtering and categorization
5. Customizable year cutoff for recency control

**Features to be implemented:** 

1. Additional API sources for papers
2. Summary of top papers + relevance to the study
3. Literature gap highlighting
4. Automatic citation generation based on preferred citation style

Under the hood, it uses LLM calls (Gemini), arXiv & Semantic Scholar scraping, SciBERT embeddings, and a small amount of TF-IDF + heuristic boosting to maximize both precision and coverage.

---

## ⛓️ Pipeline Flow

Outlined below is the step-by-step execution flow of the AI-powered literature curation pipeline. The pipeline starts with a research title and ends with a ranked and categorized list of academic papers.

1.  **Initialization & User Input**:
    * Environment variables (including API keys) are loaded from a `.env` file.
    * API clients are initialized and managed via `api_client_manager.py`, setting up a round-robin system for using available Gemini API keys.
    * The `run_pipeline.py` script starts by prompting the user for the **research title**, a **publication year cutoff**, and the desired **citation style**.

2.  **Topic & Subtheme Generation**:
    * Using the input title, `paper_collector.py` calls the Gemini API (via `api_client_manager.py`) to generate **4 high-level academic topics** (Core, Adjacent, Emerging).
    * For each of these topics, it again calls the Gemini API to generate up to **3 sub-themes** or research niches.

3.  **Initial Keyword Generation**:
    * For each "Topic ▶ Subtheme" pair, `paper_collector.py` calls the Gemini API to generate up to **5 specific search keywords**.
    * The results are structured as a nested dictionary and saved temporarily to `keywords_by_subtheme.json`.

4.  **Keyword Critique & Refinement**:
    * The `keyword_critic.py` module is invoked for each sub-theme's keyword list.
    * It sends the list to the Gemini API, asking it to act as a critic, **removing broad/redundant terms** and **suggesting replacements**.
    * The refined (kept + suggested) keywords are used for the next step.

5.  **Paper Collection & Aggregation**:
    * The `collect_papers` function in `paper_collector.py` orchestrates the fetching.
    * It iterates through each sub-theme and its refined keywords.
    * For each keyword, it concurrently queries multiple literature sources (**arXiv**, **PubMed**, **CrossRef**) using a `ThreadPoolExecutor`. *Semantic Scholar is noted as excluded for initial search but used later for enrichment*.
    * Each source fetches a set number of papers per keyword per attempt/offset.
    * As papers are fetched, they are **filtered by year** and checked for **valid DOI and abstract**. They are also **deduplicated within each source's batch**.
    * Papers are added to their respective sub-theme "bucket" until a minimum target is reached, with global deduplication within the bucket.

6.  **Data Enrichment**:
    * After collection, all papers are pooled together.
    * The `enrich_with_semantic_scholar` function is called to **fetch missing abstracts or author details** for papers using their DOIs.

7.  **Intermediate Save & Load**:
    * `paper_collector.py` generates initial **domain terms** using the Gemini API, which are then critiqued via `keyword_critic.py`.
    * The collected (and enriched) papers, the query title, and the critiqued domain terms are saved to `raw_candidates.json`.
    * This JSON file is immediately reloaded using `load_candidates_from_json` in `filter_and_rank.py`.

8.  **Initial Filtering**:
    * The loaded papers undergo stricter filters using `filter_and_rank.py`:
        * `filter_by_doi`: Ensures a valid DOI exists.
        * `filter_by_abstract`: Ensures a valid abstract exists.
        * `dedupe_by_doi`: Performs a final global deduplication.

9.  **Application & Technique Term Generation**:
    * `paper_collector.py` calls the Gemini API to generate **application-centric terms** and **technique/methodology terms**.
    * These lists are also critiqued using `keyword_critic.py`.
    * The critiqued lists (along with domain terms) are further **cleaned** using internal stop-word lists.

10. **Technique Term Clustering (Optional)**:
    * If enough technique terms exist, their **SciBERT embeddings** are calculated using `embeddings.py`.
    * **K-Means clustering** is applied to group similar techniques.
    * A smaller, **representative set of technique terms** is selected.

11. **Semantic Ranking & Boosting**:
    * All filtered papers are ranked using `semantic_rank_papers` from `filter_and_rank.py`.
    * This involves getting **SciBERT embeddings** for each paper (`embeddings.py`) and calculating **cosine similarity** against the query.
    * Scores are then **boosted** if a paper matches application or technique terms.
    * Papers are re-sorted based on boosted scores.

12. **Categorization**:
    * **Regex patterns** are created for application, technique, and domain terms.
    * Papers are categorized into:
        * **Focused Papers**: Matching a combination of Application, Technique, and Domain terms.
        * **Exploratory Papers**: Matching Domain and/or Technique terms, excluding those already in 'Focused'.
    * The pipeline prints the top **Focused** and **Exploratory** papers (title, year, score, DOI) to the console.
    * The total execution time is reported.
   
13.  **🧠 Insight Generation & Thematic Grouping**:
    * The top-ranked **Focused** and **Exploratory** papers are passed into `identify_themes_and_group` from `theme_classifier.py`.
    * The Gemini API analyzes the abstracts in batch and outputs:
        * A list of research themes, each with:
            * A concise academic title.
            * A one-to-two sentence description.
            * A list of relevant paper titles associated with the theme.
    * The themes and their associated papers are structured into a nested JSON dictionary.
    * Each individual paper is also passed to `generate_insights` from `extract_insights.py`, which returns:
        * A formal academic summary of the abstract.
        * A second-person explanation of how the paper supports or relates to the user’s research title.
    * These insights are embedded into the paper’s data structure under the keys `insight` and `relevance`.

```mermaid
graph TD
    A[Start: Input Research Title, Year, Style] --> B(Generate Topics);
    B --> C(Generate Subthemes);
    C --> D(Generate Keywords per Subtheme);
    D --> E{Keyword Critique};
    E --> F[Collect Papers];

    subgraph Paper Collection
        F --> G1(Query arXiv);
        F --> G2(Query PubMed);
        F --> G3(Query CrossRef);
        G1 --> H(Aggregate & Initial Filter);
        G2 --> H;
        G3 --> H;
    end

    H --> I(Enrich with Semantic Scholar);
    I --> J(Generate/Critique Domain Terms);
    J --> K[Save to raw_candidates.json];
    K --> L[Load from raw_candidates.json];
    L --> M[Apply Stricter Filters: DOI, Abstract, Dedupe];
    M --> N(Generate/Critique App & Tech Terms);
    N --> O(Clean All Terms);
    O --> P{Tech Term Clustering?};
    P -- Yes --> Q(Cluster & Select Tech Terms);
    P -- No --> R(Use Cleaned Tech Terms);
    Q --> S[Prepare Final Term Lists & Patterns];
    R --> S;
    M --> T(Embed & Rank Papers - SciBERT);
    S --> U{Boost Scores based on Terms};
    T --> U;
    U --> V[Re-sort Papers];
    V --> W{Categorize Papers};
    W -- Focused --> X(Select Focused Papers);
    W -- Exploratory --> Y(Select Exploratory Papers);
    X --> Z[Output: Print Focused List];
    Y --> AA[Output: Print Exploratory List];
    Z --> BB[End];
    AA --> BB;

    %% Styling
    classDef io fill:#dceefb,stroke:#2a4365,stroke-width:2px,color:#1a202c;
    classDef process fill:#bee3f8,stroke:#2a4365,stroke-width:2px,color:#1a202c;
    classDef decision fill:#90cdf4,stroke:#2a4365,stroke-width:2px,color:#1a202c;
    classDef storage fill:#b3cde0,stroke:#2a4365,stroke-width:2px,color:#1a202c;
    
    class A,K,L,Z,AA,BB io;
    class B,C,D,F,G1,G2,G3,H,I,J,M,N,O,Q,R,S,T,V,X,Y process;
    class E,P,W,U decision;
```

---

## 📂 Project Structure

```
└── team-Autonominds-2025/
    ├── AUTHEMATIC CHANGELOG.txt
    ├── embeddings.py
    ├── filter_and_rank.py
    ├── keyword_generation.py
    ├── paper_collector.py
    ├── README.md
    ├── run_pipeline.py
    ├── __pycache__/
    │   ├── embeddings.cpython-311.pyc
    │   ├── filter_and_rank.cpython-311.pyc
    │   └── paper_collector.cpython-311.pyc
    ├── web_app/
    │   ├── .env
    │   ├── app.py
    │   ├── requirements.txt
    │   ├── web_collector.py
    │   ├── templates/
    │   │   ├── base.html
    │   │   └── index.html
    │   └── static/
    │       ├── js/
    │       │   └── main.js
    │       └── css/
    │           └── style.css
    └── .git/
        ├── COMMIT_EDITMSG
        ├── config
        ├── description
        ├── HEAD
        ├── index
        ├── packed-refs
        ├── logs/
        │   ├── HEAD
        │   └── refs/
        │       ├── heads/
        │       │   ├── main
        │       │   └── web-interface
        │       └── remotes/
        │           └── origin/
        │               ├── HEAD
        │               └── web-interface
        ├── info/
        │   └── exclude
        ├── hooks/
        │   ├── applypatch-msg.sample
        │   ├── commit-msg.sample
        │   ├── fsmonitor-watchman.sample
        │   ├── post-update.sample
        │   ├── pre-applypatch.sample
        │   ├── pre-commit.sample
        │   ├── pre-merge-commit.sample
        │   ├── pre-push.sample
        │   ├── pre-rebase.sample
        │   ├── pre-receive.sample
        │   ├── prepare-commit-msg.sample
        │   ├── push-to-checkout.sample
        │   └── update.sample
        └── 

```

### Core Modules

#### 1. `paper_collector.py`

- **LLM prompts** (Gemini) for topic and keyword generation
- **Fetch functions** for arXiv and Semantic Scholar
- **Collection orchestration** for multi-source retrieval

#### 2. `filter_and_rank.py`

- **JSON loading and validation**
- **Filtering functions** for DOI, abstract, and deduplication
- **Semantic ranking** using SciBERT embeddings
- **Heuristic boosting** for domain relevance

#### 3. `embeddings.py`

- SciBERT wrapper for generating 768-dim embeddings
- In-memory caching for performance

#### 4. `run_pipeline.py`

- **CLI interface** for pipeline execution
- **Term extraction and cleaning**
- **Result categorization and display**

#### 5. `web_app/app.py`

- **Flask web application** with SocketIO
- **Interactive chat interface**
- **Background processing** for non-blocking operation

---

## 🔧 Installation & Setup

### Core Requirements

1. **Clone the repo**  
   ```bash
   git clone https://github.com/GenAIPHBuilders-org/team-Autonominds-2025.git
   cd team-Autonominds-2025
2. **Create & activate a Python 3.10+ virtual environment**
   ```python
   python -m venv venv
   source venv/bin/activate     # Linux / macOS
   venv\Scripts\activate.bat    # Windows
   ```
   
3. **Install dependencies**
   ```python
   pip install python-dotenv google-genai requests feedparser transformers torch numpy scikit-learn joblib flask flask-socketio
   ```
4. **Set up your Gemini API key
Create a .env file in the project root with your Gemini API key:**
```
GEMINI_API_KEY=your_api_key_here
```

## 🚀 Usage
### Command Line Interface
Run the pipeline directly from the command line:
```python
python run_pipeline.py
```

Follow the prompts to enter:

1. Your research paper title
2. Publication year cutoff
3. (Optional) Citation style

### Web Interface
Start the web application:
```python
cd web_app
python app.py
```

Then open your browser to `http://127.0.0.1:5000/` and:

1. Interact with Authematic through the chat interface
2. Enter your research title by typing "Research title: [your title]"
3. Specify a year cutoff when prompted
4. View your results in the focused and exploratory categories
5. Use the Reset button to start a new search

## 🔍 Troubleshooting
- `API Key Issues`: Ensure your Gemini API key is correctly set in the .env file
- `Missing Directories`: The web app will automatically create required directories
- `Model Downloads`: The first run may take longer as it downloads SciBERT

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Open a pull request
