# Authematic: Agentic Literature Curation Pipeline

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

10. **Web Interface or Console Output**  
    - Displays the two lists with title, authors, year, score, and DOI  

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
