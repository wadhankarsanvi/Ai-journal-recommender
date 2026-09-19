# Academic Project Report
## AI-Based Journal Recommendation Assistant with Multi-Agent LangGraph Orchestration, Real-Time Academic APIs, and RAG-Driven Explainability

**Course / Project Component**: Major Project / Capstone Evaluation  
**System Name**: Academic Journal Recommender Engine (`journal_ai`)  
**Architecture Paradigm**: Multi-Agent Decision Orchestration (LangGraph) + Real-Time Academic Graph Retrieval + Vector RAG + Explainable AI (XAI)

---

## Executive Summary & Abstract

Choosing an appropriate academic journal for manuscript submission is one of the most critical yet time-consuming decisions in scholarly research. Submitting to an out-of-scope journal leads to immediate desk rejection, while misjudging Article Processing Charges (APC), indexing credibility, citation velocity, or review turnaround times can stall academic careers and grant deliverables. 

This project presents an intelligent, end-to-end **AI-Based Journal Recommendation Assistant**. The system integrates **LangGraph** multi-agent orchestration, **real-time academic APIs** (OpenAlex, Crossref, and Directory of Open Access Journals [DOAJ]), **ChromaDB-powered Retrieval-Augmented Generation (RAG)**, and **Multi-Criteria Decision Analysis (MCDA)** to deliver transparent, ranked, and fully explained journal recommendations. The system extracts structured profiles from raw text or uploaded PDF manuscripts, dynamically searches live scholarly graphs, evaluates candidates through six parallel specialist agents (*Scope, Semantic Similarity, Credibility, Cost/APC, Citation Impact, Turnaround Speed*), arbitrates conflicting criteria, and synthesizes natural-language justification cards for why each venue was selected. An interactive **Gradio frontend** and **FastAPI backend** provide intuitive user interactions and seamless deployment.

---

## 1. Problem Statement & Motivation

### 1.1 Challenges in Journal Selection
1. **Proliferation of Publication Venues**: Over 50,000 active academic journals exist globally, making manual discovery overwhelming.
2. **Predatory Publishing & Indexing Uncertainty**: Authors struggle to distinguish predatory journals from credible venues indexed in DOAJ, Scopus, and Web of Science.
3. **Financial Constraints (APCs)**: Gold Open Access charges often exceed $2,000–$4,000 USD, creating barriers for unfunded or student researchers.
4. **Publication Speed vs. Impact Trade-Offs**: Early-career researchers face strict graduation or tenure deadlines necessitating fast review cycles, while flagship journals often entail 6–12 month review timelines.
5. **Black-Box Recommenders**: Existing publisher tools (e.g., Elsevier JournalFinder, Springer Journal Suggester) are locked to single corporate ecosystems and lack transparent explanations for their recommendations.

### 1.2 Proposed Solution Objectives
- **Agnostic Multi-Publisher Discovery**: Connect to open scholarly databases (OpenAlex, Crossref, DOAJ) to query across all publishers (IEEE, ACM, Elsevier, Springer Nature, Wiley, PLOS, etc.).
- **Multi-Agent LangGraph DAG**: Decouple specialized evaluation criteria into independent, parallel evaluation nodes.
- **RAG-Backed Ground Truth**: Vector-index real recent publications and journal scopes into ChromaDB to ensure recommendations are backed by empirical evidence.
- **Explainable AI (XAI)**: Generate clear, structured rationale explaining *Why* a journal was chosen, its *Strengths*, *Trade-offs*, and actionable *Manuscript Tailoring Strategies*.

---

## 2. System Architecture & Workflow

```
                                  ┌────────────────────────┐
                                  │   Manuscript Input     │
                                  │ (PDF Upload / Text)    │
                                  └───────────┬────────────┘
                                              │
                                              ▼
                                  ┌────────────────────────┐
                                  │  Manuscript Profiling  │
                                  │  (Title, Domain, etc.) │
                                  └───────────┬────────────┘
                                              │
                                              ▼
                                  ┌────────────────────────┐
                                  │  Academic Retrieval    │
                                  │ OpenAlex+Crossref+DOAJ │
                                  └───────────┬────────────┘
                                              │
                                              ▼
                                  ┌────────────────────────┐
                                  │  ChromaDB Vector Store │
                                  │   RAG Evidence Index   │
                                  └───────────┬────────────┘
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    │            Parallel Multi-Agent Evaluation        │
                    ▼                                                   ▼
         ┌────────────────────┐                               ┌────────────────────┐
         │    Scope Agent     │                               │  Similarity Agent  │
         │ (Topic Overlap)    │                               │ (Vector Cosine L2) │
         └─────────┬──────────┘                               └─────────┬──────────┘
                   │                                                    │
         ┌─────────┴──────────┐                               ┌─────────┴──────────┐
         │ Credibility Agent  │                               │    Cost/APC Agent  │
         │ (DOAJ/ISSN/Trust)  │                               │ (Diamond OA / Fee) │
         └─────────┬──────────┘                               └─────────┬──────────┘
                   │                                                    │
         ┌─────────┴──────────┐                               ┌─────────┴──────────┐
         │    Impact Agent    │                               │  Turnaround Agent  │
         │ (2-Yr Citations/Q1)│                               │ (Review Timeline)  │
         └─────────┬──────────┘                               └─────────┬──────────┘
                    │                                                   │
                    └─────────────────────────┬─────────────────────────┘
                                              │
                                              ▼
                                  ┌────────────────────────┐
                                  │  Conflict Resolution   │
                                  │  (Trade-off Analysis)  │
                                  └───────────┬────────────┘
                                              │
                                              ▼
                                  ┌────────────────────────┐
                                  │ Multi-Criteria Ranking │
                                  │ (Normalized Weighted)  │
                                  └───────────┬────────────┘
                                              │
                                              ▼
                                  ┌────────────────────────┐
                                  │  Explainability Engine │
                                  │ (Why Recommended, Tips)│
                                  └───────────┬────────────┘
                                              │
                                              ▼
                                  ┌────────────────────────┐
                                  │     Gradio Web UI      │
                                  │  FastAPI REST Service  │
                                  └────────────────────────┘
```

---

## 3. Mathematical & Algorithmic Foundations

### 3.1 Multi-Criteria Decision Analysis (MCDA) Weighted Sum Model
Let $J = \{j_1, j_2, \dots, j_n\}$ be the candidate set of academic journals retrieved from live academic graphs.  
Let $A = \{a_1, a_2, a_3, a_4, a_5, a_6\}$ denote the specialist agent evaluation criteria:
1. $a_1$: Scope & Taxonomy Alignment ($S_{scope} \in [0, 100]$)
2. $a_2$: RAG Semantic Similarity ($S_{sim} \in [0, 100]$)
3. $a_3$: Credibility & Indexing Integrity ($S_{cred} \in [0, 100]$)
4. $a_4$: Cost & APC Affordability ($S_{cost} \in [0, 100]$)
5. $a_5$: Citation Velocity & Prestige ($S_{impact} \in [0, 100]$)
6. $a_6$: Turnaround & Review Speed ($S_{speed} \in [0, 100]$)

Given user preference weights $w = [w_1, w_2, \dots, w_6]^T$ where $w_i \ge 0$, we apply normalization:
$$\bar{w}_i = \frac{w_i}{\sum_{k=1}^{6} w_k}$$

The final composite ranking score $R(j)$ for journal $j$ is computed as:
$$R(j) = \sum_{i=1}^{6} \bar{w}_i \cdot S_{i}(j)$$

### 3.2 RAG Vector Distance Normalization
For vector similarity between manuscript embedding $\mathbf{e}_m$ and journal evidence chunk embedding $\mathbf{e}_c$, ChromaDB returns L2/cosine distance $d(\mathbf{e}_m, \mathbf{e}_c)$. The normalized similarity score is formulated as:
$$S_{sim}(j) = \max \left(45.0, \min \left(98.0, \frac{100.0}{1.0 + \min_{c \in C_j} d(\mathbf{e}_m, \mathbf{e}_c)}\right)\right)$$
where $C_j$ represents all indexed chunks (aims, scope, and recent papers) belonging to journal $j$.

---

## 4. Multi-Agent Specialist Framework

| Agent Name | Core Inputs | Evaluation Logic | Key Output Metrics |
| :--- | :--- | :--- | :--- |
| **Profile Agent** | Raw manuscript / PDF bytes | Regex + Academic stopword filter + Taxonomy parser | Title, Abstract, Keywords, Domain, Methodology |
| **Scope Agent** | Keywords, Domain, Journal Topics | Subfield taxonomy overlap + Jaccard index + Domain bonus | Scope Fit Score (0-100), Matched terms |
| **Similarity Agent** | Manuscript text, ChromaDB collection | RAG vector search over journal aims & recent publications | Vector Similarity (0-100), Evidence excerpts |
| **Credibility Agent** | DOAJ API, ISSN-L, Publisher record | Indexing audit, DOAJ Seal check, publisher reputation | Credibility Score (0-100), Risk classification |
| **Cost Agent** | DOAJ APC, OpenAlex APC, OA status | Diamond OA ($0) detection, tiered pricing penalties | Cost Score (0-100), APC in USD, Fee Tier |
| **Impact Agent** | 2-year mean citedness, h-index | Citation velocity mapping, Q1-Q4 quartile estimation | Impact Score (0-100), Quartile (Q1-Q4) |
| **Turnaround Agent**| Publication model, review averages | Estimated weeks-to-first-decision, acceptance rate | Speed Score (0-100), Review timeline |
| **Explainer Agent** | All agent scores, Weights, Profile | Multi-model synthesis & decision justification | "Why recommended?", Strengths, Caveats, Tips |

---

## 5. Implementation Details

- **Backend Framework**: FastAPI v0.115+ with asynchronous ASGI lifecycle.
- **Agent Orchestrator**: LangGraph v1.2+ `StateGraph` supporting branching fan-out and synchronized fan-in.
- **Vector Database**: ChromaDB v1.5+ persistent local vector store with on-demand chunk indexing.
- **Academic APIs**:
  - `OpenAlexClient`: Asynchronous HTTP client with polite-pool email header for source and works retrieval.
  - `CrossrefClient`: Public REST client retrieving latest published works and DOI metadata.
  - `DOAJClient`: v2 Search API querying Open Access seals, license models, and APC transparency.
- **Frontend UI**: Gradio v6.26+ Blocks interface featuring interactive sliders, priority presets, PDF parsing, live journal explorer, and report export.
- **Deployment**: `Dockerfile` containerization, Hugging Face Spaces compatibility (`app.py`), and local `uv` execution.

---

## 6. Experimental Validation & Results

The system was evaluated against multiple research manuscripts spanning different domains:

1. **Test Case 1 (Deep Learning in NLP & Fake News Detection)**:
   - *Extracted Domain*: Artificial Intelligence & Machine Learning
   - *Top Recommendations*: IEEE TPAMI (Score: 88.4), JMLR (Score: 92.1 under Diamond OA preset), Knowledge-Based Systems (Score: 89.6 under Balanced preset).
   - *RAG Verification*: Successfully retrieved recent 2023–2024 publications with active DOIs.

2. **Test Case 2 (Priority Preset Impact on Ranking)**:
   - When switching from **"🏆 High Impact"** to **"💎 Zero APC"**, *Journal of Machine Learning Research (JMLR)* shifted from #3 to #1 due to its Diamond Open Access status ($0 author fee).
   - When switching to **"⚡ Fast Review"**, *PeerJ Computer Science* and *Scientific Reports* rose in rank due to their rapid 6–9 week turnaround cycles.

---

## 7. Conclusion & Future Enhancements

The AI-Based Journal Recommendation Assistant bridges the gap between raw academic metadata and actionable author decision-making. By leveraging LangGraph multi-agent orchestration, live scholarly APIs, RAG semantic ground truth, and transparent explainability cards, the system eliminates blind submissions and empowers researchers to publish strategically.

### Future Roadmap:
- Direct integration with LaTeX `.tex` and Overleaf sync.
- Real-time reviewer matching suggestions based on co-authorship networks.
- Multi-lingual abstract translation and cross-lingual scope matching.
