# Viva Voce & Technical Defense Guide
## AI-Based Academic Journal Recommendation Assistant

This document provides in-depth questions and answers designed for academic evaluation, viva voce defense, and technical project presentations.

---

### Category 1: Architecture & Multi-Agent Orchestration (LangGraph)

#### Q1: What is LangGraph and why did you choose it over a standard linear LangChain SequentialChain?
**Answer**:
LangGraph is a stateful orchestration framework designed for complex, non-linear multi-agent workflows modeled as Directed Acyclic Graphs (DAGs) or cyclic graphs with state loops. A linear chain executes strictly step-by-step, whereas our journal recommendation system requires **parallel fan-out** (evaluating candidate journals across 6 specialist dimensions simultaneously: Scope, Similarity, Credibility, Cost, Impact, and Turnaround) followed by a **synchronized fan-in** to a Conflict Resolution node. LangGraph manages shared `JournalState`, handles branching, and guarantees all parallel agent results are collated before ranking.

#### Q2: How does state management work within your LangGraph implementation?
**Answer**:
We define a typed dictionary `JournalState` in `src/journal_ai/orchestration/state.py`. Each node in the graph receives the current state and returns a dictionary with updated or appended keys (e.g., `paper_profile`, `candidate_journals`, `scope_results`, `similarity_results`, `conflicts`, `recommendations`). LangGraph automatically merges these updates into the shared state object.

#### Q3: What happens in the Conflict Resolution node?
**Answer**:
Academic journal selection inherently involves conflicting objectives (e.g., high prestige vs. high Article Processing Charge; top-tier rigor vs. slow review turnaround). The Conflict Resolution node inspects the score distribution across all agents for each journal. If the spread exceeds a defined threshold ($\ge 25$ points) or matches specific trade-off signatures, it logs an explicit trade-off justification showing how the user's custom weights arbitrated the ranking.

---

### Category 2: Real-Time APIs & Academic Data Ingestion

#### Q4: What real-time APIs are used in this project, and what is the role of each?
**Answer**:
1. **OpenAlex API**: An open scholarly bibliographic graph providing comprehensive journal metadata, topic taxonomies, 2-year mean citedness, h-index, and host publisher information.
2. **Crossref REST API**: Resolves journal ISSNs to fetch the most recent real published articles (with DOIs, author lists, and timestamps) for semantic ground truth.
3. **DOAJ (Directory of Open Access Journals) API**: Verifies Open Access compliance, peer-review transparency, DOAJ Seal accreditation, and exact Article Processing Charges (APC).

#### Q5: How does your system handle API rate limits, timeouts, or network outages?
**Answer**:
We implemented an `AcademicDataAggregator` with resilience mechanisms:
- Timeouts are enforced via `httpx.AsyncClient(timeout=...)`.
- Calls to Crossref and DOAJ execute concurrently using `asyncio.gather(*tasks, return_exceptions=True)`.
- If OpenAlex returns fewer candidates due to network filters or rate limits, the aggregator blends in pre-indexed, verified benchmark journals (`FALLBACK_JOURNAL_DATABASE`) so the application never crashes.

#### Q6: Are paid API keys mandatory to run this system?
**Answer**:
**No**. OpenAlex, Crossref, and DOAJ are 100% free and open public academic APIs. By supplying an email address in `.env`, the system enters the OpenAlex and Crossref "polite pool" for higher throughput. Optional LLM keys (Gemini or OpenAI) can be added for enhanced generative synthesis, but the system functions with full fidelity out-of-the-box using built-in heuristic explainability.

---

### Category 3: RAG (Retrieval-Augmented Generation) & ChromaDB

#### Q7: How is RAG used in this project?
**Answer**:
Instead of relying on static keyword matching, we dynamically convert the retrieved candidate journals' aims, scopes, and recent publication titles/abstracts into vector chunks. These chunks are stored in **ChromaDB**. When a user submits a manuscript, we embed the manuscript's title, abstract, and core keywords to perform a semantic similarity search across the vector store. This retrieves actual recent publications and scope chunks that match the manuscript's methodological and conceptual focus.

#### Q8: What embedding model and distance metric does ChromaDB use?
**Answer**:
By default, ChromaDB uses the `DefaultEmbeddingFunction` (based on an ONNX-optimized `all-MiniLM-L6-v2` transformer model). It computes vector distances using L2/cosine distance. We normalize this distance into a human-interpretable $0–100\%$ semantic similarity score using the formula:
$$S_{sim} = \frac{100}{1 + \text{distance}}$$

#### Q9: How does RAG prevent hallucinations in academic recommendations?
**Answer**:
Generative models without RAG often hallucinate journal names, impact factors, or fake recent articles. In our system, the RAG pipeline indexes verified real-time metadata directly from OpenAlex, Crossref, and DOAJ. The Explainability Agent only cites recent publications that have verified DOIs in the vector database.

---

### Category 4: Decision Analysis & Ranking (MCDA)

#### Q10: What ranking algorithm is utilized to sort recommendations?
**Answer**:
We implement **Multi-Criteria Decision Analysis (MCDA)** using the **Weighted Sum Model (WSM)**. The user sets preferences for 6 dimensions ($\text{Scope}, \text{Similarity}, \text{Credibility}, \text{Cost}, \text{Impact}, \text{Turnaround}$). Weights are normalized such that $\sum \bar{w}_i = 1.0$. The composite score is the dot product between the normalized weight vector and the agent score vector:
$$R(j) = \sum_{i=1}^{6} \bar{w}_i \cdot S_i(j)$$

#### Q11: How do the preset buttons (e.g., "⚡ Fast Review" vs "💎 Zero APC") alter the behavior?
**Answer**:
Preset buttons dynamically reweight the decision vector:
- **⚡ Fast Review**: Boosts Turnaround Speed ($w_{speed}=0.30$) and Scope ($w_{scope}=0.20$), prioritizing quick decision cycles for student graduation or urgent submissions.
- **🏆 High Impact**: Boosts Impact ($w_{impact}=0.30$) and Credibility ($w_{cred}=0.20$).
- **💎 Zero APC**: Heavily weights Cost ($w_{cost}=0.35$), pushing Diamond Open Access venues ($0 fee) to the top of the leaderboard.
- **⚖️ Balanced**: Distributes weights evenly across all dimensions.

---

### Category 5: Frontend, Deployment & Security

#### Q12: How is Gradio integrated with FastAPI?
**Answer**:
In `src/journal_ai/main.py`, we construct the Gradio Blocks application via `create_gradio_app()` and mount it directly onto the FastAPI ASGI server using `gr.mount_gradio_app(app, gradio_app, path="/gradio")`. Root requests (`/`) redirect to `/gradio`, allowing users to interact with either the visual UI or call the underlying REST API endpoints (`/api/recommend`, `/api/recommend/upload`).

#### Q13: How can this application be deployed?
**Answer**:
1. **Localhost**: Run `python app.py` or `uv run python app.py` (opens `http://127.0.0.1:7860`).
2. **Hugging Face Spaces**: Push repository directly to HF Spaces with `app.py` as entrypoint.
3. **Docker**: Build and run via `docker build -t journal-ai . && docker run -p 7860:7860 journal-ai`.

#### Q14: How does the PDF parser handle uploaded manuscripts?
**Answer**:
The `pypdf` library extracts text streams from all pages, removes non-printable characters, normalizes whitespace, and parses section boundaries (Abstract, Introduction, Keywords, Methodology) to construct a structured `paper_profile`.

---

### Category 6: Comprehensive Viva Questions Summary

| Question Topic | Key Defense Keyword |
| :--- | :--- |
| **Workflow Engine** | LangGraph StateGraph, Fan-out/Fan-in, DAG |
| **Vector DB** | ChromaDB, persistent collection, L2/cosine distance |
| **Open Access Verification** | DOAJ API, APC USD pricing, Diamond OA ($0 fee) |
| **Citation Velocity** | 2-year mean citedness, h-index, Q1-Q4 quartile proxy |
| **Explainable AI (XAI)** | Decision cards, Strengths, Trade-offs, Tailoring advice |
| **Predatory Detection** | ISSN-L audit, publisher reputation, DOAJ Seal check |
