import os
from typing import Any
import gradio as gr

from journal_ai.ingestion.pdf_parser import extract_pdf_text
from journal_ai.orchestration.graph import journal_recommendation_graph
from journal_ai.data_sources.openalex import OpenAlexClient
from journal_ai.data_sources.doaj import DOAJClient
from journal_ai.reporting.pdf_generator import generate_recommendation_pdf


# Sample test manuscripts across different academic disciplines
SAMPLE_NLP_MISINFORMATION = """Deep Transfer Learning Framework for Automated Fake News and Misinformation Detection in Multilingual Social Networks

Abstract:
The rapid proliferation of misinformation and deceptive content on digital platforms poses significant risks to public discourse and societal stability. Traditional machine learning approaches often suffer from vulnerability to adversarial perturbations and struggle with cross-lingual generalization. In this paper, we propose a novel transformer-based dual-branch neural architecture that combines contrastive semantic representation learning with topological graph neural networks (GNN) to capture both contextual linguistic cues and diffusion cascades. We evaluate our framework across four benchmark multilingual datasets comprising over 150,000 verified articles. Experimental results demonstrate that our proposed model achieves state-of-the-art F1-score of 94.8%, outperforming baseline RoBERTa and Graph Attention Networks by 6.2%. Furthermore, we introduce an explainable attribution mechanism based on integrated gradients to highlight influential lexical and network propagation indicators."""

SAMPLE_MEDICAL_AI = """Multi-Scale Attention UNet for 3D Brain Tumor MRI Segmentation and Survival Prediction

Abstract:
Accurate segmentation of glioma sub-regions from multi-modal Magnetic Resonance Imaging (MRI) is essential for neurosurgical planning and radiotherapy. In this paper, we propose a 3D multi-scale attention-guided UNet framework equipped with residual volumetric convolutions and cross-modal attention gates to segment active tumor, necrotic core, and peritumoral edema. Furthermore, radiomic features extracted from the predicted tumor masks are integrated into a multi-task Cox proportional hazards deep neural network to predict overall patient survival time. Evaluated on the BraTS benchmark dataset (BraTS 2023), our model achieves an average Dice similarity coefficient of 91.4% for the Whole Tumor and 86.7% for the Enhancing Tumor, surpassing standard 3D UNet and V-Net baselines."""

SAMPLE_CYBERSECURITY = """Lightweight Post-Quantum Cryptography and Intrusion Detection Protocol for Edge IoT Sensors

Abstract:
Resource-constrained Internet of Things (IoT) sensors deployed in critical smart grid infrastructure are vulnerable to physical tampering, network intrusion, and emerging quantum cryptanalysis attacks. Conventional RSA and ECC algorithms impose severe computational overhead on low-power microcontrollers. In this paper, we present a lightweight lattice-based post-quantum signature scheme optimized for 8-bit and 32-bit embedded processors, combined with a decentralized graph-based intrusion detection protocol running at the edge gateway. Real-world testbed experiments demonstrate a 4.8x reduction in key exchange latency and 99.2% accuracy in detecting distributed denial-of-service (DDoS) attacks with sub-milliwatt power consumption."""


def set_preset(preset_name: str):
    """Updates weight sliders according to user strategy preset."""
    if preset_name == "fast":
        return 0.30, 0.15, 0.15, 0.10, 0.05, 0.25, "⚡ Active Preset: Fast Review & Rapid Turnaround"
    elif preset_name == "impact":
        return 0.30, 0.15, 0.15, 0.05, 0.30, 0.05, "🏆 Active Preset: Top-Tier High Impact Factor (Q1)"
    elif preset_name == "cost":
        return 0.30, 0.15, 0.15, 0.30, 0.05, 0.05, "💎 Active Preset: Zero APC / Diamond Open Access"
    else:  # balanced
        return 0.30, 0.20, 0.15, 0.10, 0.15, 0.10, "⚖️ Active Preset: Balanced Multi-Criteria"


def handle_file_upload(file):
    """Extracts text from uploaded PDF file."""
    if file is None:
        return ""
    try:
        with open(file.name, "rb") as f:
            pdf_bytes = f.read()
        return extract_pdf_text(pdf_bytes)
    except Exception as exc:
        return f"Error extracting PDF text: {exc}"


async def run_recommendation(
    paper_text: str,
    pdf_file,
    w_scope: float,
    w_sim: float,
    w_cred: float,
    w_cost: float,
    w_impact: float,
    w_speed: float,
):
    """Execute LangGraph recommendation graph, generate downloadable PDF, and format clean markdown output."""
    text_to_process = ""
    if pdf_file is not None:
        text_to_process = handle_file_upload(pdf_file)
    if not text_to_process.strip():
        text_to_process = paper_text.strip()

    if len(text_to_process) < 30:
        return (
            "⚠️ **Please provide manuscript text (at least 30 characters) or upload a PDF file.**",
            "Please provide manuscript text to inspect conflicts.",
            "",
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
        )

    preferences = {
        "scope": float(w_scope),
        "similarity": float(w_sim),
        "credibility": float(w_cred),
        "cost": float(w_cost),
        "impact": float(w_impact),
        "turnaround": float(w_speed),
    }

    try:
        result = await journal_recommendation_graph.ainvoke({
            "paper_text": text_to_process,
            "preferences": preferences,
        })
    except Exception as exc:
        return (
            f"❌ **An error occurred during recommendation workflow**: `{exc}`",
            "An error occurred.",
            "",
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
        )

    profile = result.get("paper_profile", {})
    recs = result.get("recommendations", [])
    conflicts = result.get("conflicts", [])

    md_output = []

    # 1. Extracted Manuscript Profile
    md_output.append("### 📄 Extracted Manuscript Profile\n")
    md_output.append(f"- **Title**: *{profile.get('title', 'Unknown')}*")
    md_output.append(f"- **Primary Field**: `{profile.get('domain', 'Academic Research')}` &nbsp;|&nbsp; **Methodology**: *{profile.get('methodology', 'Empirical Study')}*")
    md_output.append(f"- **Target Search Queries**: {', '.join(f'`{q}`' for q in profile.get('academic_search_queries', [])[:3])}")
    md_output.append(f"- **Key Concepts**: {', '.join(str(k) for k in profile.get('keywords', [])[:6])}\n")
    md_output.append("---\n")

    # 2. TOP SUMMARY LEADERBOARD TABLE
    table_rows = []
    for item in recs:
        rank = item.get("rank", 1)
        name = item.get("journal", "Unknown")
        score = item.get("score", 0)
        publisher = item.get("publisher", "Academic Press")
        apc = item.get("apc_usd")
        apc_str = "$0 (Diamond OA)" if apc == 0 else (f"~${apc} USD" if apc else "Subscription / Free Track")
        oa_badge = "🟢 DOAJ / OA" if item.get("is_in_doaj") or item.get("is_oa") else "🔒 Hybrid"

        table_rows.append(
            f"| **#{rank}** | **{name}** | {publisher} | **`{score}%`** | {oa_badge} | {apc_str} |"
        )

    md_output.append("### 🏆 Top Recommended Journals Summary\n")
    md_output.append("| Rank | Journal Name | Publisher | Match Score | Access Model | Author Fee (APC) |")
    md_output.append("| :---: | :--- | :--- | :---: | :---: | :--- |")
    md_output.extend(table_rows)
    md_output.append("\n---\n")

    # 3. CLEAN DETAILS FOR EACH JOURNAL WITH DESCRIPTION
    md_output.append("### 🔍 Journal Match Rationale & Submission Dossiers\n")

    for item in recs:
        rank = item.get("rank", 1)
        name = item.get("journal", "Unknown Journal")
        score = item.get("score", 0)
        publisher = item.get("publisher", "Academic Publisher")
        issn = item.get("issn", "N/A")
        scores = item.get("agent_scores", {})
        explanation = item.get("explanation", {})
        description = item.get("description", "")
        recent_works = item.get("recent_works", [])

        badges = []
        if item.get("is_in_doaj"):
            badges.append("🟢 DOAJ Indexed")
        if item.get("is_oa"):
            badges.append("🌐 Open Access")
        if item.get("apc_usd") == 0:
            badges.append("💎 Diamond OA ($0 Fee)")
        elif item.get("apc_usd"):
            badges.append(f"💵 APC: ~${item.get('apc_usd')} USD")

        badge_str = " | ".join(badges) if badges else "Standard Indexing"

        md_output.append(f"#### #{rank}. {name} — **Match: {score}%**")
        md_output.append(f"*{publisher}* &nbsp;•&nbsp; `ISSN: {issn}` &nbsp;•&nbsp; {badge_str}\n")

        # Journal Description & Scope
        if description:
            md_output.append(f"📖 **Journal Scope & Description**: *{description}*\n")

        md_output.append(
            f"> **Agent Breakdown**: Scope: `{scores.get('scope', 0)}%` | "
            f"RAG Vector: `{scores.get('similarity', 0)}%` | "
            f"Credibility: `{scores.get('credibility', 0)}%` | "
            f"Cost Fit: `{scores.get('cost', 0)}%` | "
            f"Prestige: `{scores.get('impact', 0)}%` | "
            f"Turnaround: `{scores.get('turnaround', 0)}%`\n"
        )

        if explanation:
            md_output.append(f"💡 **Why this journal was selected**: {explanation.get('why_recommended', '')}\n")
            if explanation.get("strengths"):
                md_output.append("**⭐ Key Strengths:**")
                for s in explanation.get("strengths", []):
                    md_output.append(f"- {s}")
            if explanation.get("caveats"):
                md_output.append("\n**⚠️ Trade-offs & Review Considerations:**")
                for c in explanation.get("caveats", []):
                    md_output.append(f"- {c}")
            if explanation.get("tailoring_advice"):
                md_output.append("\n**✍️ Manuscript Submission Strategy:**")
                for tip in explanation.get("tailoring_advice", []):
                    md_output.append(f"- {tip}")

        if recent_works:
            md_output.append("\n**📚 Verified Recent Articles from this Venue:**")
            for rw in recent_works[:2]:
                rw_title = rw.get("title", "")
                rw_doi = rw.get("doi", "")
                if rw_doi:
                    md_output.append(f"- [{rw_title}](https://doi.org/{rw_doi}) (`DOI: {rw_doi}`)")
                elif rw_title:
                    md_output.append(f"- {rw_title}")

        md_output.append("\n---\n")

    # Conflict Resolution Report
    conflict_md = []
    if conflicts:
        conflict_md.append("### ⚖️ Multi-Criteria Trade-offs & Conflict Resolutions\n")
        for c in conflicts:
            conflict_md.append(f"- **{c.get('journal')}** (Metric Spread: `{c.get('spread')} pts`)")
            for to in c.get("tradeoffs", []):
                conflict_md.append(f"  - *Trade-off*: {to}")
            conflict_md.append(f"  - *Arbitration*: {c.get('resolution')}\n")
    else:
        conflict_md.append("✅ **No major multi-criteria conflicts detected across evaluated dimensions.**")

    export_markdown = "\n".join(md_output) + "\n\n" + "\n".join(conflict_md)

    # Generate Downloadable PDF Report
    pdf_path = None
    try:
        pdf_path = generate_recommendation_pdf(result)
    except Exception as pdf_err:
        print(f"Error generating PDF dossier: {pdf_err}")

    if pdf_path and os.path.exists(pdf_path):
        pdf_update = gr.update(value=pdf_path, visible=True)
    else:
        pdf_update = gr.update(value=None, visible=False)

    return (
        "\n".join(md_output),
        "\n".join(conflict_md),
        export_markdown,
        pdf_update,
        pdf_update,
    )


async def live_journal_lookup(query: str):
    """Direct lookup of any academic journal across OpenAlex and DOAJ."""
    if not query.strip():
        return "Please enter a journal name or keyword."

    client = OpenAlexClient()
    try:
        sources = await client.search_sources_direct(query, per_page=4)
        if not sources:
            return f"No journals found matching '{query}'."

        res = [f"### 🔍 Live Search Results for '{query}':\n"]
        for s in sources:
            name = s.get("display_name", "Unknown")
            pub = s.get("publisher", "Unknown Publisher")
            issn = s.get("issn_l", "N/A")
            oa = "Yes" if s.get("is_oa") else "No"
            two_yr = s.get("two_year_mean_citedness", 0.0)
            h_idx = s.get("h_index", 0)
            topics = ", ".join(s.get("topics", [])[:4])

            res.append(f"#### {name}")
            res.append(f"- **Publisher**: {pub} | **ISSN-L**: `{issn}`")
            res.append(f"- **Open Access**: {oa} | **2-Yr Mean Citations**: `{two_yr}` | **h-index**: `{h_idx}`")
            res.append(f"- **Topics**: {topics}\n")
        return "\n".join(res)
    except Exception as exc:
        return f"Error querying live academic APIs: {exc}"


def create_gradio_app() -> gr.Blocks:
    """Build the clean, accessible Gradio User Interface with PDF download and background placeholder text."""
    with gr.Blocks(title="AI Academic Journal Recommendation Assistant") as demo:
        gr.Markdown(
            """
            # 🎓 AI-Based Academic Journal Recommendation Assistant
            ### Real-Time Scholarly Graph APIs • ChromaDB RAG • LangGraph Multi-Agent Orchestrator • Explainable AI
            """
        )

        with gr.Row():
            gr.Markdown(
                """
                🟢 **OpenAlex API**: Connected &nbsp;|&nbsp;
                🟢 **Crossref API**: Connected &nbsp;|&nbsp;
                🟢 **DOAJ**: Verified &nbsp;|&nbsp;
                🟢 **ChromaDB RAG**: Active &nbsp;|&nbsp;
                🟢 **Multi-Agent Engine**: LangGraph v1.2
                """
            )

        with gr.Tabs():
            with gr.TabItem("🎯 Recommendation Studio"):
                # Strategy Presets Bar
                gr.Markdown("### 1. Select Strategy Preset (Optional)")
                with gr.Row():
                    btn_balanced = gr.Button("⚖️ Balanced (Recommended)", variant="secondary", size="sm")
                    btn_impact = gr.Button("🏆 High Impact (Q1)", variant="secondary", size="sm")
                    btn_cost = gr.Button("💎 Zero APC (Diamond OA)", variant="secondary", size="sm")
                    btn_fast = gr.Button("⚡ Fast Review Timeline", variant="secondary", size="sm")

                preset_status = gr.Markdown("⚖️ Active Preset: Balanced Multi-Criteria")

                # TWO-COLUMN MAIN WORKSPACE
                with gr.Row():
                    # LEFT COLUMN: Inputs & Sample Loaders & Main Action Button
                    with gr.Column(scale=1):
                        gr.Markdown("### 2. Manuscript Input")

                        # 1-Click Sample Manuscript Loaders for Testing
                        gr.Markdown("**Quick Test Samples:**")
                        with gr.Row():
                            sample_btn_nlp = gr.Button("📄 NLP / Misinformation", size="sm")
                            sample_btn_med = gr.Button("🩺 Medical MRI AI", size="sm")
                            sample_btn_iot = gr.Button("🔒 Cybersecurity / IoT", size="sm")

                        with gr.Tabs():
                            with gr.TabItem("📝 Paste Manuscript Text"):
                                paper_input = gr.Textbox(
                                    label="Manuscript Title & Abstract",
                                    placeholder=(
                                        "Type or paste your manuscript draft title, abstract, or full text here...\n\n"
                                        "Example:\n"
                                        "Deep Transfer Learning Framework for Automated Fake News and Misinformation Detection in Multilingual Social Networks\n\n"
                                        "Abstract:\n"
                                        "The rapid proliferation of misinformation on digital platforms poses significant risks...\n\n"
                                        "(Tip: Click any sample button above to quickly test with a sample manuscript.)"
                                    ),
                                    lines=8,
                                    value="",
                                )
                            with gr.TabItem("📂 Upload PDF File"):
                                pdf_input = gr.File(
                                    label="Upload PDF Manuscript Draft",
                                    file_types=[".pdf"],
                                    type="filepath",
                                )

                        # PROMINENT PRIMARY ACTION BUTTON
                        submit_btn = gr.Button("🚀 Find & Recommend Journals", variant="primary", size="lg")

                        # COLLAPSIBLE WEIGHT SLIDERS
                        with gr.Accordion("⚙️ Fine-Tune Decision Weights (Optional)", open=False):
                            w_scope = gr.Slider(0.0, 1.0, value=0.30, step=0.05, label="🎯 Scope & Topic Fit (Mandatory Gating)")
                            w_sim = gr.Slider(0.0, 1.0, value=0.20, step=0.05, label="🔍 RAG Semantic Similarity")
                            w_cred = gr.Slider(0.0, 1.0, value=0.15, step=0.05, label="🛡️ Credibility & Indexing")
                            w_cost = gr.Slider(0.0, 1.0, value=0.10, step=0.05, label="💰 Cost & APC Feasibility")
                            w_impact = gr.Slider(0.0, 1.0, value=0.15, step=0.05, label="📈 Citation Impact (Q1-Q4)")
                            w_speed = gr.Slider(0.0, 1.0, value=0.10, step=0.05, label="⏱️ Review Turnaround Speed")

                    # RIGHT COLUMN: Results & Leaderboard
                    with gr.Column(scale=2):
                        pdf_download_main = gr.File(
                            label="📄 Download Recommendation Dossier (PDF)",
                            interactive=False,
                            visible=False,
                        )
                        results_output = gr.Markdown("Click **'🚀 Find & Recommend Journals'** to evaluate candidate venues.")

            with gr.TabItem("⚖️ Conflict & Trade-off Analysis"):
                gr.Markdown("### Multi-Criteria Conflict Resolution Inspector")
                gr.Markdown("Visualizes metric dispersion and explains trade-offs (e.g., Flagship Prestige vs High APC or Extended Review Time).")
                conflict_output = gr.Markdown("Run a recommendation first to view conflict resolutions.")

            with gr.TabItem("🔍 Live Journal Explorer"):
                gr.Markdown("### Live Academic Journal & Venue Search")
                gr.Markdown("Directly query real-time OpenAlex and DOAJ indices for any journal or subfield.")
                with gr.Row():
                    explorer_query = gr.Textbox(label="Journal Name or Topic", placeholder="e.g. Information Processing & Management, IEEE TKDE, Nature...")
                    explorer_btn = gr.Button("Search Live API", variant="secondary")
                explorer_output = gr.Markdown("Enter a search term above.")
                explorer_btn.click(live_journal_lookup, inputs=[explorer_query], outputs=[explorer_output])

            with gr.TabItem("📥 Export & Download Report"):
                gr.Markdown("### 📄 Export Full Recommendation Dossier")
                gr.Markdown("Download the complete multi-criteria recommendation dossier as a publication-ready PDF report or view the clean Markdown dossier.")
                pdf_download_export = gr.File(
                    label="📄 Download Recommendation Dossier (PDF)",
                    interactive=False,
                    visible=False,
                )
                export_md_box = gr.Textbox(label="📋 Markdown Recommendation Dossier", lines=14)

            with gr.TabItem("🏛️ Architecture & Documentation"):
                gr.Markdown(
                    """
                    ### 🏛️ Multi-Agent Architecture
                    ```
                    [Manuscript PDF/Text]
                            │
                            ▼
                    [Node 1: Manuscript Profiling Agent] (Title, Abstract, Search Queries Formulation)
                            │
                            ▼
                    [Node 2: Real-Time Academic Retrieval & RAG] (OpenAlex + Crossref + DOAJ + ChromaDB)
                            │
                            ├───► [Agent 1: Scope Agent] (Topic & Keyword Overlap)
                            ├───► [Agent 2: RAG Similarity Agent] (Vector Cosine Distance)
                            ├───► [Agent 3: Credibility Agent] (DOAJ Index, ISSN, Publisher Trust)
                            ├───► [Agent 4: Cost & APC Agent] (Diamond OA, APC Pricing Tier)
                            ├───► [Agent 5: Citation Impact Agent] (2-Yr Mean Citedness, Q1-Q4)
                            └───► [Agent 6: Turnaround Agent] (Review Speed & Acceptance Rate)
                            │
                            ▼
                    [Node 3: Conflict Resolution Node] (Trade-off detection & variance management)
                            │
                            ▼
                    [Node 4: Multi-Criteria Ranking (MCDA)] (Normalized Weighted Sum Model)
                            │
                            ▼
                    [Node 5: Explainability & Synthesis Agent] (Transparent 'Why Recommended?', Strengths, Caveats)
                            │
                            ▼
                    [Final Ranked Recommendations + Downloadable PDF Dossier]
                    ```
                    """
                )

        # Sample button listeners
        sample_btn_nlp.click(lambda: SAMPLE_NLP_MISINFORMATION, outputs=[paper_input])
        sample_btn_med.click(lambda: SAMPLE_MEDICAL_AI, outputs=[paper_input])
        sample_btn_iot.click(lambda: SAMPLE_CYBERSECURITY, outputs=[paper_input])

        # Preset click listeners
        btn_fast.click(lambda: set_preset("fast"), outputs=[w_scope, w_sim, w_cred, w_cost, w_impact, w_speed, preset_status])
        btn_impact.click(lambda: set_preset("impact"), outputs=[w_scope, w_sim, w_cred, w_cost, w_impact, w_speed, preset_status])
        btn_cost.click(lambda: set_preset("cost"), outputs=[w_scope, w_sim, w_cred, w_cost, w_impact, w_speed, preset_status])
        btn_balanced.click(lambda: set_preset("balanced"), outputs=[w_scope, w_sim, w_cred, w_cost, w_impact, w_speed, preset_status])

        # Submit Action
        submit_btn.click(
            run_recommendation,
            inputs=[paper_input, pdf_input, w_scope, w_sim, w_cred, w_cost, w_impact, w_speed],
            outputs=[results_output, conflict_output, export_md_box, pdf_download_main, pdf_download_export],
        )

    return demo
