import asyncio
from typing import Any
from langgraph.graph import StateGraph, START, END

from journal_ai.orchestration.state import JournalState
from journal_ai.agents.profile_agent import extract_manuscript_profile_async
from journal_ai.agents.scope_agent import run_scope_agent
from journal_ai.agents.similarity_agent import run_similarity_agent
from journal_ai.agents.credibility_agent import run_credibility_agent
from journal_ai.agents.cost_agent import run_cost_agent
from journal_ai.agents.impact_agent import run_impact_agent
from journal_ai.agents.turnaround_agent import run_turnaround_agent
from journal_ai.agents.explainer_agent import run_explainer_agent
from journal_ai.data_sources.aggregator import AcademicDataAggregator
from journal_ai.rag.pipeline import get_rag_pipeline
from journal_ai.config.settings import settings


async def manuscript_analysis_node(state: JournalState) -> dict[str, Any]:
    """Extract structured manuscript profile and search queries (LLM-assisted)."""
    text = state.get("paper_text", "")
    profile = await extract_manuscript_profile_async(text)
    return {"paper_profile": profile}


async def retrieve_candidates_node(state: JournalState) -> dict[str, Any]:
    """
    Retrieve live academic candidates from OpenAlex, Crossref, and DOAJ,
    index candidate publications into ChromaDB, and retrieve semantic RAG evidence.
    """
    profile = state.get("paper_profile", {})
    queries = profile.get("academic_search_queries", [])
    if not queries:
        keywords = profile.get("keywords", [])
        domain = profile.get("domain", "")
        queries = [" ".join(keywords[:5]) or domain or "machine learning"]

    # 1. Fetch real-time enriched candidates across targeted queries
    aggregator = AcademicDataAggregator()
    candidates = await aggregator.fetch_candidates(queries=queries, limit=6)

    # 2. Index candidate journal profiles & recent papers into RAG vector store (if enabled)
    evidence = []
    if settings.enable_rag:
        try:
            rag = get_rag_pipeline()
            await asyncio.wait_for(
                asyncio.to_thread(rag.index_journals, candidates),
                timeout=3.0,
            )
            paper_query = f"Title: {profile.get('title', '')}. Abstract: {profile.get('abstract', '')[:400]}. Keywords: {', '.join(profile.get('keywords', [])[:8])}."
            evidence = await asyncio.wait_for(
                asyncio.to_thread(rag.retrieve_evidence, paper_query, 10),
                timeout=2.0,
            )
        except Exception as exc:
            print(f"[RAG] Indexing/retrieval fallback note: {exc}", flush=True)

    return {
        "candidate_journals": candidates,
        "rag_evidence": evidence,
    }


def conflict_resolution_node(state: JournalState) -> dict[str, Any]:
    """
    Multi-Criteria Conflict Resolution Node:
    Detects high-variance trade-offs across agent evaluations and provides arbitration rationale.
    """
    result_sets = {
        "scope": state.get("scope_results", []),
        "similarity": state.get("similarity_results", []),
        "credibility": state.get("credibility_results", []),
        "cost": state.get("cost_results", []),
        "impact": state.get("impact_results", []),
        "turnaround": state.get("turnaround_results", []),
    }

    conflicts = []
    resolved = []

    for journal in state.get("candidate_journals", []):
        j_id = journal["id"]
        scores: dict[str, float] = {}

        for agent_name, results in result_sets.items():
            for item in results:
                if str(item.get("journal_id")) == str(j_id):
                    scores[agent_name] = float(item.get("score", 70.0))

        if scores:
            spread = max(scores.values()) - min(scores.values())
            highest_metric = max(scores, key=scores.get)
            lowest_metric = min(scores, key=scores.get)
            high_val = scores[highest_metric]
            low_val = scores[lowest_metric]

            metric_labels = {
                "scope": "Manuscript Scope Fit",
                "similarity": "Semantic Content Alignment",
                "credibility": "Publisher Indexing & Credibility",
                "cost": "Author APC Cost / Fee Feasibility",
                "impact": "Citation Impact & Prestige",
                "turnaround": "Review Speed & Turnaround",
            }

            tradeoffs = []
            # Specific heuristic rules
            if scores.get("impact", 0) >= 80 and scores.get("cost", 0) <= 55:
                tradeoffs.append(f"Top-Tier Impact ({scores.get('impact', 0):.0f}%) vs. Higher Author APC Fee ({scores.get('cost', 0):.0f}%)")
            if scores.get("impact", 0) >= 80 and scores.get("turnaround", 0) <= 60:
                tradeoffs.append(f"Flagship Prestige ({scores.get('impact', 0):.0f}%) vs. Extended Peer-Review Timeline ({scores.get('turnaround', 0):.0f}%)")
            if scores.get("cost", 0) >= 85 and scores.get("impact", 0) < 70:
                tradeoffs.append(f"Affordable / Open Access ({scores.get('cost', 0):.0f}%) vs. Moderate Citation Velocity ({scores.get('impact', 0):.0f}%)")
            if scores.get("scope", 0) >= 85 and scores.get("similarity", 0) <= 65:
                tradeoffs.append(f"Broad Domain Alignment ({scores.get('scope', 0):.0f}%) vs. Moderate Keyword Match ({scores.get('similarity', 0):.0f}%)")

            # If no specific rule matched but spread is high, generate dynamic pair
            if not tradeoffs and spread >= 20:
                tradeoffs.append(
                    f"Strong {metric_labels.get(highest_metric, highest_metric)} ({high_val:.0f}%) vs. "
                    f"Lower {metric_labels.get(lowest_metric, lowest_metric)} ({low_val:.0f}%)"
                )

            if spread >= 20 or tradeoffs:
                conflicts.append({
                    "journal_id": j_id,
                    "journal": journal.get("display_name", "Unknown Journal"),
                    "scores": scores,
                    "spread": round(spread, 1),
                    "highest_metric": f"{metric_labels.get(highest_metric, highest_metric)} ({high_val:.0f}%)",
                    "lowest_metric": f"{metric_labels.get(lowest_metric, lowest_metric)} ({low_val:.0f}%)",
                    "tradeoffs": tradeoffs,
                    "resolution": (
                        f"Ranked via MCDA weighted utility prioritizing {metric_labels.get(highest_metric, highest_metric).lower()} "
                        f"while penalizing {metric_labels.get(lowest_metric, lowest_metric).lower()} according to active preset sliders."
                    ),
                })

        resolved.append({
            "journal_id": j_id,
            "journal": journal.get("display_name", "Unknown Journal"),
            "scores": scores,
        })

    return {
        "conflicts": conflicts,
        "resolved_results": resolved,
    }


def ranking_node(state: JournalState) -> dict[str, Any]:
    """
    Computes final composite score using Multi-Criteria Decision Analysis with Scope Gating.
    Out-of-scope journals (scope < 50) are penalized/filtered.
    """
    default_weights = {
        "scope": 0.30,
        "similarity": 0.20,
        "credibility": 0.15,
        "cost": 0.10,
        "impact": 0.15,
        "turnaround": 0.10,
    }

    supplied = state.get("preferences") or {}
    weights = {
        key: float(supplied.get(key, default_weights.get(key, 0.15)))
        for key in default_weights
    }

    total_weight = sum(max(0.0, w) for w in weights.values())
    if total_weight == 0:
        normalized_weights = default_weights
    else:
        normalized_weights = {k: max(0.0, v) / total_weight for k, v in weights.items()}

    recommendations = []

    for item in state.get("resolved_results", []):
        scores = item.get("scores", {})
        scope_score = scores.get("scope", 70.0)

        # Strict Scope Gating: Out-of-scope journals are proportionally downranked
        scope_multiplier = 1.0 if scope_score >= 60 else max(0.2, scope_score / 100.0)

        weighted_score = sum(scores.get(k, 60.0) * normalized_weights[k] for k in normalized_weights)
        final_score = round(weighted_score * scope_multiplier, 1)

        recommendations.append({
            "journal_id": item["journal_id"],
            "journal": item["journal"],
            "score": final_score,
            "agent_scores": scores,
            "weights": {k: round(v, 3) for k, v in normalized_weights.items()},
        })

    recommendations.sort(key=lambda x: x["score"], reverse=True)

    for rank, item in enumerate(recommendations, start=1):
        item["rank"] = rank

    return {"recommendations": recommendations}


def build_graph():
    """Constructs and compiles the full LangGraph recommendation graph."""
    graph = StateGraph(JournalState)

    # 1. Pipeline Stages
    graph.add_node("manuscript_analysis", manuscript_analysis_node)
    graph.add_node("retrieve_candidates", retrieve_candidates_node)

    # 2. Parallel Specialist Agents
    graph.add_node("scope_agent", run_scope_agent)
    graph.add_node("similarity_agent", run_similarity_agent)
    graph.add_node("credibility_agent", run_credibility_agent)
    graph.add_node("cost_agent", run_cost_agent)
    graph.add_node("impact_agent", run_impact_agent)
    graph.add_node("turnaround_agent", run_turnaround_agent)

    # 3. Decision Synthesis & Ranking
    graph.add_node("conflict_resolution", conflict_resolution_node)
    graph.add_node("ranking", ranking_node)
    graph.add_node("explainability", run_explainer_agent)

    # Graph Edges & Flow
    graph.add_edge(START, "manuscript_analysis")
    graph.add_edge("manuscript_analysis", "retrieve_candidates")

    # Parallel Fan-Out
    graph.add_edge("retrieve_candidates", "scope_agent")
    graph.add_edge("retrieve_candidates", "similarity_agent")
    graph.add_edge("retrieve_candidates", "credibility_agent")
    graph.add_edge("retrieve_candidates", "cost_agent")
    graph.add_edge("retrieve_candidates", "impact_agent")
    graph.add_edge("retrieve_candidates", "turnaround_agent")

    # Fan-In to Conflict Resolution
    graph.add_edge("scope_agent", "conflict_resolution")
    graph.add_edge("similarity_agent", "conflict_resolution")
    graph.add_edge("credibility_agent", "conflict_resolution")
    graph.add_edge("cost_agent", "conflict_resolution")
    graph.add_edge("impact_agent", "conflict_resolution")
    graph.add_edge("turnaround_agent", "conflict_resolution")

    # Ranking & Explainability Synthesis
    graph.add_edge("conflict_resolution", "ranking")
    graph.add_edge("ranking", "explainability")
    graph.add_edge("explainability", END)

    return graph.compile()


journal_recommendation_graph = build_graph()
