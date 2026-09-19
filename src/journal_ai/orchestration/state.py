from typing import Any, TypedDict


class JournalState(TypedDict, total=False):
    """Complete shared state passed through the LangGraph recommendation workflow."""

    # Input parameters
    paper_text: str
    preferences: dict[str, float]

    # Extracted Manuscript Profile
    paper_profile: dict[str, Any]

    # Retrieved Candidate Venues & RAG context
    candidate_journals: list[dict[str, Any]]
    rag_evidence: list[dict[str, Any]]

    # Parallel Specialist Agent Results
    scope_results: list[dict[str, Any]]
    similarity_results: list[dict[str, Any]]
    credibility_results: list[dict[str, Any]]
    cost_results: list[dict[str, Any]]
    impact_results: list[dict[str, Any]]
    turnaround_results: list[dict[str, Any]]

    # Decision Synthesis & Conflict Resolution
    conflicts: list[dict[str, Any]]
    resolved_results: list[dict[str, Any]]

    # Final Explainable Recommendations
    recommendations: list[dict[str, Any]]