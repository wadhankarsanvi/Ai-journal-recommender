import pytest
from journal_ai.orchestration.graph import journal_recommendation_graph


@pytest.mark.asyncio
async def test_graph_runs_and_produces_recommendations():
    result = await journal_recommendation_graph.ainvoke({
        "paper_text": (
            "Deep learning and transformer models for fake news detection and misinformation "
            "classification using multilingual natural language processing and graph neural networks."
        ),
        "preferences": {
            "scope": 0.25,
            "similarity": 0.20,
            "credibility": 0.20,
            "cost": 0.10,
            "impact": 0.15,
            "turnaround": 0.10,
        },
    })

    # Assertions on Manuscript Profile
    profile = result.get("paper_profile", {})
    assert profile.get("word_count", 0) > 0
    assert len(profile.get("keywords", [])) > 0
    assert "domain" in profile

    # Assertions on Candidate Retrieval and RAG
    candidates = result.get("candidate_journals", [])
    assert len(candidates) > 0

    # Assertions on Recommendations & Explanation
    recommendations = result.get("recommendations", [])
    assert len(recommendations) > 0
    assert recommendations[0]["rank"] == 1
    assert "score" in recommendations[0]
    assert "agent_scores" in recommendations[0]

    # Verify all 6 specialist agent scores are present
    agent_scores = recommendations[0]["agent_scores"]
    for agent_key in ["scope", "similarity", "credibility", "cost", "impact", "turnaround"]:
        assert agent_key in agent_scores, f"Missing agent key: {agent_key}"

    # Verify explainability card
    explanation = recommendations[0].get("explanation", {})
    assert "why_recommended" in explanation
    assert "strengths" in explanation
    assert "caveats" in explanation
    assert "tailoring_advice" in explanation
