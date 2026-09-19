import pytest
from journal_ai.agents.profile_agent import extract_manuscript_profile
from journal_ai.agents.scope_agent import run_scope_agent
from journal_ai.agents.credibility_agent import run_credibility_agent
from journal_ai.agents.cost_agent import run_cost_agent
from journal_ai.agents.impact_agent import run_impact_agent
from journal_ai.agents.turnaround_agent import run_turnaround_agent


def test_profile_agent_extraction():
    text = (
        "Quantum Machine Learning Algorithms for High Energy Physics\n"
        "Abstract: We propose a variational quantum circuit for particle track reconstruction..."
    )
    profile = extract_manuscript_profile(text)
    assert profile["word_count"] > 10
    assert len(profile["keywords"]) > 0
    assert "methodology" in profile


def test_specialist_agents_execution():
    mock_state = {
        "paper_profile": {
            "title": "Quantum ML",
            "keywords": ["quantum", "machine", "learning", "neural"],
            "domain": "Artificial Intelligence & Machine Learning",
            "methodology": "Empirical Study",
        },
        "candidate_journals": [
            {
                "id": "test-j1",
                "display_name": "IEEE Transactions on Pattern Analysis and Machine Intelligence",
                "publisher": "IEEE Computer Society",
                "topics": ["Machine Learning", "Artificial Intelligence"],
                "is_in_doaj": False,
                "is_oa": False,
                "apc_usd": 2495,
                "citedness_2yr_percentile": 99.0,
                "review_time_weeks": 14,
            },
            {
                "id": "test-j2",
                "display_name": "Journal of Machine Learning Research",
                "publisher": "Microtome",
                "topics": ["Machine Learning", "Optimization"],
                "is_in_doaj": True,
                "is_oa": True,
                "apc_usd": 0,
                "citedness_2yr_percentile": 96.0,
                "review_time_weeks": 8,
            },
        ],
    }

    scope_res = run_scope_agent(mock_state)
    assert len(scope_res["scope_results"]) == 2
    assert scope_res["scope_results"][0]["score"] > 60

    cred_res = run_credibility_agent(mock_state)
    assert len(cred_res["credibility_results"]) == 2

    cost_res = run_cost_agent(mock_state)
    assert len(cost_res["cost_results"]) == 2
    # JMLR is Diamond OA ($0 fee), should receive maximum cost score (100)
    jmlr_cost = next(r for r in cost_res["cost_results"] if r["journal_id"] == "test-j2")
    assert jmlr_cost["score"] == 100.0

    impact_res = run_impact_agent(mock_state)
    assert len(impact_res["impact_results"]) == 2
    assert impact_res["impact_results"][0]["score"] >= 90.0

    turnaround_res = run_turnaround_agent(mock_state)
    assert len(turnaround_res["turnaround_results"]) == 2
