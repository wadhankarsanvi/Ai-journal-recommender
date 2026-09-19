import os
import pytest
from journal_ai.reporting.pdf_generator import generate_recommendation_pdf


def test_pdf_generation():
    dummy_state = {
        "paper_profile": {
            "title": "Deep Transfer Learning for Misinformation Detection",
            "domain": "Computer Science & AI",
            "methodology": "Graph Neural Networks & Contrastive Learning",
            "keywords": ["NLP", "Fake News", "Graph Neural Networks", "Transformers"],
            "academic_search_queries": ["misinformation detection NLP", "fake news graph neural networks"],
        },
        "recommendations": [
            {
                "rank": 1,
                "journal": "Information Processing & Management",
                "publisher": "Elsevier BV",
                "issn": "0306-4573",
                "score": 93.5,
                "is_oa": False,
                "is_in_doaj": False,
                "apc_usd": 3150,
                "description": "Focuses on computing and information science at the intersection of information retrieval, natural language processing, and social media analytics.",
                "agent_scores": {
                    "scope": 95.0,
                    "similarity": 92.0,
                    "credibility": 96.0,
                    "cost": 50.0,
                    "impact": 94.0,
                    "turnaround": 88.0,
                },
                "explanation": {
                    "why_recommended": "Strong thematic alignment with natural language processing and misinformation modeling.",
                    "strengths": ["High Q1 citation impact", "Established Elsevier reputation"],
                    "caveats": ["Standard review timeline applies"],
                    "tailoring_advice": ["Emphasize algorithmic evaluation and societal implications in abstract"],
                },
                "recent_works": [
                    {"title": "Multimodal Fake News Detection", "doi": "10.1016/j.ipm.2024.103720"}
                ],
            }
        ],
        "conflicts": [
            {
                "journal": "Information Processing & Management",
                "spread": 22.0,
                "tradeoffs": ["High Prestige vs. APC Fee"],
                "resolution": "MCDA normalized weighting applied.",
            }
        ],
    }

    pdf_path = generate_recommendation_pdf(dummy_state)
    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 1000

    # Cleanup temporary test file
    try:
        os.remove(pdf_path)
    except Exception:
        pass
