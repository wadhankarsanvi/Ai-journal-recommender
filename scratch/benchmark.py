import asyncio
import time
from journal_ai.orchestration.graph import journal_recommendation_graph
from journal_ai.agents.profile_agent import extract_manuscript_profile_async
from journal_ai.data_sources.aggregator import AcademicDataAggregator
from journal_ai.rag.pipeline import get_rag_pipeline
from journal_ai.agents.explainer_agent import run_explainer_agent

async def main():
    text = (
        "Multi-Scale Attention UNet for 3D Brain Tumor MRI Segmentation and Survival Prediction\n\n"
        "Abstract:\n"
        "Accurate segmentation of glioma sub-regions from multi-modal Magnetic Resonance Imaging (MRI) is essential..."
    )

    t0 = time.perf_counter()
    profile = await extract_manuscript_profile_async(text)
    t1 = time.perf_counter()
    print(f"1. Profile extraction took: {t1 - t0:.2f}s")
    print("Queries:", profile.get("academic_search_queries"))

    agg = AcademicDataAggregator()
    candidates = await agg.fetch_candidates(queries=profile.get("academic_search_queries", []), limit=6)
    t2 = time.perf_counter()
    print(f"2. Candidates fetch took: {t2 - t1:.2f}s ({len(candidates)} candidates)")

    rag = get_rag_pipeline()
    indexed = rag.index_journals(candidates)
    t3 = time.perf_counter()
    print(f"3. RAG index took: {t3 - t2:.2f}s ({indexed} docs)")

    state = {
        "paper_profile": profile,
        "candidate_journals": candidates,
        "recommendations": [{"journal_id": c["id"], "score": 85.0, "agent_scores": {}, "weights": {}} for c in candidates]
    }
    exp = await run_explainer_agent(state)
    t4 = time.perf_counter()
    print(f"4. Explainer agent took: {t4 - t3:.2f}s")

    print(f"Total pipeline test took: {t4 - t0:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
