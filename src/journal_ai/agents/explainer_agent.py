import asyncio
import json
from typing import Any
from journal_ai.config.settings import settings
from journal_ai.orchestration.llm_client import LLMClient


async def run_explainer_agent(state: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """
    Multi-Model Explainability & Synthesis Agent.
    Generates personalized, high-fidelity justification cards for why each journal was recommended.
    Uses OpenAI / Gemini / Groq if configured, with deterministic fallback.
    """
    profile = state.get("paper_profile", {})
    recommendations = state.get("recommendations", [])
    candidates = {j["id"]: j for j in state.get("candidate_journals", [])}
    llm = LLMClient()

    async def _process_single(item: dict[str, Any]) -> dict[str, Any]:
        rec = dict(item)
        j_id = rec.get("journal_id")
        journal_info = candidates.get(j_id, {})
        scores = rec.get("agent_scores", {})
        weights = rec.get("weights", {})

        # Generate explanation (LLM or heuristic)
        try:
            explanation = await asyncio.wait_for(
                generate_explanation_card(
                    llm=llm,
                    profile=profile,
                    journal=journal_info,
                    scores=scores,
                    weights=weights,
                    final_score=rec.get("score", 0),
                ),
                timeout=4.0,
            )
        except Exception:
            explanation = generate_heuristic_explanation(
                profile=profile,
                journal=journal_info,
                scores=scores,
                final_score=rec.get("score", 0),
            )

        rec["explanation"] = explanation
        rec["publisher"] = journal_info.get("publisher", "Academic Publisher")
        rec["issn"] = journal_info.get("issn_l") or journal_info.get("issn")
        rec["is_oa"] = journal_info.get("is_oa", False)
        rec["is_in_doaj"] = journal_info.get("is_in_doaj", False)
        rec["apc_usd"] = journal_info.get("apc_usd")
        rec["topics"] = journal_info.get("topics", [])
        rec["recent_works"] = journal_info.get("recent_works", [])[:3]
        rec["description"] = journal_info.get("description") or (
            f"{journal_info.get('display_name', 'This journal')} is a peer-reviewed academic publication by "
            f"{journal_info.get('publisher', 'an academic publisher')} specializing in {', '.join(journal_info.get('topics', [])[:4]) or 'scholarly research'}."
        )
        return rec

    # Process all recommendations concurrently for maximum speed
    tasks = [_process_single(item) for item in recommendations]
    enhanced_recommendations = await asyncio.gather(*tasks)

    return {"recommendations": list(enhanced_recommendations)}


async def generate_explanation_card(
    llm: LLMClient,
    profile: dict[str, Any],
    journal: dict[str, Any],
    scores: dict[str, float],
    weights: dict[str, float],
    final_score: float,
) -> dict[str, Any]:
    """Generate structured decision explanation card using LLM or structured rules."""
    j_name = journal.get("display_name", "Unknown Journal")
    title = profile.get("title", "Manuscript")
    domain = profile.get("domain", "Academic Research")
    methodology = profile.get("methodology", "Empirical Evaluation")
    topics = ", ".join(journal.get("topics", [])[:5])

    # If LLM API key is present (e.g. OpenAI / Groq), generate deep synthesis
    if llm.is_llm_available() and settings.enable_llm_explanations:
        system_prompt = (
            "You are a Senior Academic Journal Editor and Peer Review Advisor. "
            "Explain specifically why this journal is suitable for the submitted manuscript."
        )
        prompt = f"""
Manuscript Title: "{title}"
Manuscript Abstract: "{profile.get('abstract', '')[:600]}"
Domain: {domain} | Methodology: {methodology}
Target Journal: "{j_name}" (Publisher: {journal.get('publisher')})
Journal Scope & Topics: {topics}
Evaluation Sub-Scores: Scope={scores.get('scope')}%, Similarity={scores.get('similarity')}%, Credibility={scores.get('credibility')}%, Cost={scores.get('cost')}%, Impact={scores.get('impact')}%, Speed={scores.get('turnaround')}%
Overall Fit Score: {final_score}/100

Return a JSON object with:
- "why_recommended": (1-2 sentences explaining specific topical alignment and why the editor would consider this paper).
- "strengths": (list of 2-3 specific bullet points highlighting prestige, readership fit, or open access).
- "caveats": (list of 1-2 practical trade-offs like APC fee, review timeline, or strict acceptance rate).
- "tailoring_advice": (list of 2 specific tips for tailoring the abstract/intro to meet this journal's reviewer expectations).
"""
        try:
            llm_res = await asyncio.wait_for(llm.generate_json(prompt, system_prompt=system_prompt), timeout=3.5)
            if llm_res and "why_recommended" in llm_res:
                return llm_res
        except Exception:
            pass

    return generate_heuristic_explanation(
        profile=profile,
        journal=journal,
        scores=scores,
        final_score=final_score,
    )


def generate_heuristic_explanation(
    profile: dict[str, Any],
    journal: dict[str, Any],
    scores: dict[str, float],
    final_score: float,
) -> dict[str, Any]:
    """Deterministic, high-speed heuristic rationale generator."""
    j_name = journal.get("display_name", "Unknown Journal")
    title = profile.get("title", "Manuscript")
    domain = profile.get("domain", "Academic Research")
    methodology = profile.get("methodology", "Empirical Evaluation")
    topics = ", ".join(journal.get("topics", [])[:5])

    why_text = (
        f"Recommended for '{title}' because of strong alignment with {j_name}'s focus on {topics or domain}. "
        f"Achieves a composite multi-criteria match score of {final_score}/100."
    )

    strengths = []
    if scores.get("impact", 0) >= 85:
        strengths.append(f"High Prestige: Exceptional citation velocity with 2-year citedness percentile at ~{journal.get('citedness_2yr_percentile', 85)}%.")
    if journal.get("is_in_doaj") and journal.get("is_oa"):
        strengths.append("Open Access Visibility: DOAJ verified, ensuring high international readership without paywalls.")
    if journal.get("apc_usd") == 0:
        strengths.append("Diamond Open Access ($0 APC): Free for authors to publish.")
    elif journal.get("apc_usd"):
        strengths.append(f"Transparent APC Fee: ~${journal.get('apc_usd')} USD.")
    if not strengths:
        strengths.append(f"Published by established press ({journal.get('publisher', 'Academic Publisher')}).")

    caveats = []
    if journal.get("apc_usd") and journal.get("apc_usd") > 2500:
        caveats.append(f"High APC Cost: ~${journal.get('apc_usd')} USD publication fee. Ensure institutional grant support.")
    if scores.get("impact", 0) >= 90:
        caveats.append(f"Rigorous Acceptance: Estimated acceptance rate is ~{journal.get('acceptance_rate_pct', 18)}%.")
    if not caveats:
        caveats.append("Standard peer-review guidelines apply; verify journal author template.")

    tailoring_tips = [
        f"In Section 1, emphasize how your work addresses open challenges in {topics.split(',')[0] if topics else domain}.",
        f"Highlight the '{methodology}' in your abstract to match {j_name}'s methodological rigor.",
    ]

    return {
        "why_recommended": why_text,
        "strengths": strengths,
        "caveats": caveats,
        "tailoring_advice": tailoring_tips,
    }
