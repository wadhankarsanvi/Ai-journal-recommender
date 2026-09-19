from typing import Any


def run_impact_agent(state: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """
    Evaluates journal academic prestige, citation impact, 2-year citedness velocity,
    h-index, and estimated quartile ranking (Q1 - Q4).
    """
    results = []

    for journal in state.get("candidate_journals", []):
        percentile = float(journal.get("citedness_2yr_percentile") or 60.0)
        two_yr = float(journal.get("two_year_mean_citedness") or (percentile / 15.0))
        h_idx = int(journal.get("h_index") or 40)

        # Quartile ranking proxy
        if percentile >= 85.0:
            quartile = "Q1 (Top 15% Prestigious)"
            tier_desc = "Flagship high-impact venue with exceptional citation visibility."
        elif percentile >= 65.0:
            quartile = "Q2 (Top Tier / Highly Cited)"
            tier_desc = "Strong reputable journal with established international readership."
        elif percentile >= 40.0:
            quartile = "Q3 (Established Academic)"
            tier_desc = "Solid specialized academic journal."
        else:
            quartile = "Q4 (Emerging / Niche)"
            tier_desc = "Emerging or regional publication venue."

        score = round(min(100.0, max(20.0, percentile)), 1)

        results.append({
            "journal_id": journal["id"],
            "journal": journal.get("display_name", "Unknown Journal"),
            "score": score,
            "quartile": quartile,
            "two_year_mean_citedness": round(two_yr, 2),
            "h_index": h_idx,
            "reason": f"{quartile} - {tier_desc} (2-Yr Mean Citations: {round(two_yr, 2)}, h-index: {h_idx})",
        })

    return {"impact_results": results}
