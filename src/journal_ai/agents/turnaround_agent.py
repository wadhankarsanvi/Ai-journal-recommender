from typing import Any


def run_turnaround_agent(state: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """
    Evaluates peer-review turnaround speed, time-to-first-decision,
    and publication timeline feasibility.
    """
    results = []

    for journal in state.get("candidate_journals", []):
        review_weeks = int(journal.get("review_time_weeks") or 12)
        acceptance_pct = int(journal.get("acceptance_rate_pct") or 25)

        # Faster review time = higher speed score
        # 4-6 weeks -> 95-90 pts, 8-10 weeks -> 85-75 pts, 16+ weeks -> 50 pts
        if review_weeks <= 6:
            speed_score = 95.0
            speed_label = "Rapid Review (~4-6 weeks)"
        elif review_weeks <= 10:
            speed_score = 82.0
            speed_label = "Standard Rapid (~8-10 weeks)"
        elif review_weeks <= 16:
            speed_score = 65.0
            speed_label = "Moderate Timeline (~12-16 weeks)"
        else:
            speed_score = 45.0
            speed_label = "Lengthy Review Cycle (~18+ weeks)"

        results.append({
            "journal_id": journal["id"],
            "journal": journal.get("display_name", "Unknown Journal"),
            "score": speed_score,
            "review_weeks": review_weeks,
            "acceptance_rate_pct": acceptance_pct,
            "speed_label": speed_label,
            "reason": f"{speed_label} with estimated ~{acceptance_pct}% acceptance rate.",
        })

    return {"turnaround_results": results}
