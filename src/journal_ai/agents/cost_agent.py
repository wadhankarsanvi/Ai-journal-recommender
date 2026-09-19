from typing import Any


def run_cost_agent(state: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """
    Evaluates financial feasibility, Article Processing Charges (APC),
    Diamond Open Access eligibility, and publication cost efficiency.
    """
    results = []

    for journal in state.get("candidate_journals", []):
        apc = journal.get("apc_usd")
        is_oa = journal.get("is_oa", False)
        is_doaj = journal.get("is_in_doaj", False)

        if apc == 0 or (is_doaj and apc is not None and apc == 0):
            score = 100.0
            cost_tier = "Diamond Open Access ($0 USD)"
            reason = "Diamond Open Access: Free to publish and free to read. Zero author-facing fees."
        elif apc is not None:
            if apc < 1000:
                score = 85.0
                cost_tier = f"Low APC (~${apc} USD)"
                reason = f"Low Article Processing Charge (${apc} USD); highly affordable for funded/institutional authors."
            elif apc <= 2000:
                score = 70.0
                cost_tier = f"Moderate APC (~${apc} USD)"
                reason = f"Moderate Article Processing Charge (${apc} USD); standard grant funding coverage."
            elif apc <= 3000:
                score = 55.0
                cost_tier = f"High APC (~${apc} USD)"
                reason = f"High Article Processing Charge (${apc} USD); check for institutional waiver eligibility."
            else:
                score = 40.0
                cost_tier = f"Premium Tier APC (~${apc} USD)"
                reason = f"Premium Article Processing Charge (${apc} USD); significant funding required."
        else:
            # Subscription / Hybrid model (authors can typically publish for free under subscription model)
            if not is_oa:
                score = 80.0
                cost_tier = "Subscription / Hybrid (Free Author Track Available)"
                reason = "Hybrid/Subscription journal: Standard non-OA publication path has $0 author fee."
            else:
                score = 60.0
                cost_tier = "Open Access (Unspecified APC)"
                reason = "Open Access venue; specific APC not listed in OpenAlex/DOAJ feed. Check journal homepage."

        results.append({
            "journal_id": journal["id"],
            "journal": journal.get("display_name", "Unknown Journal"),
            "score": score,
            "apc_usd": apc,
            "cost_tier": cost_tier,
            "reason": reason,
        })

    return {"cost_results": results}
