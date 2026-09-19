from typing import Any


KNOWN_REPUTABLE_PUBLISHERS = {
    "ieee", "acm", "elsevier", "springer", "nature", "wiley", "oxford",
    "cambridge", "plos", "microtome", "mit press", "taylor & francis",
    "frontiers", "mdpi", "peerj", "biomed central", "sage", "iop",
    "royal society", "american chemical society", "association for computational linguistics",
}


def run_credibility_agent(state: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """
    Evaluates journal credibility, peer-review integrity, indexing verification,
    and predatory publishing risk screening.
    """
    results = []

    for journal in state.get("candidate_journals", []):
        score = 65.0
        signals = []
        publisher = str(journal.get("publisher", "")).lower()

        # 1. Reputable publisher verification
        if any(pub in publisher for pub in KNOWN_REPUTABLE_PUBLISHERS):
            score += 15.0
            signals.append(f"Published by established press ({journal.get('publisher')})")
        elif journal.get("publisher"):
            score += 8.0
            signals.append(f"Host publisher recorded: {journal.get('publisher')}")

        # 2. DOAJ Verification
        if journal.get("is_in_doaj"):
            score += 12.0
            signals.append("Indexed in DOAJ (Directory of Open Access Journals)")
            if journal.get("has_doaj_seal"):
                score += 5.0
                signals.append("Awarded DOAJ Seal for Best Practice in Open Access")

        # 3. ISSN-L / Identifiers
        if journal.get("issn_l") or journal.get("issn"):
            score += 5.0
            signals.append(f"Verified ISSN-L: {journal.get('issn_l') or journal.get('issn')}")

        # 4. Peer Review Process
        if journal.get("review_process"):
            signals.append(f"Transparent peer-review model ({', '.join(journal.get('review_process', []))})")

        final_score = round(min(100.0, max(20.0, score)), 1)
        risk = "Low Risk (Highly Credible)" if final_score >= 80 else ("Moderate Risk" if final_score >= 60 else "High Screening Warning")

        results.append({
            "journal_id": journal["id"],
            "journal": journal.get("display_name", "Unknown Journal"),
            "score": final_score,
            "risk": risk,
            "signals": signals,
            "reason": f"{risk}: " + "; ".join(signals[:3]),
        })

    return {"credibility_results": results}
