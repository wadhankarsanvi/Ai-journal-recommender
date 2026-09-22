from typing import Any


def run_scope_agent(state: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """
    Evaluates scope alignment with strict domain qualification.
    Ensures journals in matching academic subfields receive high scores, while out-of-domain venues are penalized.
    """
    profile = state.get("paper_profile", {})
    keywords = [str(k).lower() for k in profile.get("keywords", [])]
    domain = str(profile.get("domain", "")).lower()
    subfields = [str(s).lower() for s in profile.get("subfields", [])]
    title = str(profile.get("title", "")).lower()

    results = []

    # Domain Flags
    is_medical_imaging = any(w in domain or w in title for w in ["mri", "tumor", "medical", "biomedical", "brain", "glioma", "segmentation", "radiomics", "radiology", "clinical"])
    is_nlp_social = any(w in domain or w in title for w in ["nlp", "misinformation", "fake news", "social network", "language", "disinformation"])
    is_cyber_iot = any(w in domain or w in title for w in ["iot", "security", "cryptography", "intrusion", "quantum", "sensor"])
    is_marine_acoustics = any(w in domain or w in title or w in " ".join(keywords) for w in ["underwater", "acoustic", "sonar", "marine", "ocean", "hydrophone"])
    is_agriculture_environment = any(w in domain or w in title or w in " ".join(keywords) for w in ["agricultur", "farm", "plant", "crop", "soil", "microbiome", "drought", "irrigation", "environment", "ecology", "carbon", "sustainable", "sustainability", "rural", "land use", "remote sensing"])

    for journal in state.get("candidate_journals", []):
        j_name = str(journal.get("display_name", "")).lower()
        topics = [str(t).lower() for t in (journal.get("topics") or [])]
        j_text = f"{j_name} {' '.join(topics)} {journal.get('domain', '')}".lower()

        is_relevant = False
        reasons = []

        if is_medical_imaging:
            medical_terms = [
                "medical imaging", "medical image analysis", "biomedical", "biology and medicine",
                "health informatics", "mri", "radiology", "radiomics", "neuroimage", "cancer",
                "clinical informatics", "tumor", "healthcare", "medical artificial intelligence"
            ]
            if any(term in j_text for term in medical_terms):
                is_relevant = True
                reasons.append("Direct focus on medical imaging, biomedical computing, and clinical AI")
            else:
                is_relevant = False
                reasons.append("Non-biomedical venue; lacks clinical and medical imaging scope")

        elif is_nlp_social:
            nlp_terms = ["information processing", "knowledge and data", "expert systems", "knowledge-based", "social network", "natural language", "computational linguistics", "web mining", "information systems"]
            if any(term in j_text for term in nlp_terms):
                is_relevant = True
                reasons.append("Premier venue for NLP, social computing, and knowledge engineering")
            elif any(cs in j_text for cs in ["machine learning", "artificial intelligence", "data mining"]):
                is_relevant = True
                reasons.append("Core machine learning and data science scope")
            else:
                is_relevant = not any(term in j_text for term in ["medicine", "biomedical", "clinical", "cancer"])
                reasons.append("Adjacent information or computing venue from live topic retrieval")

        elif is_cyber_iot:
            cyber_terms = ["internet of things", "information forensics", "security", "cryptography", "privacy", "dependable", "sensor", "networks"]
            if any(term in j_text for term in cyber_terms):
                is_relevant = True
                reasons.append("Dedicated venue for IoT, cybersecurity, and cryptography")
            elif any(cs in j_text for cs in ["machine learning", "artificial intelligence", "computer science"]):
                is_relevant = True
                reasons.append("Core computing venue")
            else:
                is_relevant = not any(term in j_text for term in ["medicine", "biomedical", "clinical", "cancer"])
                reasons.append("Adjacent engineering or sensing venue from live topic retrieval")

        elif is_marine_acoustics:
            acoustic_terms = [
                "acoustic", "underwater", "sonar", "ocean", "marine", "hydrophone",
                "oceanic engineering", "marine sensing", "marine observation",
            ]
            if any(term in j_text for term in acoustic_terms):
                is_relevant = True
                reasons.append("Relevant venue for underwater acoustics, sonar, marine sensing, or signal processing")
            else:
                is_relevant = not any(term in j_text for term in ["medicine", "biomedical", "clinical", "health informatics"])
                reasons.append("Adjacent engineering or signal venue from live topic retrieval")

        elif is_agriculture_environment:
            agriculture_terms = [
                "plant", "crop", "agricultur", "farm", "soil", "microbiome", "drought", "water",
                "irrigation", "environment", "ecology", "sustainability", "sustainable", "carbon",
                "climate", "rural", "land use", "policy", "economics", "food", "nutrition", "remote sensing",
            ]
            if any(term in j_text for term in agriculture_terms):
                is_relevant = True
                reasons.append("Relevant venue for agriculture, farming, environmental policy, sustainability, or ecology")
            else:
                is_relevant = not any(term in j_text for term in ["medicine", "biomedical", "clinical", "cancer", "cryptography"])
                reasons.append("Adjacent environmental or life-science venue from live topic retrieval")

        else:
            # Unknown domains still need observable vocabulary overlap; prestige
            # alone must not turn an unrelated venue into a recommendation.
            generic_terms = [
                "machine learning", "artificial intelligence", "computer science",
                "data science", "signal processing", "engineering", "sensing",
                "remote sensing", "optimization", "statistics",
            ]
            keyword_hits = sum(1 for kw in keywords if len(kw) >= 4 and kw in j_text)
            is_relevant = keyword_hits > 0 or any(term in j_text for term in generic_terms if term in " ".join(keywords))
            if is_relevant:
                reasons.append("Venue vocabulary overlaps the manuscript's extracted research terms")
            else:
                reasons.append("No meaningful vocabulary overlap with the manuscript")

        if not is_relevant:
            score = 15.0
            reasons = ["Out-of-scope disciplinary focus for this manuscript."]
        else:
            match_hits = sum(1 for kw in keywords if len(kw) >= 4 and kw in j_text)
            subfield_hits = sum(1 for sf in subfields if any(w in j_text for w in sf.split()))
            score = min(98.0, 75.0 + (match_hits * 4.0) + (subfield_hits * 5.0))

        results.append({
            "journal_id": journal["id"],
            "journal": journal.get("display_name", "Unknown Journal"),
            "score": round(score, 1),
            "is_relevant": is_relevant,
            "matched_terms": [k for k in keywords if len(k) >= 4 and k in j_text][:6],
            "reason": "; ".join(reasons),
        })

    return {"scope_results": results}
