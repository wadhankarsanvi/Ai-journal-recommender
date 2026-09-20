import re
from typing import Any
from journal_ai.orchestration.llm_client import LLMClient


async def extract_manuscript_profile_async(paper_text: str) -> dict[str, Any]:
    """
    Extracts structured academic profile and high-precision search queries.
    Utilizes active LLM (OpenAI/Gemini/Groq) if available, with intelligent heuristic fallback.
    """
    cleaned_text = paper_text.strip()
    llm = LLMClient()

    # 1. If LLM is available, use LLM for deep academic extraction
    if llm.is_llm_available():
        system_prompt = (
            "You are an expert academic research advisor and bibliometrics specialist. "
            "Analyze the given research paper manuscript draft and return a strict JSON object."
        )
        prompt = f"""
Analyze this manuscript draft:
\"\"\"{cleaned_text[:4000]}\"\"\"

Return a JSON object with EXACTLY these keys:
- "title": (string) Extracted precise title of the manuscript.
- "abstract": (string) Summary of the core objective, method, and findings.
- "domain": (string) Primary scholarly discipline. Options: "Biomedical & Medical Imaging", "Computer Science & Artificial Intelligence", "Information Systems & Social Computing", "Cybersecurity & Networks", "Engineering & Applied Sciences".
- "subfields": (list of strings) 3-5 specific subdisciplines (e.g., ["Medical Image Segmentation", "Brain MRI Analysis", "Radiomics", "Oncology AI"]).
- "methodology": (string) Methodology type (e.g., "3D Multi-Scale Attention UNet & Multi-Task Cox Survival Prediction").
- "keywords": (list of strings) 6-10 specific technical keyphrases (e.g. ["brain tumor", "mri segmentation", "radiomics", "brats", "glioma"]).
- "academic_search_queries": (list of strings) 3 targeted 3-5 word search queries to find the most specific journals in OpenAlex/Crossref (e.g., ["medical image analysis mri segmentation", "ieee transactions medical imaging brain tumor", "computers in biology and medicine"]).
"""
        llm_profile = await llm.generate_json(prompt, system_prompt=system_prompt)
        if llm_profile and "title" in llm_profile and "academic_search_queries" in llm_profile:
            llm_profile["word_count"] = len(cleaned_text.split())
            return llm_profile

    # 2. Heuristic High-Precision Extraction Fallback
    return extract_manuscript_profile_heuristic(cleaned_text)


def extract_manuscript_profile_heuristic(cleaned_text: str) -> dict[str, Any]:
    """Rule-based and semantic n-gram extraction of manuscript metadata."""
    words = cleaned_text.split()
    lines = [line.strip() for line in cleaned_text.split("\n") if line.strip()]

    # Title extraction
    title = lines[0] if lines else "Untitled Manuscript"
    if len(title.split()) > 30:
        title = " ".join(title.split()[:20]) + "..."

    # Abstract extraction
    abstract = ""
    abstract_match = re.search(
        r"(?:abstract|summary)\s*[:\-\n]\s*(.*?)(?=\n\s*(?:keywords|introduction|1\.|i\.)|$)",
        cleaned_text,
        re.IGNORECASE | re.DOTALL,
    )
    if abstract_match:
        abstract = abstract_match.group(1).strip()[:1500]
    elif len(lines) > 1:
        abstract = " ".join(lines[1:6])[:1500]
    else:
        abstract = cleaned_text[:1000]

    text_lower = cleaned_text.lower()

    # Topic & Domain Classification with High Sensitivity
    domain = "Computer Science & Artificial Intelligence"
    subfields = []
    queries = []

    # 1. Medical Imaging / Brain MRI / Oncology AI
    if any(k in text_lower for k in ["mri", "tumor", "glioma", "segmentation", "medical image", "radiology", "biomedical", "brats", "lesion", "brain", "ct scan", "histopathology", "cancer"]):
        domain = "Biomedical & Medical Imaging"
        subfields = ["Medical Image Analysis", "Brain MRI Segmentation", "Radiomics & Oncology AI", "Computer-Assisted Radiology"]
        queries = [
            "medical image analysis brain tumor mri",
            "ieee transactions on medical imaging segmentation",
            "computers in biology and medicine deep learning",
        ]
    # 2. NLP / Misinformation / Social Media
    elif any(k in text_lower for k in ["misinformation", "fake news", "rumor", "fact-checking", "disinformation", "nlp", "multilingual", "linguistic", "social network"]):
        domain = "Information Systems & Social Computing"
        subfields = ["Misinformation Detection", "Natural Language Processing", "Social Network Mining", "Knowledge Engineering"]
        queries = [
            "information processing and management fake news",
            "ieee transactions on knowledge and data engineering",
            "expert systems with applications misinformation",
        ]
    # 3. Cybersecurity / IoT
    elif any(k in text_lower for k in ["cryptography", "post-quantum", "intrusion detection", "iot", "security", "ddos", "sensor network", "malware", "privacy"]):
        domain = "Cybersecurity & Networks"
        subfields = ["IoT Security", "Post-Quantum Cryptography", "Intrusion Detection", "Network Forensics"]
        queries = [
            "ieee internet of things journal security",
            "ieee transactions on information forensics and security",
            "computers and security intrusion detection",
        ]
    # 4. Underwater acoustics / marine signal processing
    elif any(k in text_lower for k in ["underwater acoustic", "acoustic target", "sonar", "hydrophone", "marine acoustic", "underwater sound", "ocean acoustic"]):
        domain = "Marine Engineering & Acoustic Signal Processing"
        subfields = ["Underwater Acoustic Signal Processing", "Sonar Target Recognition", "Self-Supervised Representation Learning", "Marine Sensing"]
        queries = [
            "underwater acoustic signal processing target recognition",
            "sonar image processing marine sensing",
            "IEEE Journal of Oceanic Engineering acoustic recognition",
        ]
    # 5. Computer Vision
    elif any(k in text_lower for k in ["vision", "object detection", "image synthesis", "yolo", "pose estimation"]):
        domain = "Computer Vision & Pattern Recognition"
        subfields = ["Computer Vision", "Visual Representation Learning", "Pattern Recognition"]
        queries = [
            "ieee transactions on pattern analysis and machine intelligence",
            "pattern recognition computer vision deep learning",
        ]
    else:
        # Preserve the manuscript's actual vocabulary for unfamiliar domains.
        # Fixed AI queries caused unrelated biomedical venues to be returned.
        specific_terms = [
            word for word in keywords_from_text(cleaned_text)
            if word not in {"learning", "research", "data", "method", "model"}
        ][:8]
        query_seed = " ".join(specific_terms) or title
        queries = [
            query_seed[:120],
            f"{query_seed[:80]} journal",
        ]

    # Keyword extraction
    keywords = keywords_from_text(cleaned_text)[:10]

    return {
        "title": title,
        "abstract": abstract,
        "word_count": len(words),
        "domain": domain,
        "subfields": subfields,
        "methodology": "Empirical Neural Architecture & Experimental Benchmark Evaluation",
        "core_contribution": "Proposes a deep learning architecture with quantitative empirical benchmarking.",
        "keywords": keywords,
        "academic_search_queries": queries,
    }


def keywords_from_text(text: str) -> list[str]:
    """Extract stable content words for fallback search and profiling."""
    stopwords = {
        "about", "above", "after", "again", "against", "all", "also", "and", "any", "are",
        "because", "been", "before", "being", "below", "between", "both", "but", "by", "could",
        "did", "does", "doing", "down", "during", "each", "few", "for", "from", "further",
        "had", "has", "have", "having", "here", "how", "into", "its", "itself", "just", "more",
        "most", "other", "our", "over", "own", "same", "should", "some", "such", "than", "that",
        "the", "their", "them", "then", "these", "they", "this", "those", "through", "until", "very",
        "was", "were", "what", "when", "where", "which", "while", "who", "why", "with", "would",
        "using", "based", "paper", "study", "results", "proposed", "method", "methods", "approach",
        "model", "models", "show", "shows", "framework", "novel", "system", "performance", "analysis",
        "evaluation", "aims", "aim", "for", "learning",
    }
    return list(dict.fromkeys(
        word for word in (re.sub(r"[^\w\-]", "", value).lower() for value in text.split())
        if len(word) >= 4 and word not in stopwords
    ))


extract_manuscript_profile = extract_manuscript_profile_heuristic
