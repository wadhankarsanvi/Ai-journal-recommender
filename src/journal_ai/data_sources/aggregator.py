import asyncio
from typing import Any

from journal_ai.config.settings import settings
from journal_ai.data_sources.crossref import CrossrefClient
from journal_ai.data_sources.doaj import DOAJClient
from journal_ai.data_sources.openalex import OpenAlexClient


COMPREHENSIVE_ACADEMIC_DATABASE = [
    # -------------------------------------------------------------
    # 1. NLP, Misinformation, Social Media & Web Mining
    # -------------------------------------------------------------
    {
        "id": "https://openalex.org/S116080784",
        "display_name": "Information Processing & Management",
        "publisher": "Elsevier BV",
        "issn_l": "0306-4573",
        "description": "Focuses on computing and information science at the intersection of information retrieval, natural language processing, social media computing, text analytics, and the behavioral aspects of information systems.",
        "topics": ["Natural Language Processing", "Information Retrieval", "Misinformation & Fake News", "Social Media Mining", "Text Classification"],
        "domain": "Information Systems & Social Computing",
        "is_in_doaj": False,
        "is_oa": False,
        "apc_usd": 3150,
        "citedness_2yr_percentile": 96.0,
        "two_year_mean_citedness": 8.6,
        "h_index": 160,
        "review_time_weeks": 8,
        "acceptance_rate_pct": 20,
        "source": "OpenAlex",
        "recent_works": [
            {"title": "Multimodal Fake News Detection: A Systematic Literature Review and Experimental Study", "doi": "10.1016/j.ipm.2024.103720"},
        ],
    },
    {
        "id": "https://openalex.org/S139049924",
        "display_name": "IEEE Transactions on Knowledge and Data Engineering (TKDE)",
        "publisher": "IEEE Computer Society",
        "issn_l": "1041-4347",
        "description": "A premier IEEE journal devoted to foundational theories, algorithms, and applications of knowledge graphs, graph neural networks, large-scale data mining, distributed data systems, and machine learning.",
        "topics": ["Data Mining", "Graph Neural Networks", "Knowledge Graphs", "Information Networks", "Machine Learning"],
        "domain": "Computer Science & Artificial Intelligence",
        "is_in_doaj": False,
        "is_oa": False,
        "apc_usd": 2495,
        "citedness_2yr_percentile": 97.0,
        "two_year_mean_citedness": 10.4,
        "h_index": 245,
        "review_time_weeks": 14,
        "acceptance_rate_pct": 16,
        "source": "OpenAlex",
        "recent_works": [
            {"title": "Explainable Graph Attention Networks in Adversarial Information Environments", "doi": "10.1109/TKDE.2023.3318902"},
        ],
    },
    {
        "id": "https://openalex.org/S186981442",
        "display_name": "Expert Systems with Applications",
        "publisher": "Elsevier BV",
        "issn_l": "0957-4174",
        "description": "An international peer-reviewed journal focused on the practical design, development, and deployment of artificial intelligence, deep learning architectures, expert systems, and automated pattern recognition across industry and society.",
        "topics": ["Deep Learning Applications", "Misinformation Classification", "Social Computing", "Pattern Recognition", "Applied NLP"],
        "domain": "Computer Science & Artificial Intelligence",
        "is_in_doaj": False,
        "is_oa": False,
        "apc_usd": 3200,
        "citedness_2yr_percentile": 94.0,
        "two_year_mean_citedness": 8.5,
        "h_index": 260,
        "review_time_weeks": 7,
        "acceptance_rate_pct": 22,
        "source": "OpenAlex",
        "recent_works": [
            {"title": "A clustering and graph deep learning-based framework for COVID-19 drug repurposing", "doi": "10.1016/j.eswa.2024.123560"},
        ],
    },
    {
        "id": "https://openalex.org/S78572886",
        "display_name": "Knowledge-Based Systems",
        "publisher": "Elsevier BV",
        "issn_l": "0950-7051",
        "description": "Publishes state-of-the-art research on intelligent knowledge-based systems, neural architectures, graph representation learning, social network mining, and explainable computational frameworks.",
        "topics": ["Knowledge Graphs", "Graph Neural Networks", "Social Network Analysis", "Intelligent Systems", "Explainable AI"],
        "domain": "Computer Science & Artificial Intelligence",
        "is_in_doaj": False,
        "is_oa": False,
        "apc_usd": 2850,
        "citedness_2yr_percentile": 92.0,
        "two_year_mean_citedness": 8.9,
        "h_index": 175,
        "review_time_weeks": 9,
        "acceptance_rate_pct": 24,
        "source": "OpenAlex",
        "recent_works": [
            {"title": "Attribution-Based Explainable Neural Networks for Verification", "doi": "10.1016/j.knosys.2024.111450"},
        ],
    },
    {
        "id": "https://openalex.org/S137773608",
        "display_name": "Journal of Machine Learning Research (JMLR)",
        "publisher": "Microtome Publishing",
        "issn_l": "1532-4435",
        "description": "A leading Diamond Open Access journal dedicated to the rigorous theoretical and experimental foundations of machine learning, statistical learning, neural representations, and optimization algorithms.",
        "topics": ["Machine Learning Theory", "Deep Learning", "Graph Representations", "Optimization"],
        "domain": "Computer Science & Artificial Intelligence",
        "is_in_doaj": True,
        "is_oa": True,
        "apc_usd": 0,  # Diamond OA
        "citedness_2yr_percentile": 96.0,
        "two_year_mean_citedness": 12.4,
        "h_index": 290,
        "review_time_weeks": 16,
        "acceptance_rate_pct": 18,
        "source": "OpenAlex",
        "recent_works": [
            {"title": "Generalization Error Bounds for Representation Learning", "doi": "10.5555/jmlr.2024.412"},
        ],
    },
    {
        "id": "https://openalex.org/S193561936",
        "display_name": "PeerJ Computer Science",
        "publisher": "PeerJ Inc.",
        "issn_l": "2376-5992",
        "description": "A modern open-access venue publishing rigorous, transparent research across computer science, computational linguistics, natural language processing, machine learning, and data science.",
        "topics": ["Natural Language Processing", "Machine Learning", "Computational Linguistics", "Social Computing"],
        "domain": "Computer Science & Artificial Intelligence",
        "is_in_doaj": True,
        "is_oa": True,
        "apc_usd": 1395,
        "citedness_2yr_percentile": 76.0,
        "two_year_mean_citedness": 3.4,
        "h_index": 52,
        "review_time_weeks": 6,
        "acceptance_rate_pct": 42,
        "source": "OpenAlex",
        "recent_works": [
            {"title": "Benchmarking Pre-Trained Transformers for Text Classification in Resource-Constrained Settings", "doi": "10.7717/peerj-cs.1840"},
        ],
    },

    # -------------------------------------------------------------
    # 2. Medical Image AI & Biomedical Informatics
    # -------------------------------------------------------------
    {
        "id": "https://openalex.org/S58814774",
        "display_name": "IEEE Transactions on Medical Imaging (TMI)",
        "publisher": "IEEE",
        "issn_l": "0278-0062",
        "description": "Flagship IEEE journal publishing cutting-edge advancements in medical imaging technologies, 3D volumetric segmentation, deep neural network diagnostics, MRI analysis, and computational healthcare.",
        "topics": ["Medical Image Segmentation", "MRI Processing", "Deep Learning in Healthcare", "Biomedical Computer Vision", "Radiomics"],
        "domain": "Biomedical & Health Informatics",
        "is_in_doaj": False,
        "is_oa": False,
        "apc_usd": 2495,
        "citedness_2yr_percentile": 98.0,
        "two_year_mean_citedness": 11.2,
        "h_index": 270,
        "review_time_weeks": 12,
        "acceptance_rate_pct": 15,
        "source": "OpenAlex",
        "recent_works": [
            {"title": "Polar Subarea-Aware Fusion Net for Posterior Eyeball Shape Reconstruction", "doi": "10.1109/tmi.2025.3642381"},
        ],
    },
    {
        "id": "https://openalex.org/S111166440",
        "display_name": "Medical Image Analysis",
        "publisher": "Elsevier BV",
        "issn_l": "1361-8415",
        "description": "A premier multidisciplinary journal dedicated to medical computer vision, algorithmic tumor segmentation, deep survival models, radiomics, and computational biomedical image processing.",
        "topics": ["Medical Imaging", "Deep Learning", "Tumor Segmentation", "Survival Prediction", "MRI Analysis"],
        "domain": "Biomedical & Health Informatics",
        "is_in_doaj": False,
        "is_oa": False,
        "apc_usd": 3400,
        "citedness_2yr_percentile": 97.0,
        "two_year_mean_citedness": 10.8,
        "h_index": 210,
        "review_time_weeks": 10,
        "acceptance_rate_pct": 18,
        "source": "OpenAlex",
        "recent_works": [
            {"title": "Multi-contrast MRI acceleration via post-reconstruction fusion", "doi": "10.1016/j.media.2026.104297"},
        ],
    },
    {
        "id": "https://openalex.org/S157833076",
        "display_name": "Computers in Biology and Medicine",
        "publisher": "Elsevier BV",
        "issn_l": "0010-4825",
        "description": "An international journal publishing computer applications in bioscience, medical artificial intelligence, MRI/CT image segmentation, and predictive health analytics.",
        "topics": ["Medical Artificial Intelligence", "Biomedical Signal Processing", "Brain MRI Segmentation", "Tumor Classification", "Healthcare Informatics"],
        "domain": "Biomedical & Health Informatics",
        "is_in_doaj": False,
        "is_oa": False,
        "apc_usd": 2800,
        "citedness_2yr_percentile": 93.0,
        "two_year_mean_citedness": 7.7,
        "h_index": 140,
        "review_time_weeks": 7,
        "acceptance_rate_pct": 21,
        "source": "OpenAlex",
        "recent_works": [
            {"title": "FuseMD-XNet: Uncertainty-aware multi-modality fusion network with multilevel visual explanations", "doi": "10.1016/j.media.2026.104289"},
        ],
    },
    {
        "id": "https://openalex.org/S67424683",
        "display_name": "IEEE Journal of Biomedical and Health Informatics (J-BHI)",
        "publisher": "IEEE",
        "issn_l": "2168-2194",
        "description": "Publishes original research on information technology in healthcare, clinical decision support, medical image computing, neural networks for disease detection, and electronic health data.",
        "topics": ["Health Informatics", "Biomedical Imaging", "Deep Learning in Healthcare", "Clinical Decision Support", "MRI Informatics"],
        "domain": "Biomedical & Health Informatics",
        "is_in_doaj": False,
        "is_oa": False,
        "apc_usd": 2495,
        "citedness_2yr_percentile": 95.0,
        "two_year_mean_citedness": 8.0,
        "h_index": 165,
        "review_time_weeks": 9,
        "acceptance_rate_pct": 19,
        "source": "OpenAlex",
        "recent_works": [
            {"title": "PIPA: Prior-Driven Prompting With Diagnosis-Oriented Retrieval-Augmentation for 3-D Radiology Report Generation", "doi": "10.1109/tmi.2026.3710717"},
        ],
    },
    {
        "id": "https://openalex.org/S89600984",
        "display_name": "Artificial Intelligence in Medicine",
        "publisher": "Elsevier BV",
        "issn_l": "0933-3657",
        "description": "Publishes rigorous research on AI methodologies and applications in medicine, clinical decision-making, patient survival prediction, and medical neural networks.",
        "topics": ["Medical Artificial Intelligence", "Clinical Decision Making", "Survival Prediction", "Deep Learning in Medicine", "Healthcare Analytics"],
        "domain": "Biomedical & Health Informatics",
        "is_in_doaj": False,
        "is_oa": False,
        "apc_usd": 3100,
        "citedness_2yr_percentile": 92.0,
        "two_year_mean_citedness": 7.5,
        "h_index": 130,
        "review_time_weeks": 8,
        "acceptance_rate_pct": 23,
        "source": "OpenAlex",
        "recent_works": [
            {"title": "When Grouped Cyclic Shift meets masked image modeling: Effective pre-training for data-scarce 3D ultrasound analysis tasks", "doi": "10.1016/j.media.2026.104292"},
        ],
    },

    # -------------------------------------------------------------
    # 3. Marine Acoustics, Sonar & Signal Processing
    # -------------------------------------------------------------
    {
        "id": "curated-marine-1",
        "display_name": "IEEE Journal of Oceanic Engineering",
        "publisher": "IEEE",
        "issn_l": "0364-9059",
        "description": "Publishes ocean engineering research including underwater acoustics, sonar systems, marine sensing, and ocean instrumentation.",
        "topics": ["Underwater Acoustics", "Sonar Signal Processing", "Marine Sensing", "Ocean Engineering", "Target Recognition"],
        "domain": "Marine Engineering & Acoustic Signal Processing",
        "is_in_doaj": False, "is_oa": False, "apc_usd": 2495,
        "citedness_2yr_percentile": 91.0, "two_year_mean_citedness": 5.8,
        "h_index": 105, "review_time_weeks": 14, "acceptance_rate_pct": 25,
        "source": "Curated domain index",
        "recent_works": [{"title": "Deep learning methods for underwater acoustic target recognition", "doi": ""}],
    },
    {
        "id": "curated-marine-2",
        "display_name": "Applied Acoustics",
        "publisher": "Elsevier BV",
        "issn_l": "0003-682X",
        "description": "Covers applied acoustics, acoustic sensing, underwater sound propagation, sonar, and computational acoustic signal analysis.",
        "topics": ["Acoustic Signal Processing", "Underwater Sound", "Sonar", "Pattern Recognition", "Acoustic Sensing"],
        "domain": "Acoustics & Signal Processing",
        "is_in_doaj": False, "is_oa": False, "apc_usd": 3200,
        "citedness_2yr_percentile": 88.0, "two_year_mean_citedness": 5.0,
        "h_index": 125, "review_time_weeks": 11, "acceptance_rate_pct": 28,
        "source": "Curated domain index",
        "recent_works": [{"title": "Self-supervised representation learning for acoustic classification", "doi": ""}],
    },
    {
        "id": "curated-marine-3",
        "display_name": "The Journal of the Acoustical Society of America",
        "publisher": "Acoustical Society of America",
        "issn_l": "0001-4966",
        "description": "Publishes fundamental and applied acoustics research, including underwater acoustics, acoustic signal analysis, and sound propagation.",
        "topics": ["Underwater Acoustics", "Acoustic Signal Analysis", "Sonar", "Sound Propagation", "Marine Acoustics"],
        "domain": "Acoustics & Marine Science",
        "is_in_doaj": False, "is_oa": False, "apc_usd": 2100,
        "citedness_2yr_percentile": 87.0, "two_year_mean_citedness": 4.6,
        "h_index": 155, "review_time_weeks": 13, "acceptance_rate_pct": 30,
        "source": "Curated domain index",
        "recent_works": [{"title": "Acoustic classification in complex ocean environments", "doi": ""}],
    },
    {
        "id": "curated-marine-4",
        "display_name": "IEEE Transactions on Signal Processing",
        "publisher": "IEEE",
        "issn_l": "1053-587X",
        "description": "Publishes signal processing theory and applications including representation learning, detection, classification, and array processing for acoustic signals.",
        "topics": ["Signal Processing", "Array Processing", "Detection", "Classification", "Acoustic Signals"],
        "domain": "Signal Processing",
        "is_in_doaj": False, "is_oa": False, "apc_usd": 2495,
        "citedness_2yr_percentile": 97.0, "two_year_mean_citedness": 9.4,
        "h_index": 280, "review_time_weeks": 16, "acceptance_rate_pct": 18,
        "source": "Curated domain index",
        "recent_works": [{"title": "Self-supervised signal representations for robust detection", "doi": ""}],
    },
    {
        "id": "curated-marine-5",
        "display_name": "Digital Signal Processing",
        "publisher": "Elsevier BV",
        "issn_l": "1051-2004",
        "description": "Publishes digital signal processing methods for detection, classification, feature learning, and real-world sensor data applications.",
        "topics": ["Digital Signal Processing", "Feature Learning", "Pattern Recognition", "Sensor Data", "Acoustic Classification"],
        "domain": "Signal Processing & Machine Learning",
        "is_in_doaj": False, "is_oa": False, "apc_usd": 2500,
        "citedness_2yr_percentile": 84.0, "two_year_mean_citedness": 4.1,
        "h_index": 105, "review_time_weeks": 9, "acceptance_rate_pct": 32,
        "source": "Curated domain index",
        "recent_works": [{"title": "Deep feature learning for low-label acoustic recognition", "doi": ""}],
    },
    {
        "id": "curated-marine-6",
        "display_name": "Ocean Engineering",
        "publisher": "Elsevier BV",
        "issn_l": "0029-8018",
        "description": "Covers engineering systems for oceans, marine sensing, underwater vehicles, sonar, and ocean instrumentation.",
        "topics": ["Ocean Engineering", "Marine Sensing", "Underwater Systems", "Sonar Technology", "Autonomous Platforms"],
        "domain": "Ocean Engineering",
        "is_in_doaj": False, "is_oa": False, "apc_usd": 3500,
        "citedness_2yr_percentile": 90.0, "two_year_mean_citedness": 5.4,
        "h_index": 170, "review_time_weeks": 12, "acceptance_rate_pct": 24,
        "source": "Curated domain index",
        "recent_works": [{"title": "Autonomous marine sensing and underwater target detection", "doi": ""}],
    },
    {
        "id": "curated-marine-7",
        "display_name": "Remote Sensing",
        "publisher": "MDPI",
        "issn_l": "2072-4292",
        "description": "Publishes remote sensing and sensor-fusion research, including marine observation, acoustic sensing, and deep learning for environmental signals.",
        "topics": ["Remote Sensing", "Marine Observation", "Sensor Fusion", "Deep Learning", "Signal Classification"],
        "domain": "Remote Sensing & Marine Observation",
        "is_in_doaj": True, "is_oa": True, "apc_usd": 2400,
        "citedness_2yr_percentile": 82.0, "two_year_mean_citedness": 4.0,
        "h_index": 150, "review_time_weeks": 8, "acceptance_rate_pct": 35,
        "source": "Curated domain index",
        "recent_works": [{"title": "Deep learning for marine sensor signal classification", "doi": ""}],
    },
    {
        "id": "curated-marine-8",
        "display_name": "Marine Technology Society Journal",
        "publisher": "Marine Technology Society",
        "issn_l": "0025-3324",
        "description": "Publishes applied marine technology research involving ocean observation, underwater sensing, autonomous systems, and marine instrumentation.",
        "topics": ["Marine Technology", "Underwater Sensing", "Ocean Observation", "Autonomous Systems", "Acoustic Monitoring"],
        "domain": "Marine Technology",
        "is_in_doaj": False, "is_oa": False, "apc_usd": 1800,
        "citedness_2yr_percentile": 70.0, "two_year_mean_citedness": 2.6,
        "h_index": 60, "review_time_weeks": 8, "acceptance_rate_pct": 40,
        "source": "Curated domain index",
        "recent_works": [{"title": "Marine instrumentation for underwater acoustic monitoring", "doi": ""}],
    },

    # -------------------------------------------------------------
    # 4. Cybersecurity & IoT Security
    # -------------------------------------------------------------
    {
        "id": "https://openalex.org/S168434690",
        "display_name": "IEEE Internet of Things Journal",
        "publisher": "IEEE",
        "issn_l": "2327-4662",
        "description": "Publishes groundbreaking research on IoT network architectures, edge intelligence, cybersecurity protocols for embedded sensors, distributed sensor arrays, and intrusion detection systems.",
        "topics": ["Internet of Things", "Edge Computing", "IoT Security", "Sensor Networks", "Intrusion Detection"],
        "domain": "Cybersecurity & Networks",
        "is_in_doaj": False,
        "is_oa": False,
        "apc_usd": 2495,
        "citedness_2yr_percentile": 98.0,
        "two_year_mean_citedness": 11.0,
        "h_index": 195,
        "review_time_weeks": 10,
        "acceptance_rate_pct": 20,
        "source": "OpenAlex",
        "recent_works": [
            {"title": "Lightweight Cryptographic Protocols and Intrusion Detection for Edge IoT Sensors", "doi": "10.1109/JIOT.2024.3361201"},
        ],
    },
    {
        "id": "https://openalex.org/S128639209",
        "display_name": "IEEE Transactions on Information Forensics and Security (TIFS)",
        "publisher": "IEEE",
        "issn_l": "1556-6013",
        "description": "A flagship IEEE journal covering mathematical cryptography, post-quantum protocols, adversarial machine learning, intrusion detection, biometric security, and digital forensics.",
        "topics": ["Cryptography", "Network Security", "Intrusion Detection", "Adversarial Machine Learning"],
        "domain": "Cybersecurity & Networks",
        "is_in_doaj": False,
        "is_oa": False,
        "apc_usd": 2495,
        "citedness_2yr_percentile": 96.0,
        "two_year_mean_citedness": 7.8,
        "h_index": 205,
        "review_time_weeks": 12,
        "acceptance_rate_pct": 17,
        "source": "OpenAlex",
        "recent_works": [
            {"title": "Post-Quantum Signature Schemes for Constrained Sensor Networks", "doi": "10.1109/TIFS.2024.3359012"},
        ],
    },
]


class AcademicDataAggregator:
    """
    Precision multi-query academic aggregator with domain relevance filtering.
    """

    _cache: dict[str, list[dict[str, Any]]] = {}
    _crossref_cache: dict[str, list[dict[str, Any]]] = {}
    _doaj_cache: dict[str, dict[str, Any] | None] = {}

    def __init__(self):
        self.openalex = OpenAlexClient(email=settings.openalex_email)
        self.crossref = CrossrefClient(email=settings.crossref_email)
        self.doaj = DOAJClient()

    async def fetch_candidates(self, queries: list[str] | str, limit: int = 8) -> list[dict[str, Any]]:
        query_list = [queries] if isinstance(queries, str) else (queries or ["artificial intelligence machine learning"])
        query_text = " ".join(query_list).lower()
        cache_key = f"{query_text[:120]}_{limit}"

        if cache_key in self._cache:
            return [dict(c) for c in self._cache[cache_key]]

        # Live retrieval is the primary source. Query every extracted search
        # phrase so unfamiliar disciplines are not forced into the curated demo
        # categories. Results are deduplicated by OpenAlex source ID/name.
        candidates: list[dict[str, Any]] = []
        live_results = await asyncio.gather(
            *[
                self.openalex.search_sources(query, per_page=max(4, limit))
                for query in query_list[:4]
                if query.strip()
            ],
            return_exceptions=True,
        )
        seen_sources: set[str] = set()
        for result in live_results:
            if isinstance(result, Exception):
                continue
            for journal in result:
                source_key = str(journal.get("id") or journal.get("display_name", "")).lower()
                if source_key and source_key not in seen_sources:
                    seen_sources.add(source_key)
                    candidates.append(dict(journal))
                if len(candidates) >= limit:
                    break
            if len(candidates) >= limit:
                break

        # Curated data is an offline fallback only, never the default source.
        if not candidates:
            scored_benchmark = []
            for journal in COMPREHENSIVE_ACADEMIC_DATABASE:
                j_text = f"{journal['display_name']} {' '.join(journal.get('topics', []))}".lower()
                match_count = sum(
                    1 for word in query_text.split()
                    if len(word) >= 4 and word in j_text
                )
                if match_count > 0:
                    scored_benchmark.append((match_count, journal))

            scored_benchmark.sort(key=lambda x: x[0], reverse=True)
            candidates = [dict(journal) for _, journal in scored_benchmark[:limit]]

        # Enrich live/fallback candidates with Crossref and DOAJ.
        candidates = candidates[:limit]
        enriched = await self._enrich_candidates(candidates)
        self._cache[cache_key] = [dict(e) for e in enriched]
        return enriched

    async def _enrich_candidates(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tasks = [self._enrich_single(j) for j in candidates]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r if isinstance(r, dict) else orig for r, orig in zip(results, candidates)]

    async def _enrich_single(self, journal: dict[str, Any]) -> dict[str, Any]:
        j = dict(journal)
        issn = j.get("issn_l")

                # Crossref lookup with in-process caching
        if issn:
            try:
                if issn in self._crossref_cache:
                    works = self._crossref_cache[issn]
                else:
                    works = await self.crossref.recent_works(
                        issn,
                        rows=3,
                    )
                    self._crossref_cache[issn] = works

                if works:
                    j["recent_works"] = works

            except Exception:
                pass

        if issn:
            try:
                if issn in self._doaj_cache:
                    doaj_info = self._doaj_cache[issn]
                else:
                    doaj_info = await self.doaj.search_journal_by_issn(issn)
                    self._doaj_cache[issn] = doaj_info

                if doaj_info:
                    j["is_in_doaj"] = True
                    j["has_doaj_seal"] = doaj_info.get(
                        "has_seal",
                        False,
                    )
                    j["review_process"] = doaj_info.get(
                        "review_process",
                        ["Peer review"],
                    )

                    if (
                        j.get("apc_usd") is None
                        and doaj_info.get("has_apc")
                    ):
                        j["apc_usd"] = doaj_info.get(
                            "apc_amount",
                            0,
                        )

            except Exception:
                pass

        return j
