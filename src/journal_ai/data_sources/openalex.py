import asyncio
from typing import Any

import httpx


class OpenAlexClient:
    """
    Client for OpenAlex Academic Graph API.
    Provides comprehensive bibliographic data, source metadata, citation impact,
    open access status, APC fees, and topic taxonomy.
    """

    BASE_URL = "https://api.openalex.org"

    def __init__(self, email: str | None = None, timeout: float = 15.0):
        self.email = email
        self.timeout = timeout

    def _params(self, extra: dict[str, Any]) -> dict[str, Any]:
        params = dict(extra)
        if self.email:
            params["mailto"] = self.email
        return params

    async def search_works(
        self,
        query: str,
        per_page: int = 20,
    ) -> list[dict[str, Any]]:
        """Search scholarly works related to the research manuscript topic."""
        params = self._params({
            "search": query,
            "per-page": per_page,
            "sort": "relevance_score:desc",
        })

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.BASE_URL}/works",
                params=params,
            )
            response.raise_for_status()
            payload = response.json()

        return payload.get("results", [])

    async def search_sources_direct(
        self,
        query: str,
        per_page: int = 10,
    ) -> list[dict[str, Any]]:
        """Search journal and venue sources directly by keyword/topic."""
        params = self._params({
            "search": query,
            "per-page": per_page,
            "filter": "type:journal",
        })

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.BASE_URL}/sources",
                params=params,
            )
            if response.status_code != 200:
                return []
            payload = response.json()

        return [self._normalize_source(s) for s in payload.get("results", [])]

    async def get_source(
        self,
        source_id: str,
    ) -> dict[str, Any] | None:
        """Retrieve complete journal/source metadata by ID."""
        source_id = source_id.rstrip("/")
        source_key = source_id.split("/")[-1] if "/" in source_id else source_id

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.BASE_URL}/sources/{source_key}",
                params=self._params({}),
            )
            if response.status_code != 200:
                return None
            return response.json()

    async def search_sources(
        self,
        query: str,
        per_page: int = 8,
    ) -> list[dict[str, Any]]:
        """
        Hybrid search: Queries relevant works first to find actual publishing venues,
        complemented with direct source matching.
        """
        sources_dict: dict[str, dict[str, Any]] = {}

        # 1. Try finding sources via relevant works
        try:
            works = await self.search_works(query, per_page=min(20, per_page * 2))
            for work in works:
                loc = work.get("primary_location") or {}
                source = loc.get("source") or {}
                source_id = source.get("id")
                if source_id and source_id not in sources_dict and source.get("type") == "journal":
                    sources_dict[source_id] = {
                        "id": source_id,
                        "display_name": source.get("display_name", "Unknown Journal"),
                        "publisher": source.get("host_organization_name"),
                        "issn_l": source.get("issn_l"),
                        "is_oa": bool(source.get("is_oa", False)),
                        "is_in_doaj": bool(source.get("is_in_doaj", False)),
                        "sample_work_title": work.get("title", ""),
                        "sample_work_doi": work.get("doi", ""),
                    }
                if len(sources_dict) >= per_page:
                    break
        except Exception:
            pass

        # 2. If needed, supplement with direct source search
        if len(sources_dict) < per_page:
            try:
                direct_sources = await self.search_sources_direct(query, per_page=per_page)
                for ds in direct_sources:
                    if ds["id"] not in sources_dict:
                        sources_dict[ds["id"]] = ds
                    if len(sources_dict) >= per_page:
                        break
            except Exception:
                pass

        # 3. Retrieve full source metadata concurrently
        selected_sources = list(sources_dict.items())[:per_page]

        async def enrich_source(
            s_id: str,
            s_stub: dict[str, Any],
        ) -> dict[str, Any]:
            try:
                full_source = await self.get_source(s_id)
            except Exception:
                full_source = None

            if full_source:
                norm = self._normalize_source(full_source)

                if s_stub.get("sample_work_title"):
                    norm["sample_work_title"] = s_stub["sample_work_title"]
                    norm["sample_work_doi"] = s_stub["sample_work_doi"]

                return norm

            return self._normalize_source(s_stub)

        return await asyncio.gather(
            *[
                enrich_source(s_id, s_stub)
                for s_id, s_stub in selected_sources
            ]
        )

    @staticmethod
    def _normalize_source(item: dict[str, Any]) -> dict[str, Any]:
        ids = item.get("ids") or {}
        summary = item.get("summary_stats") or {}
        open_access = item.get("open_access") or {}
        apc_usd = item.get("apc_usd")
        if apc_usd is None and item.get("apc_prices"):
            prices = item.get("apc_prices")
            if isinstance(prices, list) and len(prices) > 0:
                apc_usd = prices[0].get("price")

        topics = []
        for topic in item.get("topics") or []:
            if isinstance(topic, dict):
                name = topic.get("display_name")
                if name:
                    topics.append(name)
            elif isinstance(topic, str):
                topics.append(topic)

        # Calculate estimated 2-yr citedness and percentile
        two_yr_mean = float(summary.get("2yr_mean_citedness") or 0.0)
        h_index = int(summary.get("h_index") or 0)
        i10_index = int(summary.get("i10_index") or 0)
        works_count = int(item.get("works_count") or 0)
        cited_by_count = int(item.get("cited_by_count") or 0)

        # Estimate impact percentile (0-100)
        impact_score = min(100.0, max(15.0, round(two_yr_mean * 22.0 + (h_index * 0.4), 1)))

        return {
            "id": item.get("id", "openalex-src"),
            "display_name": item.get("display_name") or "Unknown Academic Journal",
            "publisher": item.get("host_organization_name") or item.get("publisher") or "Academic Press",
            "issn_l": item.get("issn_l"),
            "issn": item.get("issn") or ([item.get("issn_l")] if item.get("issn_l") else []),
            "topics": topics[:8],
            "is_in_doaj": bool(item.get("is_in_doaj")),
            "is_oa": bool(open_access.get("is_oa", item.get("is_oa", False))),
            "apc_usd": apc_usd,
            "citedness_2yr_percentile": impact_score,
            "two_year_mean_citedness": round(two_yr_mean, 2),
            "h_index": h_index,
            "i10_index": i10_index,
            "works_count": works_count,
            "cited_by_count": cited_by_count,
            "country_code": item.get("country_code"),
            "homepage_url": item.get("homepage_url") or ids.get("issn_l"),
            "source": "OpenAlex",
            "raw": {
                "ids": ids,
                "homepage_url": item.get("homepage_url"),
                "works_count": works_count,
            },
        }