from typing import Any
import httpx


class CrossrefClient:
    """Client for the public Crossref REST API."""

    BASE_URL = "https://api.crossref.org"

    def __init__(self, email: str | None = None, timeout: float = 12.0):
        self.email = email
        self.timeout = timeout

    async def recent_works(
        self,
        issn: str,
        rows: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve recent publications for a journal by ISSN."""
        if not issn:
            return []

        clean_issn = issn.replace("-", "").strip()
        params = {
            "filter": "from-pub-date:2021-01-01",
            "rows": rows,
            "sort": "published",
            "order": "desc",
        }

        if self.email:
            params["mailto"] = self.email

        headers = {
            "User-Agent": (
                "AI-Journal-Recommendation-Assistant/1.0 (mailto:"
                + (self.email or "researcher@academic-assistant.ai")
                + ")"
            )
        }

        url = f"{self.BASE_URL}/journals/{clean_issn}/works"

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                headers=headers,
            ) as client:
                response = await client.get(url, params=params)
                if response.status_code != 200:
                    return []
                payload = response.json()

            items = payload.get("message", {}).get("items", [])
            results = []

            for item in items:
                title = item.get("title") or []
                if isinstance(title, list):
                    title = title[0] if title else ""

                authors = []
                for author in item.get("author", [])[:3]:
                    given = author.get("given", "")
                    family = author.get("family", "")
                    if family:
                        authors.append(f"{given} {family}".strip())

                doi = item.get("DOI", "")
                results.append({
                    "title": title,
                    "doi": doi,
                    "url": f"https://doi.org/{doi}" if doi else item.get("URL", ""),
                    "published": item.get("published-print", item.get("published", {})),
                    "created_date": item.get("created", {}).get("date-time"),
                    "authors": authors,
                    "is_referenced_by_count": item.get("is-referenced-by-count", 0),
                    "type": item.get("type"),
                    "source": "Crossref",
                })

            return results
        except Exception:
            return []