from typing import Any
import httpx


class DOAJClient:
    """
    Client for Directory of Open Access Journals (DOAJ) public API v2.
    Provides verified information on:
    - Open Access status and DOAJ Seal of approval
    - Article Processing Charges (APC) and waiver policies
    - Peer review models (double-blind, single-blind, open peer review)
    - CC Licensing and copyright retention
    """

    BASE_URL = "https://doaj.org/api/v2"

    def __init__(self, timeout: float = 12.0):
        self.timeout = timeout

    async def search_journal_by_issn(self, issn: str) -> dict[str, Any] | None:
        """Search journal verification details by ISSN or ISSN-L."""
        if not issn:
            return None

        clean_issn = issn.replace("-", "").strip()
        url = f"{self.BASE_URL}/search/journals/issn:{clean_issn}"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(url)
                if resp.status_code != 200:
                    return None
                data = resp.json()
                results = data.get("results", [])
                if not results:
                    return None

                item = results[0]
                bib = item.get("bibjson", {})

                # APC parsing
                apc_info = bib.get("apc", {})
                has_apc = apc_info.get("has_apc", False)
                max_apc = apc_info.get("max", [])
                apc_amount = 0
                apc_currency = "USD"
                if max_apc and isinstance(max_apc, list) and len(max_apc) > 0:
                    apc_amount = max_apc[0].get("price", 0)
                    apc_currency = max_apc[0].get("currency", "USD")

                # Editorial / Review process
                editorial = bib.get("editorial", {})
                review_process = editorial.get("review_process", ["Peer review"])

                # Licensing & Seal
                licenses = [lic.get("type") for lic in bib.get("license", []) if isinstance(lic, dict)]
                has_seal = bib.get("oa_seal", False)

                return {
                    "is_doaj_indexed": True,
                    "has_seal": bool(has_seal),
                    "has_apc": bool(has_apc),
                    "apc_amount": apc_amount,
                    "apc_currency": apc_currency,
                    "review_process": review_process,
                    "licenses": licenses,
                    "waiver_url": bib.get("waiver_policy", {}).get("url") if bib.get("waiver_policy") else None,
                    "homepage": bib.get("journal_url"),
                }
        except Exception as exc:
            # DOAJ is supplementary, gracefully return None on timeout/network issue
            return None

    async def search_journal_by_title(self, title: str) -> dict[str, Any] | None:
        """Search DOAJ by journal title."""
        if not title:
            return None

        url = f"{self.BASE_URL}/search/journals/{title}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(url, params={"pageSize": 1})
                if resp.status_code != 200:
                    return None
                data = resp.json()
                results = data.get("results", [])
                if not results:
                    return None
                item = results[0]
                bib = item.get("bibjson", {})
                apc_info = bib.get("apc", {})
                return {
                    "is_doaj_indexed": True,
                    "has_seal": bool(bib.get("oa_seal", False)),
                    "has_apc": bool(apc_info.get("has_apc", False)),
                    "apc_amount": apc_info.get("max", [{}])[0].get("price", 0) if apc_info.get("max") else 0,
                    "apc_currency": apc_info.get("max", [{}])[0].get("currency", "USD") if apc_info.get("max") else "USD",
                    "licenses": [l.get("type") for l in bib.get("license", []) if isinstance(l, dict)],
                }
        except Exception:
            return None
