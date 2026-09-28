import asyncio
import httpx
from journal_ai.data_sources.openalex import OpenAlexClient

async def debug_openalex():
    oa = OpenAlexClient()
    query = "medical image analysis brain tumor mri"

    async with httpx.AsyncClient() as client:
        # Test 1: works
        r1 = await client.get(f"https://api.openalex.org/works?search={query}&per-page=5")
        print("Works status:", r1.status_code)
        if r1.status_code == 200:
            works = r1.json().get("results", [])
            print(f"Found {len(works)} works")
            for w in works:
                loc = w.get("primary_location") or {}
                source = loc.get("source") or {}
                print("  Work source:", source.get("display_name"), "type:", source.get("type"))
        else:
            print("Works error:", r1.text[:200])

        # Test 2: sources direct
        r2 = await client.get(f"https://api.openalex.org/sources?search={query}&per-page=5&filter=type:journal")
        print("Sources status:", r2.status_code)
        if r2.status_code == 200:
            sources = r2.json().get("results", [])
            print(f"Found {len(sources)} sources")
            for s in sources:
                print("  Source:", s.get("display_name"))
        else:
            print("Sources error:", r2.text[:200])

if __name__ == "__main__":
    asyncio.run(debug_openalex())
