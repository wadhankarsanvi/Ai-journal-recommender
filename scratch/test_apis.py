import asyncio
import time
from journal_ai.data_sources.openalex import OpenAlexClient
from journal_ai.data_sources.crossref import CrossrefClient
from journal_ai.data_sources.doaj import DOAJClient

async def test_api_timings():
    oa = OpenAlexClient()
    cr = CrossrefClient()
    doaj = DOAJClient()

    query = "medical image analysis brain tumor mri"
    print("Testing OpenAlex search_sources...")
    t0 = time.perf_counter()
    sources = await oa.search_sources(query, per_page=4)
    t1 = time.perf_counter()
    print(f"OpenAlex search_sources took: {t1 - t0:.2f}s ({len(sources)} sources)")

    print("Testing Crossref recent_works...")
    t2 = time.perf_counter()
    works = await cr.recent_works("0278-0062", rows=3)
    t3 = time.perf_counter()
    print(f"Crossref recent_works took: {t3 - t2:.2f}s ({len(works)} works)")

    print("Testing DOAJ search_journal_by_issn...")
    t4 = time.perf_counter()
    d_info = await doaj.search_journal_by_issn("0278-0062")
    t5 = time.perf_counter()
    print(f"DOAJ search_journal_by_issn took: {t5 - t4:.2f}s")

if __name__ == "__main__":
    asyncio.run(test_api_timings())
