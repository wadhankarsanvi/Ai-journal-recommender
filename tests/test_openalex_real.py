import pytest

from journal_ai.data_sources.openalex import OpenAlexClient


@pytest.mark.asyncio
async def test_openalex_finds_real_journals():

    client = OpenAlexClient()

    journals = await client.search_sources(
        "deep learning fake news detection natural language processing",
        per_page=8,
    )

    print("\nREAL OPENALEX JOURNALS:")

    for journal in journals:
        print(
            journal["display_name"],
            "|",
            journal["source"],
        )

    assert len(journals) > 0

    assert all(
        journal["source"] == "OpenAlex"
        for journal in journals
    )
