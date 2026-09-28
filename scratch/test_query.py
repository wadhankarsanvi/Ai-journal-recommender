import asyncio
import httpx

async def test_search():
    query = "deep learning fake news detection natural language processing"
    words = [w for w in query.split() if len(w) > 3]
    print("Words:", words)

    async with httpx.AsyncClient() as client:
        # Full query
        r1 = await client.get(f"https://api.openalex.org/sources?search={query}&filter=type:journal")
        print("Full query count:", len(r1.json().get("results", [])))

        # 2-3 key terms
        subquery = " ".join(words[:3])
        r2 = await client.get(f"https://api.openalex.org/sources?search={subquery}&filter=type:journal")
        res2 = r2.json().get("results", [])
        print(f"Subquery '{subquery}' count:", len(res2))
        for s in res2[:3]:
            print(" -", s.get("display_name"))

if __name__ == "__main__":
    asyncio.run(test_search())
