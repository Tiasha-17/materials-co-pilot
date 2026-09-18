import arxiv

search = arxiv.Search(
    query='cat:cond-mat.mtrl-sci AND all:"high entropy alloy" AND all:mechanical',
    #query='cat:cond-mat.mtrl-sci AND all:"high entropy alloy"',
    #query="high entropy alloys mechanical properties",
    max_results=5,
    sort_by=arxiv.SortCriterion.Relevance
)

client = arxiv.Client()

results = client.results(search)

#for result in results:
    #print(result.title)

for result in results:
    print("\nTITLE:")
    print(result.title)

    print("\nURL:")
    print(result.entry_id)

    print("\nABSTRACT:")
    print(result.summary[:500])

    print("\n" + "-" * 80)