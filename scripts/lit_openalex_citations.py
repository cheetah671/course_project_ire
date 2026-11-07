"""Citing papers (OpenAlex) for the seed papers; writes one JSON per seed and a merged keyword-filtered list."""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(sys.argv[1])
SEEDS = {
    "wang2024_best_practices": "Searching for Best Practices in Retrieval-Augmented Generation",
    "autorag": "AutoRAG: Automated Framework for optimization of Retrieval Augmented Generation Pipeline",
    "autorag_hp": "AutoRAG-HP: Automatic Online Hyper-Parameter Tuning for Retrieval-Augmented Generation",
    "orbach2025_hpo": "An Analysis of Hyper-Parameter Optimization Methods for Retrieval Augmented Generation",
    "adaptive_rag": "Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity",
    "metis": "METIS: Fast Quality-Aware RAG Systems with Configuration Adaptation",
}
UA = {"User-Agent": "ragroute-litcheck (course project)"}


def get(url):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            print("retry", attempt, e, file=sys.stderr)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(url)


KEY = re.compile(r"adaptiv|rout|per-query|query-level|configur|cost|efficien|latency|pareto|budget|chunk|hyper-?param|"
                 r"autom|tuning|complexity|select|trade-?off|when to retrieve|benchmark", re.I)
merged = {}
for key, title in SEEDS.items():
    res = get("https://api.openalex.org/works?search=" + urllib.parse.quote(title) + "&per-page=5")["results"]
    seed = next((w for w in res if w["display_name"].lower().startswith(title.split(":")[0].lower()[:30])), res[0] if res else None)
    if seed is None:
        print(key, "NOT FOUND"); continue
    wid = seed["id"].rsplit("/", 1)[1]
    citing, n = [], 1
    while True:  # page numbers (cursor paging returned 401 without a key)
        page = get(f"https://api.openalex.org/works?filter=cites:{wid}&per-page=200&page={n}"
                   "&select=id,display_name,publication_year,primary_location,ids,cited_by_count")
        citing += page["results"]
        if len(page["results"]) < 200 or n >= 10:
            break
        n += 1
        time.sleep(1)
    print(f"{key}: seed '{seed['display_name']}' ({wid}, cited_by={seed.get('cited_by_count')}) -> {len(citing)} citing")
    (OUT / f"oa_{key}.json").write_text(json.dumps(citing))
    for w in citing:
        if w["display_name"] and KEY.search(w["display_name"]):
            m = merged.setdefault(w["id"], {"title": w["display_name"], "year": w["publication_year"],
                                            "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
                                            "cited_by": w.get("cited_by_count"), "cites_seeds": []})
            m["cites_seeds"].append(key)
rows = sorted(merged.values(), key=lambda m: (-len(m["cites_seeds"]), -(m["cited_by"] or 0)))
(OUT / "oa_filtered.json").write_text(json.dumps(rows, indent=1))
print(len(rows), "filtered candidates")
