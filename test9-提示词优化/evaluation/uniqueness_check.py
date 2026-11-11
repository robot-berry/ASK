"""Compute status totals over the 150 cases without double-counting overlapping reviews."""
import json, glob, os, collections

G = {os.path.basename(f)[:-5]: json.load(open(f, encoding="utf-8")) for f in glob.glob("evaluation/current_model_math150/gpt_reviews/*.json")}
M = {os.path.basename(f)[:-5]: json.load(open(f, encoding="utf-8")) for f in glob.glob("evaluation/current_model_math150/model_reviews/*.json")}


def tally(rows):
    return collections.Counter(r["status"] for r in rows.values())


dedup_gpt = {**M, **G}   # GPT wins on overlap
dedup_mdl = {**G, **M}   # model wins on overlap
print("overlap-preferred-gpt", tally(dedup_gpt))
print("overlap-preferred-model", tally(dedup_mdl))
diff = [(k, G[k]["status"], M[k]["status"]) for k in sorted(G.keys() & M) if G[k]["status"] != M[k]["status"]]
print("overlap disagreements", len(diff))
