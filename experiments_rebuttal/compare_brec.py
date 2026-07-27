"""
Compare BREC per-pair JSON outputs (gin / ppgn / topo) from brec_expressivity.py
and report the headline numbers -- robustly, without trusting a category map
(this dump is an 800-pair variant, not canonical BREC-400).

Key, category-independent evidence for "expressiveness beyond 3-WL":
  * the 3-WL-hard pairs are exactly the ones PPGN FAILS,
  * the gain is how many of THOSE pairs PPGN+PH (topo) distinguishes.

  python experiments_rebuttal/compare_brec.py \
      rebuttal_results/brec/gin.json \
      rebuttal_results/brec/ppgn.json \
      rebuttal_results/brec/topo.json
"""
import sys
import json


def load(path):
    with open(path) as f:
        d = json.load(f)
    # index records by unique-pair id
    return d["model"], {r["pair"]: r for r in d["records"]}


def size_bucket(n):
    if n <= 16:   return "small (n<=16: Basic/Regular)"
    if n <= 40:   return "medium (17-40)"
    return "large (n>40: Extension/CFI)"


def main():
    paths = sys.argv[1:]
    if not paths:
        print("usage: compare_brec.py <gin.json> <ppgn.json> <topo.json> ...")
        sys.exit(1)
    models = {}
    for p in paths:
        name, rec = load(p)
        models[name] = rec
    names = list(models)
    # common set of pairs evaluated by all models
    common = set.intersection(*[set(r) for r in models.values()])
    common = sorted(common)
    print(f"models: {names}")
    print(f"pairs evaluated by all: {len(common)}\n")

    # (1) totals
    print(f"{'model':<10}{'distinguished / total':>26}{'rate':>10}")
    for m in names:
        d = sum(models[m][p]["distinguished"] for p in common)
        print(f"{m:<10}{f'{d} / {len(common)}':>26}{f'{100*d/max(1,len(common)):.1f}%':>10}")

    # (2) headline: among pairs PPGN fails, how many does topo get?
    if "ppgn" in models and "topo" in models:
        ppgn, topo = models["ppgn"], models["topo"]
        hard = [p for p in common if not ppgn[p]["distinguished"]]   # 3-WL-hard
        gained = [p for p in hard if topo[p]["distinguished"]]
        lost = [p for p in common
                if ppgn[p]["distinguished"] and not topo[p]["distinguished"]]
        print(f"\n=== beyond-3-WL (category-independent) ===")
        print(f"3-WL-hard pairs (PPGN fails):            {len(hard)}")
        print(f"  of which PPGN+PH distinguishes:        {len(gained)}"
              f"   <-- beyond-3-WL GAIN")
        print(f"pairs PPGN got but PPGN+PH lost (regress): {len(lost)}")
        if gained:
            print(f"  gained pair ids: {gained[:40]}"
                  + (" ..." if len(gained) > 40 else ""))

        # (3) by graph-size bucket (proxy for category)
        print(f"\n=== by graph size (proxy for category) ===")
        buckets = {}
        for p in common:
            n = ppgn[p].get("n", 0)
            b = size_bucket(n)
            buckets.setdefault(b, [])
            buckets[b].append(p)
        hdr = f"{'bucket':<32}" + "".join(f"{m:>10}" for m in names) + f"{'topo>ppgn':>12}"
        print(hdr)
        for b in sorted(buckets):
            ps = buckets[b]
            row = f"{b:<32}"
            for m in names:
                d = sum(models[m][p]["distinguished"] for p in ps)
                row += f"{f'{d}/{len(ps)}':>10}"
            if "ppgn" in models and "topo" in models:
                g = sum(1 for p in ps if topo[p]["distinguished"]
                        and not ppgn[p]["distinguished"])
                row += f"{g:>12}"
            print(row)


if __name__ == "__main__":
    main()
