import json, sys
from pathlib import Path
D = Path(sys.argv[1]); lab = json.load(open(D / "labels.json")); g = json.load(open(D / sys.argv[2]))
thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
from collections import Counter
keep = {k for k, a in g.items() if a["is_parking_spot"] and a["is_free"] and a["confidence"] >= thr}
c_all, c_keep = Counter(lab.values()), Counter(lab[k] for k in keep)
print("sin IA :", dict(c_all), f"precisión {c_all['T']/len(lab):.0%} (contando dudosos como buenos {(c_all['T']+c_all['A'])/len(lab):.0%})")
print("con IA :", dict(c_keep), f"de {len(keep)} mostrados; precisión {c_keep['T']/max(1,len(keep)):.0%} (con dudosos {(c_keep['T']+c_keep['A'])/max(1,len(keep)):.0%})")
print("verdaderos perdidos:", [k for k in lab if lab[k] == "T" and k not in keep])
tok = [a["usage"].get("totalTokenCount", 0) for a in g.values()]; lat = [a["latency"] for a in g.values()]
print(f"tokens medios {sum(tok)/len(tok):.0f}, latencia media {sum(lat)/len(lat):.1f} s")
