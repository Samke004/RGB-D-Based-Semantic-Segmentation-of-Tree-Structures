import re, random, sys
from pathlib import Path
from collections import defaultdict

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
UDJELI = (0.70, 0.15, 0.15)

slike = sorted(p.stem for p in Path("RGB").glob("*.png"))
stabla = defaultdict(list)
for s in slike:
    m = re.search(r"Drvo(\d+)", s)
    if not m:
        sys.exit(f"ne mogu iščitati stablo iz: {s}")
    stabla[m.group(1)].append(s)

print(f"{len(slike)} slika, {len(stabla)} stabala")

ids = sorted(stabla)
random.Random(SEED).shuffle(ids)

uk = len(slike)
cilj = [UDJELI[0]*uk, UDJELI[1]*uk, UDJELI[2]*uk]
skupovi = [[], [], []]
brojevi = [0, 0, 0]

# najveća stabla prva, svako ide u skup koji je najdalje od svog cilja
for tid in sorted(ids, key=lambda t: -len(stabla[t])):
    i = max(range(3), key=lambda k: cilj[k] - brojevi[k])
    skupovi[i].append(tid)
    brojevi[i] += len(stabla[tid])

for naziv, grupa in zip(("train", "val", "test"), skupovi):
    redci = sorted(s for t in grupa for s in stabla[t])
    Path(f"{naziv}.txt").write_text("\n".join(redci) + "\n")
    print(f"{naziv:6} {len(grupa):3} stabala  {len(redci):4} slika  ({100*len(redci)/uk:.1f}%)")

# provjera
presjek = set(skupovi[0]) & set(skupovi[1]) | set(skupovi[0]) & set(skupovi[2]) | set(skupovi[1]) & set(skupovi[2])
print("PREKLAPANJE STABALA:", presjek if presjek else "nema")