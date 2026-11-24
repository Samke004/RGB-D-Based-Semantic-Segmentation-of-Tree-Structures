#!/usr/bin/env python3
"""
ugradi_branch.py
────────────────────────────────────────────────────────────────────────────
Registrira BRANCH skup u sluzbeni ESANet repo. Mijenja tri datoteke:

  src/datasets/__init__.py   -> izvozi Branch
  src/prepare_data.py        -> grana za --dataset branch
  src/args.py                -> dodaje 'branch' u dopustene izbore

Prije toga rucno kopiraj mapu `branch/` (s branch.py, pytorch_dataset.py,
__init__.py) u src/datasets/.

Svaka izmjena se radi samo ako vec nije napravljena, pa je skriptu sigurno
pokrenuti dvaput. Originali se spremaju kao *.prije_branch.

Usporaba:
    cd ~/Desktop/ESANetOriginal          # korijen njihovog repoa
    python ugradi_branch.py
"""

import os
import shutil
import sys

GRANA = """    if args.dataset == 'branch':
        Dataset = Branch
        dataset_kwargs = {'n_classes': 5}
        valid_set = 'valid'
    elif args.dataset == 'sunrgbd':"""


def zakrpaj(put, stara, nova, oznaka):
    if not os.path.exists(put):
        sys.exit(f"Nema {put} — jesi li u korijenu ESANet repoa?")
    t = open(put).read()
    if oznaka in t:
        print(f"  [vec ugradeno] {put}")
        return False
    if stara not in t:
        sys.exit(f"Ne nalazim ocekivani tekst u {put}:\n  {stara[:70]}...")
    if not os.path.exists(put + ".prije_branch"):
        shutil.copy(put, put + ".prije_branch")
    open(put, "w").write(t.replace(stara, nova, 1))
    print(f"  [ugradeno]     {put}")
    return True


def main():
    if not os.path.isdir("src/datasets"):
        sys.exit("Pokreni iz korijena ESANet repoa (mora postojati src/datasets).")

    if not os.path.isdir("src/datasets/branch"):
        sys.exit("Nema src/datasets/branch/ — prvo ondje kopiraj branch.py, "
                 "pytorch_dataset.py i __init__.py")

    for f in ("branch.py", "pytorch_dataset.py", "__init__.py"):
        p = os.path.join("src/datasets/branch", f)
        if not os.path.exists(p):
            sys.exit(f"Nedostaje {p}")

    print("Ugradnja BRANCH skupa:")

    zakrpaj("src/datasets/__init__.py",
            "from .sunrgbd.pytorch_dataset import SUNRGBD",
            "from .sunrgbd.pytorch_dataset import SUNRGBD\n"
            "from .branch.pytorch_dataset import Branch",
            "from .branch.pytorch_dataset import Branch")

    # uvoz u prepare_data: izravno iz podmodula, da ne povlaci Cityscapes
    # (on trazi cityscapesscripts, koje mozda nije instalirano)
    zakrpaj("src/prepare_data.py",
            "from src.datasets import SUNRGBD",
            "from src.datasets import SUNRGBD\n"
            "from src.datasets.branch.pytorch_dataset import Branch",
            "from src.datasets.branch.pytorch_dataset import Branch")

    zakrpaj("src/prepare_data.py",
            "    if args.dataset == 'sunrgbd':",
            GRANA,
            "args.dataset == 'branch'")

    zakrpaj("src/args.py",
            "choices=['sunrgbd',",
            "choices=['branch',\n                                   'sunrgbd',",
            "'branch',")

    print("\nGotovo. Provjera:")
    print("  python -c \"import sys; sys.path.insert(0,'.'); "
          "from src.datasets.branch.pytorch_dataset import Branch; "
          "d=Branch(data_dir='<PUT_DO_BranchDataset>', split='train'); "
          "print(len(d), d.class_names, d.depth_mean, d.depth_std)\"")


if __name__ == "__main__":
    main()
