#!/usr/bin/env python3
"""
rezultati.py — izvuče rezultate iz već odrađenih treninga.
────────────────────────────────────────────────────────────────────────────
Ne trenira ništa. Prođe kroz checkpoints/<model>/ mape, nađe checkpointe i
logove, pa napiše:

    checkpoints/SAZETAK.csv        — jedan redak po modelu, s najboljim mIoU
    checkpoints/<model>/krivulja.csv — mIoU po epohama (za graf u radu)
    checkpoints/<model>/checkpoints.txt — svi checkpointi, najbolji na vrhu

UPORABA (iz korijena DFormera)
    python rezultati.py
    python rezultati.py --mapa checkpoints
    python rezultati.py --cisti        # obriši sve osim najboljeg checkpointa
"""

import argparse
import csv
import os
import re
import sys

# "epoch-165_miou_88.24.pth"
RE_CKPT = re.compile(r"miou_([0-9]+\.?[0-9]*)\.pth$")
RE_EPOHA_CKPT = re.compile(r"epoch-([0-9]+)_")
# "Epoch 200 validation result: mIoU 85.34, best mIoU 85.54"
RE_VAL = re.compile(r"Epoch\s+(\d+)\s+validation result:\s*mIoU\s+([0-9.]+)"
                    r"(?:,\s*best mIoU\s+([0-9.]+))?", re.I)
# "miou 85.34 best 85.54"
RE_VAL2 = re.compile(r"\bmiou\s+([0-9.]+)\s+best\s+([0-9.]+)", re.I)
# iz INFO.txt
RE_INFO = {
    "pocetak": re.compile(r"^Početak:\s*(.+)$", re.M),
    "trajanje": re.compile(r"^Trajanje:\s*(\d+)", re.M),
    "status": re.compile(r"^Status:\s*(\S+)", re.M),
}


def nadji_checkpointe(mapa):
    """Rekurzivno: [(miou, epoha, putanja)], sortirano silazno po miou."""
    nadjeni = []
    for koren, _, datoteke in os.walk(mapa):
        for f in datoteke:
            m = RE_CKPT.search(f)
            if not m:
                continue
            e = RE_EPOHA_CKPT.search(f)
            nadjeni.append((float(m.group(1)),
                            int(e.group(1)) if e else -1,
                            os.path.join(koren, f)))
    return sorted(nadjeni, key=lambda t: -t[0])


def citaj_logove(mapa):
    """Vraća [(epoha, miou, best)] iz svih .log datoteka u mapi."""
    tocke = []
    for koren, _, datoteke in os.walk(mapa):
        for f in sorted(datoteke):
            if not f.endswith(".log"):
                continue
            put = os.path.join(koren, f)
            try:
                with open(put, errors="replace") as fh:
                    for redak in fh:
                        m = RE_VAL.search(redak)
                        if m:
                            tocke.append((int(m.group(1)), float(m.group(2)),
                                          float(m.group(3)) if m.group(3) else None))
                            continue
                        m = RE_VAL2.search(redak)
                        if m and not RE_VAL.search(redak):
                            tocke.append((None, float(m.group(1)), float(m.group(2))))
            except OSError:
                pass
    # Isti redci znaju postojati u dva loga (tee + DFormerov vlastiti), pa se
    # duplikati miču: po epohi, zadnji upis pobjeđuje.
    s_epohom = [t for t in tocke if t[0] is not None]
    if s_epohom:
        po_epohi = {}
        for e, m, b in s_epohom:
            po_epohi[e] = (m, b)
        return [(e, po_epohi[e][0], po_epohi[e][1]) for e in sorted(po_epohi)]
    return tocke


def citaj_info(mapa):
    put = os.path.join(mapa, "INFO.txt")
    podaci = {}
    if os.path.exists(put):
        try:
            tekst = open(put, errors="replace").read()
            for k, rx in RE_INFO.items():
                m = rx.search(tekst)
                if m:
                    podaci[k] = m.group(1).strip()
        except OSError:
            pass
    return podaci


def main():
    p = argparse.ArgumentParser(description="Sažmi rezultate odrađenih treninga")
    p.add_argument("--mapa", default="checkpoints",
                   help="mapa s podmapama modela (zadano: checkpoints)")
    p.add_argument("--cisti", action="store_true",
                   help="obriši sve checkpointe osim najboljeg po modelu")
    args = p.parse_args()

    korijen = os.path.abspath(args.mapa)
    if not os.path.isdir(korijen):
        print(f"Nema mape {korijen}")
        sys.exit(1)

    modeli = sorted(
        d for d in os.listdir(korijen)
        if os.path.isdir(os.path.join(korijen, d))
        and "_staro_" not in d
        and not d.startswith("pretrained")
    )
    if not modeli:
        print(f"Nema mapa modela u {korijen}")
        sys.exit(1)

    redci = []
    oslobodjeno = 0

    for model in modeli:
        mapa = os.path.join(korijen, model)
        ckpt = nadji_checkpointe(mapa)
        krivulja = citaj_logove(mapa)
        info = citaj_info(mapa)

        najbolji = ckpt[0] if ckpt else None

        # checkpoints.txt
        with open(os.path.join(mapa, "checkpoints.txt"), "w") as fh:
            for miou, epoha, put in ckpt:
                fh.write(f"{miou:8.2f}  epoha {epoha:>4}  "
                         f"{os.path.relpath(put, korijen)}\n")

        # krivulja.csv
        if krivulja:
            with open(os.path.join(mapa, "krivulja.csv"), "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["epoha", "miou", "najbolji_do_sad"])
                for e, m, b in krivulja:
                    w.writerow([e if e is not None else "", m,
                                b if b is not None else ""])

        # najbolji iz loga (može biti viši od onog u imenu datoteke)
        log_best = max((b for _, _, b in krivulja if b is not None), default=None)
        log_zadnji = krivulja[-1][1] if krivulja else None

        redci.append({
            "model": model,
            "pocetak": info.get("pocetak", ""),
            "trajanje_min": info.get("trajanje", ""),
            "status": info.get("status", ""),
            "najbolji_miou": f"{najbolji[0]:.2f}" if najbolji else "",
            "najbolji_epoha": najbolji[1] if najbolji else "",
            "miou_zadnja_epoha": f"{log_zadnji:.2f}" if log_zadnji else "",
            "best_iz_loga": f"{log_best:.2f}" if log_best else "",
            "broj_checkpointa": len(ckpt),
            "epoha_u_logu": len(krivulja),
            "najbolji_checkpoint": os.path.relpath(najbolji[2], korijen) if najbolji else "",
        })

        if args.cisti and len(ckpt) > 1:
            for _, _, put in ckpt[1:]:
                try:
                    oslobodjeno += os.path.getsize(put)
                    os.remove(put)
                except OSError:
                    pass

    # SAZETAK.csv
    izlaz = os.path.join(korijen, "SAZETAK.csv")
    polja = list(redci[0].keys())
    with open(izlaz, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=polja)
        w.writeheader()
        w.writerows(redci)

    # Ispis, sortiran po mIoU
    poredak = sorted(redci,
                     key=lambda r: float(r["najbolji_miou"] or 0), reverse=True)
    print()
    print(f"{'model':<16}{'mIoU':>8}{'epoha':>7}{'zadnja':>9}"
          f"{'ckpt':>6}{'epoha u logu':>14}{'min':>7}")
    print("─" * 70)
    for r in poredak:
        print(f"{r['model']:<16}{r['najbolji_miou']:>8}{str(r['najbolji_epoha']):>7}"
              f"{r['miou_zadnja_epoha']:>9}{r['broj_checkpointa']:>6}"
              f"{r['epoha_u_logu']:>14}{r['trajanje_min']:>7}")
    print()
    print(f"Sažetak:  {izlaz}")
    print(f"Krivulje: {korijen}/<model>/krivulja.csv")
    if args.cisti:
        print(f"Oslobođeno: {oslobodjeno / 1e9:.2f} GB")
    bez = [r["model"] for r in redci if not r["najbolji_miou"]]
    if bez:
        print(f"\nBez pronađenog checkpointa: {', '.join(bez)}")


if __name__ == "__main__":
    main()