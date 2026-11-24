#!/usr/bin/env python3
"""
dformer_u_chen.py

Pretvara DFormer skup (RGB/ + Depth/ + Label/ + train.txt/val.txt/test.txt)
u format koji jede Chen pipeline (Pokusaj_Chen_MultiDepth):

    data/<split>/images/<ime>_rgbd.npy   float32 (H, W, 4), RGB u 0..1, depth u metrima
    data/<split>/masks/<ime>_mask.png    uint8 (H, W), indeksi klasa

Podjela se NE radi iznova - preuzima se doslovno iz .txt datoteka DFormer skupa,
da bi usporedba s DFormer rezultatima bila poštena.

Primjer:
    python3 dformer_u_chen.py --dformer ~/Desktop/BranchDataset --izlaz data_novo
    python3 dformer_u_chen.py --dformer ~/Desktop/BranchDataset --izlaz data_novo --provjeri data
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image

KLASE = ["ostalo", "deblo", "grane", "potpora", "trava"]


# ─── pomocne ────────────────────────────────────────────────────────────────

def ucitaj_popis(put):
    """Cita train.txt/val.txt/test.txt. Podnosi i 'RGB/a.png Label/a.png' i goli 'a.png' i 'a'."""
    stavke = []
    with open(put) as fh:
        for red in fh:
            red = red.strip()
            if not red:
                continue
            prvi = red.split()[0]
            stem = os.path.splitext(os.path.basename(prvi))[0]
            stavke.append(stem)
    return stavke


def nadji(mapa, stem):
    """Nalazi datoteku po korijenu imena, bez obzira na nastavak."""
    for nastavak in (".png", ".jpg", ".jpeg", ".PNG", ".JPG"):
        put = os.path.join(mapa, stem + nastavak)
        if os.path.exists(put):
            return put
    return None


def odredi_skalu(depth_mapa, stavke, zadano=None):
    """Gleda nekoliko depth datoteka i zakljucuje jesu li milimetri ili metri."""
    if zadano is not None:
        return zadano, f"rucno zadano ({zadano})"

    maksimumi, dtypes = [], set()
    for stem in stavke[:30]:
        put = nadji(depth_mapa, stem)
        if put is None:
            continue
        a = np.array(Image.open(put))
        dtypes.add(str(a.dtype))
        if a.ndim == 3:
            a = a[..., 0]
        maksimumi.append(float(a.max()))

    if not maksimumi:
        raise SystemExit("Nijedna depth datoteka nije procitana - provjeri --dformer putanju.")

    vrh = max(maksimumi)
    if vrh > 100:
        return 1000.0, f"milimetri (max {vrh:.0f}, dtype {'/'.join(sorted(dtypes))})"
    return 1.0, f"metri (max {vrh:.3f}, dtype {'/'.join(sorted(dtypes))})"


# ─── glavna pretvorba ───────────────────────────────────────────────────────

def pretvori_split(korijen, split, izlaz, skala, broj_klasa):
    txt = os.path.join(korijen, f"{split}.txt")
    if not os.path.exists(txt):
        print(f"  [{split}] nema {split}.txt - preskacem")
        return None

    stavke = ucitaj_popis(txt)
    img_izlaz = os.path.join(izlaz, split, "images")
    msk_izlaz = os.path.join(izlaz, split, "masks")
    os.makedirs(img_izlaz, exist_ok=True)
    os.makedirs(msk_izlaz, exist_ok=True)

    rgb_mapa = os.path.join(korijen, "RGB")
    dep_mapa = os.path.join(korijen, "Depth")
    lab_mapa = os.path.join(korijen, "Label")

    piksela_po_klasi = np.zeros(broj_klasa, dtype=np.int64)
    napisano, preskoceno = 0, []
    oblik = None

    for stem in stavke:
        p_rgb, p_dep, p_lab = nadji(rgb_mapa, stem), nadji(dep_mapa, stem), nadji(lab_mapa, stem)
        if None in (p_rgb, p_dep, p_lab):
            fali = [n for n, p in zip(("RGB", "Depth", "Label"), (p_rgb, p_dep, p_lab)) if p is None]
            preskoceno.append((stem, "fali " + "+".join(fali)))
            continue

        rgb = np.array(Image.open(p_rgb).convert("RGB")).astype(np.float32) / 255.0

        dep = np.array(Image.open(p_dep))
        if dep.ndim == 3:
            dep = dep[..., 0]
        dep = dep.astype(np.float32) / skala

        lab = np.array(Image.open(p_lab))
        if lab.ndim == 3:
            lab = lab[..., 0]
        lab = lab.astype(np.uint8)

        if rgb.shape[:2] != dep.shape[:2] or rgb.shape[:2] != lab.shape[:2]:
            preskoceno.append((stem, f"nejednake dimenzije {rgb.shape[:2]}/{dep.shape[:2]}/{lab.shape[:2]}"))
            continue

        if oblik is None:
            oblik = rgb.shape[:2]

        nevazece = lab[(lab >= broj_klasa) & (lab != 255)]
        if nevazece.size:
            preskoceno.append((stem, f"indeksi klasa izvan 0..{broj_klasa-1}: {np.unique(nevazece)[:5]}"))
            continue

        rgbd = np.concatenate([rgb, dep[..., None]], axis=2).astype(np.float32)
        np.save(os.path.join(img_izlaz, f"{stem}_rgbd.npy"), rgbd)
        Image.fromarray(lab).save(os.path.join(msk_izlaz, f"{stem}_mask.png"))

        vazeci = lab[lab != 255]
        piksela_po_klasi += np.bincount(vazeci.ravel(), minlength=broj_klasa)[:broj_klasa]
        napisano += 1

    return {
        "split": split,
        "trazeno": len(stavke),
        "napisano": napisano,
        "preskoceno": preskoceno,
        "piksela": piksela_po_klasi,
        "oblik": oblik,
    }


# ─── provjera prema starom skupu ────────────────────────────────────────────

def provjeri(novi, stari):
    """Usporeduje raspon vrijednosti novih npy-eva sa starima, da se uhvati kriva skala."""
    print("\n=== PROVJERA PREMA STAROM SKUPU ===")

    def uzorak(korijen):
        import glob
        putovi = sorted(glob.glob(os.path.join(korijen, "train", "images", "*_rgbd.npy")))[:20]
        if not putovi:
            return None
        rgb_max, d_max, d_mean, d_nula = [], [], [], []
        for p in putovi:
            a = np.load(p)
            rgb_max.append(float(a[..., :3].max()))
            d = a[..., 3]
            d_max.append(float(d.max()))
            d_mean.append(float(d[d > 0].mean()) if (d > 0).any() else 0.0)
            d_nula.append(float((d == 0).mean()))
        return {
            "n": len(putovi),
            "rgb_max": max(rgb_max),
            "d_max": max(d_max),
            "d_mean": float(np.mean(d_mean)),
            "d_nula": float(np.mean(d_nula)),
        }

    s, n = uzorak(stari), uzorak(novi)
    if s is None or n is None:
        print("  Nema dovoljno datoteka za usporedbu.")
        return

    print(f"  {'':16} {'staro':>12} {'novo':>12}")
    print(f"  {'RGB max':16} {s['rgb_max']:>12.3f} {n['rgb_max']:>12.3f}")
    print(f"  {'depth max (m)':16} {s['d_max']:>12.3f} {n['d_max']:>12.3f}")
    print(f"  {'depth mean>0':16} {s['d_mean']:>12.3f} {n['d_mean']:>12.3f}")
    print(f"  {'praznih %':16} {100*s['d_nula']:>12.1f} {100*n['d_nula']:>12.1f}")

    problemi = []
    if n["rgb_max"] > 1.01:
        problemi.append("RGB nije u 0..1 - provjeri izvorne RGB PNG-ove")
    if s["d_max"] > 0 and not (0.2 < n["d_max"] / s["d_max"] < 5):
        problemi.append(f"depth je {n['d_max']/s['d_max']:.1f}x drugaciji - skala je vjerojatno kriva")

    if problemi:
        print("\n  UPOZORENJE:")
        for p in problemi:
            print(f"    - {p}")
    else:
        print("\n  Rasponi se slazu sa starim skupom.")


# ─── ulaz ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="DFormer skup -> Chen pipeline format")
    ap.add_argument("--dformer", required=True, help="korijen DFormer skupa (sadrzi RGB/, Depth/, Label/, *.txt)")
    ap.add_argument("--izlaz", default="data_novo", help="izlazna mapa (zadano: data_novo)")
    ap.add_argument("--klase", type=int, default=5, help="broj klasa (zadano: 5)")
    ap.add_argument("--skala-dubine", type=float, default=None,
                    help="djelitelj dubine; bez njega se detektira (1000 za mm, 1 za m)")
    ap.add_argument("--provjeri", metavar="STARI_DATA", default=None,
                    help="usporedi rezultat sa starom data/ mapom")
    args = ap.parse_args()

    korijen = os.path.expanduser(args.dformer)
    if not os.path.isdir(korijen):
        sys.exit(f"Nema mape: {korijen}")

    for pod in ("RGB", "Depth", "Label"):
        if not os.path.isdir(os.path.join(korijen, pod)):
            sys.exit(f"Nema podmape {pod}/ u {korijen} - je li ovo korijen DFormer skupa?")

    izlaz = os.path.expanduser(args.izlaz)
    if os.path.exists(izlaz) and os.listdir(izlaz):
        sys.exit(f"{izlaz} vec postoji i nije prazna. Makni je ili zadaj drugi --izlaz.")

    svi = []
    for s in ("train", "val", "test"):
        txt = os.path.join(korijen, f"{s}.txt")
        if os.path.exists(txt):
            svi += ucitaj_popis(txt)

    skala, opis = odredi_skalu(os.path.join(korijen, "Depth"), svi, args.skala_dubine)
    print(f"Dubina: {opis} -> dijelim s {skala:g}")
    print(f"Izlaz:  {izlaz}\n")

    rezultati = []
    for split in ("train", "val", "test"):
        r = pretvori_split(korijen, split, izlaz, skala, args.klase)
        if r:
            rezultati.append(r)
            print(f"  [{r['split']:5}] {r['napisano']:4} / {r['trazeno']:4} slika"
                  + (f"   PRESKOCENO {len(r['preskoceno'])}" if r["preskoceno"] else ""))

    # preklapanje splitova - curenje podataka
    print("\n=== PREKLAPANJE SPLITOVA ===")
    skupovi = {}
    for s in ("train", "val", "test"):
        mapa = os.path.join(izlaz, s, "images")
        if os.path.isdir(mapa):
            skupovi[s] = {f.replace("_rgbd.npy", "") for f in os.listdir(mapa)}
    curi = False
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if a in skupovi and b in skupovi:
            p = skupovi[a] & skupovi[b]
            if p:
                curi = True
                print(f"  CURENJE {a}/{b}: {len(p)} zajednickih - {sorted(p)[:5]}")
    if not curi:
        print("  Nema preklapanja.")

    # raspodjela klasa
    print("\n=== PIKSELI PO KLASI ===")
    imena = KLASE[:args.klase] if args.klase <= len(KLASE) else [f"klasa_{i}" for i in range(args.klase)]
    print(f"  {'split':6} " + " ".join(f"{n:>10}" for n in imena))
    ukupno = np.zeros(args.klase, dtype=np.int64)
    for r in rezultati:
        uk = r["piksela"].sum()
        ukupno += r["piksela"]
        udjeli = 100 * r["piksela"] / max(uk, 1)
        print(f"  {r['split']:6} " + " ".join(f"{v:>9.2f}%" for v in udjeli))
    uk = ukupno.sum()
    print(f"  {'UKUPNO':6} " + " ".join(f"{v:>9.2f}%" for v in 100 * ukupno / max(uk, 1)))

    prazne = [imena[i] for i in range(args.klase) if ukupno[i] == 0]
    if prazne:
        print(f"\n  UPOZORENJE: klase bez ijednog piksela: {', '.join(prazne)}")
        print("  Provjeri jesu li indeksi u Label/ PNG-ovima doista 0..{}".format(args.klase - 1))

    # preskoceno
    sva_preskocena = [(r["split"], s, z) for r in rezultati for s, z in r["preskoceno"]]
    if sva_preskocena:
        print(f"\n=== PRESKOCENO ({len(sva_preskocena)}) ===")
        for split, stem, zasto in sva_preskocena[:20]:
            print(f"  [{split}] {stem}: {zasto}")
        if len(sva_preskocena) > 20:
            print(f"  ... i jos {len(sva_preskocena) - 20}")
        print("\n  Ovo mora biti prazno prije treniranja - inace skup nije isti kao DFormerov.")

    oblici = {r["oblik"] for r in rezultati if r["oblik"]}
    print(f"\nDimenzije slika: {oblici}")
    print(f"Gotovo. Sljedece: u config.py postavi DATA_DIR = \"{os.path.basename(izlaz)}\", "
          f"NUM_CLASSES = {args.klase}")

    if args.provjeri:
        provjeri(izlaz, os.path.expanduser(args.provjeri))

    if sva_preskocena or curi:
        print("\nNIJE CISTO - rijesi gornja upozorenja prije treniranja.")
        sys.exit(1)


if __name__ == "__main__":
    main()
