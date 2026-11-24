#!/usr/bin/env python3
"""
globalni_miou.py
────────────────────────────────────────────────────────────────────────────
Racuna IoU na testnom skupu na DVA nacina i stavlja ih jedan uz drugi:

  PO SLICI   - IoU se racuna za svaku sliku zasebno pa se uzme prosjek.
               Tako radi postojeci metrics.py / aggregate_metrics.
               Zamka: kad klase nema ni u istini ni u predikciji,
               per_class_iou vrati 0.0 (jer je 0/(0+1e-10)), pa svaka
               slika bez te klase upisuje nulu u prosjek. Rijetke klase
               time ispadaju sustavno podcijenjene.

  GLOBALNO   - jedna matrica zabune preko svih slika, IoU tek na kraju.
               Tako racunaju DFormer, mmsegmentation, Cityscapes i vecina
               radova. Svaki piksel nosi jednaku tezinu.

Brojke iz rada treba uzeti iz GLOBALNOG stupca, i samo su one usporedive
s DFormerovima.

Usporaba:
    python globalni_miou.py
    python globalni_miou.py --modeli esanet_resnet34 segnet_vgg16bn
    python globalni_miou.py --csv usporedba_globalno.csv
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config as cfg
from dataset import AppleTreeDataset
from augmentations import get_val_transforms
from metrics import CLASS_NAMES


# ─── registar modela ────────────────────────────────────────────────────────
# ime -> (funkcija koja gradi model, ime checkpointa, kljuc u checkpointu)

def _registar():
    from models import (build_unet, build_deeplabv3, build_deeplabv3plus,
                        build_segformer)
    r = {}

    r["unet_resnet34"] = (
        lambda: build_unet("resnet34", weights=None,
                           in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
        "unet_resnet34_best.pth", "model_state")

    r["unet_efficientnet_b4"] = (
        lambda: build_unet("efficientnet-b4", weights=None,
                           in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
        "unet_efficientnet_b4_best.pth", "model_state")

    r["deeplabv3plus_resnet50"] = (
        lambda: build_deeplabv3(pretrained=False,
                                in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
        "deeplabv3plus_resnet50_best.pth", "model_state")

    r["deeplabv3plus_mobilenetv2"] = (
        lambda: build_deeplabv3plus(encoder="mobilenet_v2", weights=None,
                                    in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
        "deeplabv3plus_mobilenetv2_best.pth", "model_state")

    r["segformer_b1"] = (
        lambda: build_segformer(variant=cfg.SEGFORMER_VARIANT, weights=None,
                                in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
        "segformer_b1_best.pth", "model_state")

    try:
        from models import build_segnet
        r["segnet_vgg16bn"] = (
            lambda: build_segnet(in_channels=cfg.IN_CHANNELS,
                                 num_classes=cfg.NUM_CLASSES, pretrained=False),
            "segnet_vgg16bn_best.pth", "model_state")
    except ImportError:
        pass

    try:
        from models.esanet import build_esanet
        r["esanet_resnet34"] = (
            lambda: build_esanet(pretrained=False, in_channels=cfg.IN_CHANNELS,
                                 num_classes=cfg.NUM_CLASSES),
            "esanet_resnet34_best.pth", "model_state")
    except ImportError:
        pass

    try:
        from models import build_esanet_official
        r["esanet_official_nyuv2"] = (
            lambda: build_esanet_official(in_channels=cfg.IN_CHANNELS,
                                          num_classes=cfg.NUM_CLASSES, checkpoint=None,
                                          depth_clip_max=cfg.DEPTH_CLIP_MAX,
                                          height=cfg.IMAGE_HEIGHT, width=cfg.IMAGE_WIDTH),
            "esanet_official_nyuv2_best.pth", "model_state")
    except ImportError:
        pass

    return r


# ─── racun ──────────────────────────────────────────────────────────────────

def iou_iz_matrice(h):
    """h[i, j] = broj piksela istinite klase i predvidenih kao j."""
    dijag = np.diag(h).astype(np.float64)
    po_istini = h.sum(axis=1).astype(np.float64)
    po_predikciji = h.sum(axis=0).astype(np.float64)
    unija = po_istini + po_predikciji - dijag
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(unija > 0, dijag / unija, np.nan)
    return iou


def iou_po_slici(pred, gt, k):
    """Isti racun kao postojeci per_class_iou — ukljucujuci 0.0 za odsutnu klasu."""
    out = []
    for c in range(k):
        p, g = pred == c, gt == c
        presjek = np.logical_and(p, g).sum()
        unija = np.logical_or(p, g).sum()
        out.append(float(presjek / (unija + 1e-10)))
    return out


def obradi(model, loader, uredaj, k):
    h = np.zeros((k, k), dtype=np.int64)
    po_slici = []
    prisutna = np.zeros(k, dtype=np.int64)

    model.eval().to(uredaj)
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(uredaj)
            y = batch["mask"].numpy()
            with torch.cuda.amp.autocast(enabled=str(uredaj).startswith("cuda")):
                logits = model(x)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            p = logits.float().argmax(dim=1).cpu().numpy()

            for i in range(len(p)):
                pi, gi = p[i].ravel(), y[i].ravel()
                vazeci = gi < k
                h += np.bincount(gi[vazeci] * k + pi[vazeci],
                                 minlength=k * k).reshape(k, k)
                po_slici.append(iou_po_slici(p[i], y[i], k))
                for c in range(k):
                    if (y[i] == c).any():
                        prisutna[c] += 1

    return h, np.array(po_slici), prisutna


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modeli", nargs="*", default=None)
    ap.add_argument("--csv", default="results/usporedba_globalno.csv")
    args = ap.parse_args()

    k = cfg.NUM_CLASSES
    uredaj = cfg.DEVICE
    reg = _registar()
    imena = args.modeli or list(reg.keys())

    ds = AppleTreeDataset(cfg.TEST_IMG_DIR, cfg.TEST_MASK_DIR,
                          transform=get_val_transforms(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH),
                          img_size=(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH),
                          num_classes=k,
                          depth_clip_max=getattr(cfg, "DEPTH_CLIP_MAX", 4.0))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    redci = []
    prisutna_ukupno = None

    for ime in imena:
        if ime not in reg:
            print(f"[preskacem] {ime}: nije u registru")
            continue
        gradi, ckpt_ime, kljuc = reg[ime]
        put = os.path.join(cfg.CHECKPOINT_DIR, ckpt_ime)
        if not os.path.exists(put):
            print(f"[preskacem] {ime}: nema {put}")
            continue

        model = gradi()
        ck = torch.load(put, map_location="cpu")
        model.load_state_dict(ck[kljuc] if kljuc in ck else ck)

        h, po_sl, prisutna = obradi(model, loader, uredaj, k)
        prisutna_ukupno = prisutna
        g = iou_iz_matrice(h)
        s = po_sl.mean(axis=0)

        red = {"model": ime}
        for i, n in enumerate(CLASS_NAMES):
            red[f"glob_{n}"] = g[i]
            red[f"slika_{n}"] = s[i]
        red["glob_mIoU"] = float(np.nanmean(g[1:]))     # bez pozadine
        red["slika_mIoU"] = float(np.mean(s[1:]))
        red["glob_mIoU_bez_trave"] = float(np.nanmean(g[1:4]))
        redci.append(red)

        del model
        if str(uredaj).startswith("cuda"):
            torch.cuda.empty_cache()

    if not redci:
        sys.exit("Nijedan model nije obraden.")

    print(f"\nTestni skup: {len(ds)} slika, {k} klasa")
    print("Slika na kojima je klasa uopce prisutna:")
    for i, n in enumerate(CLASS_NAMES):
        print(f"  {n:12} {prisutna_ukupno[i]:3d} / {len(ds)}")

    print(f"\n{'model':26}{'mIoU glob':>11}{'mIoU slika':>12}{'razlika':>10}"
          f"{'glob bez trave':>16}")
    print("-" * 75)
    for r in sorted(redci, key=lambda d: -d["glob_mIoU"]):
        print(f"{r['model']:26}{100*r['glob_mIoU']:11.2f}{100*r['slika_mIoU']:12.2f}"
              f"{100*(r['glob_mIoU']-r['slika_mIoU']):+10.2f}{100*r['glob_mIoU_bez_trave']:16.2f}")

    print(f"\nPo klasama (globalno naspram po slici):")
    print(f"{'model':26}" + "".join(f"{n:>22}" for n in CLASS_NAMES[1:]))
    print("-" * (26 + 22 * (len(CLASS_NAMES) - 1)))
    for r in sorted(redci, key=lambda d: -d["glob_mIoU"]):
        red = f"{r['model']:26}"
        for n in CLASS_NAMES[1:]:
            red += f"{100*r[f'glob_{n}']:9.2f} /{100*r[f'slika_{n}']:7.2f}   "
        print(red)

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    import csv as _csv
    with open(args.csv, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(redci[0].keys()))
        w.writeheader()
        w.writerows(redci)
    print(f"\nSpremljeno: {args.csv}")


if __name__ == "__main__":
    main()
