#!/usr/bin/env python3
"""
provjera_metrika.py
────────────────────────────────────────────────────────────────────────────
Nezavisna provjera tvrdnje da je razlika globalno/po-slici stvarna, a ne
posljedica greske u mojoj skripti.

Za JEDAN model racuna IoU na pet nacina i stavlja ih jedan uz drugi:

  A) globalno, matrica zabune preko bincount   (isto kao globalni_miou.py)
  B) globalno, sklearn.metrics.confusion_matrix (nezavisna implementacija)
  C) globalno, grubo brojanje piksela po klasi  (treca nezavisna implementacija)
  D) globalno, DFormerova formula doslovno iz njegovog Metrics razreda
  E) prosjek po slici, kako to radi postojeci metrics.py

Ako se A, B, C i D slazu do na zaokruzivanje, globalni racun je ispravan.
Razlika prema E je onda stvarna osobina nacina mjerenja.

Dodatno ispisuje, za svaku klasu:
  - prosjek po slici preko SVIH slika              (ono sto je u tablici)
  - prosjek po slici preko slika gdje klasa POSTOJI (posteno usporedivo)
  - koliko slika je upisalo prisilnu nulu

Usporaba:
    python provjera_metrika.py --model esanet_resnet34
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config as cfg
from dataset import AppleTreeDataset
from augmentations import get_val_transforms
from metrics import CLASS_NAMES


def gradi(ime):
    from models import (build_unet, build_deeplabv3, build_deeplabv3plus,
                        build_segformer)
    k, c = cfg.NUM_CLASSES, cfg.IN_CHANNELS
    if ime == "unet_resnet34":
        return build_unet("resnet34", weights=None, in_channels=c, num_classes=k)
    if ime == "unet_efficientnet_b4":
        return build_unet("efficientnet-b4", weights=None, in_channels=c, num_classes=k)
    if ime == "deeplabv3plus_resnet50":
        return build_deeplabv3(pretrained=False, in_channels=c, num_classes=k)
    if ime == "deeplabv3plus_mobilenetv2":
        return build_deeplabv3plus(encoder="mobilenet_v2", weights=None,
                                   in_channels=c, num_classes=k)
    if ime == "segformer_b1":
        return build_segformer(variant=cfg.SEGFORMER_VARIANT, weights=None,
                               in_channels=c, num_classes=k)
    if ime == "segnet_vgg16bn":
        from models import build_segnet
        return build_segnet(in_channels=c, num_classes=k, pretrained=False)
    if ime == "esanet_resnet34":
        from models.esanet import build_esanet
        return build_esanet(pretrained=False, in_channels=c, num_classes=k)
    if ime == "esanet_official_nyuv2":
        from models import build_esanet_official
        return build_esanet_official(in_channels=c, num_classes=k, checkpoint=None,
                                     depth_clip_max=cfg.DEPTH_CLIP_MAX,
                                     height=cfg.IMAGE_HEIGHT, width=cfg.IMAGE_WIDTH)
    raise ValueError(f"Nepoznat model: {ime}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esanet_resnet34")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    k = cfg.NUM_CLASSES
    uredaj = cfg.DEVICE
    ckpt = args.checkpoint or os.path.join(cfg.CHECKPOINT_DIR, f"{args.model}_best.pth")

    ds = AppleTreeDataset(cfg.TEST_IMG_DIR, cfg.TEST_MASK_DIR,
                          transform=get_val_transforms(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH),
                          img_size=(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH),
                          num_classes=k,
                          depth_clip_max=getattr(cfg, "DEPTH_CLIP_MAX", 4.0))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    model = gradi(args.model)
    ck = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(ck["model_state"] if "model_state" in ck else ck)
    model.eval().to(uredaj)

    # ── skupi sve predikcije i istine ───────────────────────────────────────
    sve_p, sve_g = [], []
    with torch.no_grad():
        for b in loader:
            x = b["image"].to(uredaj)
            with torch.cuda.amp.autocast(enabled=str(uredaj).startswith("cuda")):
                o = model(x)
            if isinstance(o, (list, tuple)):
                o = o[0]
            sve_p.append(o.float().argmax(1).cpu().numpy())
            sve_g.append(b["mask"].numpy())
    P = np.concatenate(sve_p)          # [N, H, W]
    G = np.concatenate(sve_g)
    print(f"\nModel: {args.model}   slika: {len(P)}   oblik: {P.shape[1:]}")

    pf, gf = P.ravel(), G.ravel()

    # ── A) bincount matrica ─────────────────────────────────────────────────
    hA = np.bincount(gf * k + pf, minlength=k * k).reshape(k, k).astype(np.float64)
    d = np.diag(hA)
    A = d / (hA.sum(1) + hA.sum(0) - d)

    # ── B) sklearn ──────────────────────────────────────────────────────────
    try:
        from sklearn.metrics import confusion_matrix
        hB = confusion_matrix(gf, pf, labels=list(range(k))).astype(np.float64)
        dB = np.diag(hB)
        B = dB / (hB.sum(1) + hB.sum(0) - dB)
        slaze_hist = np.array_equal(hA.astype(np.int64), hB.astype(np.int64))
    except ImportError:
        B = np.full(k, np.nan); slaze_hist = None

    # ── C) grubo brojanje, bez ikakve matrice ───────────────────────────────
    C = np.zeros(k)
    for c in range(k):
        p, g = (pf == c), (gf == c)
        C[c] = np.logical_and(p, g).sum() / np.logical_or(p, g).sum()

    # ── D) DFormerova formula doslovno ──────────────────────────────────────
    hD = torch.zeros(k, k)
    hD += torch.bincount(torch.from_numpy(gf).long() * k + torch.from_numpy(pf).long(),
                         minlength=k ** 2).view(k, k).float()
    hh = hD.float(); dd = hh.diag()
    D = (dd / (hh.sum(1) + hh.sum(0) - dd + 1e-6)).numpy()

    # ── E) prosjek po slici ─────────────────────────────────────────────────
    po_slici = np.zeros((len(P), k))
    prisutna = np.zeros((len(P), k), dtype=bool)
    for i in range(len(P)):
        for c in range(k):
            p, g = P[i] == c, G[i] == c
            po_slici[i, c] = np.logical_and(p, g).sum() / (np.logical_or(p, g).sum() + 1e-10)
            prisutna[i, c] = g.any()
    E = po_slici.mean(0)

    # ── ispis ───────────────────────────────────────────────────────────────
    print(f"\nGLOBALNI RACUN — cetiri nezavisne implementacije")
    print(f"{'klasa':12}{'A bincount':>12}{'B sklearn':>12}{'C brojanje':>12}{'D DFormer':>12}")
    print('-' * 60)
    for c in range(k):
        print(f"{CLASS_NAMES[c]:12}{100*A[c]:12.4f}{100*B[c]:12.4f}{100*C[c]:12.4f}{100*D[c]:12.4f}")

    najveca = np.nanmax(np.abs(np.vstack([A - C, A - D, (A - B) if not np.isnan(B).all() else A - A])))
    print(f"\n  najveca razlika medu implementacijama: {100*najveca:.6f} postotnih bodova")
    if slaze_hist is not None:
        print(f"  bincount i sklearn matrica identicne: {slaze_hist}")
    print(f"  {'SLAZU SE — globalni racun je ispravan' if najveca < 1e-6 else 'NE SLAZU SE — ima greske'}")

    print(f"\nPO SLICI — odakle dolazi razlika")
    print(f"{'klasa':12}{'svih 77':>10}{'samo prisutne':>15}{'n prisutnih':>13}{'prisilnih nula':>16}{'globalno':>11}")
    print('-' * 78)
    for c in range(k):
        n = prisutna[:, c].sum()
        pr = po_slici[prisutna[:, c], c].mean() if n else float('nan')
        print(f"{CLASS_NAMES[c]:12}{100*E[c]:10.2f}{100*pr:15.2f}{n:13d}{len(P)-n:16d}{100*A[c]:11.2f}")

    print(f"\nmIoU bez pozadine:  globalno {100*np.nanmean(A[1:]):.2f}   po slici {100*E[1:].mean():.2f}")


if __name__ == "__main__":
    main()
