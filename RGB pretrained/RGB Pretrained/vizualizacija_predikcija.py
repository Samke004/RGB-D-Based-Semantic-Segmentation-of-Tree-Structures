#!/usr/bin/env python3
"""
vizualizacija_predikcija.py
────────────────────────────────────────────────────────────────────────────
Za svaku testnu sliku slaze jednu plocu sa cetiri panela:

    RGB  |  tocna maska  |  predikcija  |  slaganje / pogreske

Cetvrti panel: sivo = tocno pogodeno, obojeno = pogreska, obojena bojom
klase KOJA JE PROPUSTENA (dakle bojom iz tocne maske). Tako se odmah vidi
koju klasu model gubi, a ne samo da grijesi.

Iznad svake ploce pise IoU po klasi za tu sliku, i oznaka "—" za klasu koje
na slici uopce nema. Te slike su one koje su u prosjeku po slici upisivale
prisilnu nulu.

Na kraju ispisuje mIoU na oba nacina (globalno i po slici) da imas brojku
uz slike.

Usporaba:
    python vizualizacija_predikcija.py
    python vizualizacija_predikcija.py --model esanet_resnet34 --n 15
    python vizualizacija_predikcija.py --sortiraj grass        # najgore prvo
    python vizualizacija_predikcija.py --samo-s-klasom grass
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from torch.utils.data import DataLoader

from config import Config as cfg
from dataset import AppleTreeDataset, IMAGENET_MEAN, IMAGENET_STD
from augmentations import get_val_transforms
from metrics import CLASS_NAMES

BOJE = np.array([
    [ 40,  40,  40],   # 0 pozadina  tamno siva
    [220,  50,  47],   # 1 deblo     crvena
    [ 38, 139, 210],   # 2 grane     plava
    [ 70, 200,  90],   # 3 potpora   zelena
    [240, 200,  40],   # 4 trava     zuta
], dtype=np.uint8)


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


def rgb_natrag(t):
    """Iz normaliziranog tenzora natrag u prikazivu sliku."""
    a = t[:3].cpu().numpy().transpose(1, 2, 0)
    a = a * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(a, 0, 1)


def iou_slike(pred, gt, k):
    """IoU po klasi za jednu sliku; nan kad klase nema ni u istini ni u predikciji."""
    out = []
    for c in range(k):
        p, g = pred == c, gt == c
        u = np.logical_or(p, g).sum()
        out.append(np.logical_and(p, g).sum() / u if u > 0 else np.nan)
    return np.array(out, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esanet_resnet34")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--izlaz", default=None)
    ap.add_argument("--n", type=int, default=0, help="koliko ploca spremiti (0 = sve)")
    ap.add_argument("--sortiraj", default=None,
                    help="ime klase — najgore po toj klasi prvo")
    ap.add_argument("--samo-s-klasom", default=None,
                    help="ime klase — samo slike na kojima ta klasa postoji")
    ap.add_argument("--dpi", type=int, default=110)
    args = ap.parse_args()

    k = cfg.NUM_CLASSES
    uredaj = cfg.DEVICE
    ckpt = args.checkpoint or os.path.join(cfg.CHECKPOINT_DIR, f"{args.model}_best.pth")
    izlaz = args.izlaz or os.path.join(cfg.RESULTS_DIR, f"vizualizacija_{args.model}")
    os.makedirs(izlaz, exist_ok=True)

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
    print(f"Model: {args.model}   checkpoint: {os.path.basename(ckpt)}")

    uzorci = []
    hist = np.zeros((k, k), dtype=np.int64)

    with torch.no_grad():
        for b in loader:
            x = b["image"].to(uredaj)
            with torch.cuda.amp.autocast(enabled=str(uredaj).startswith("cuda")):
                o = model(x)
            if isinstance(o, (list, tuple)):
                o = o[0]
            pred = o.float().argmax(1)[0].cpu().numpy()
            gt = b["mask"][0].numpy()

            hist += np.bincount(gt.ravel() * k + pred.ravel(),
                                minlength=k * k).reshape(k, k)
            uzorci.append({
                "stem": b["stem"][0],
                "rgb": rgb_natrag(b["image"][0]),
                "gt": gt, "pred": pred,
                "iou": iou_slike(pred, gt, k),
                "ima": np.array([(gt == c).any() for c in range(k)]),
            })

    # ── filtriranje i sortiranje ────────────────────────────────────────────
    if args.samo_s_klasom:
        ci = CLASS_NAMES.index(args.samo_s_klasom)
        uzorci = [u for u in uzorci if u["ima"][ci]]
        print(f"Zadrzano {len(uzorci)} slika na kojima postoji '{args.samo_s_klasom}'")

    if args.sortiraj:
        ci = CLASS_NAMES.index(args.sortiraj)
        uzorci.sort(key=lambda u: (np.nan_to_num(u["iou"][ci], nan=9.0)))
        print(f"Sortirano po IoU klase '{args.sortiraj}', najgore prvo")

    za_crtanje = uzorci[:args.n] if args.n else uzorci

    # ── crtanje ─────────────────────────────────────────────────────────────
    legenda = [Patch(facecolor=BOJE[c] / 255, label=CLASS_NAMES[c]) for c in range(k)]

    for i, u in enumerate(za_crtanje):
        gt_rgb = BOJE[u["gt"]]
        pr_rgb = BOJE[u["pred"]]

        tocno = u["gt"] == u["pred"]
        diff = np.full((*u["gt"].shape, 3), 205, dtype=np.uint8)   # svijetlo sivo = tocno
        diff[tocno] = 225
        krivo = ~tocno
        diff[krivo] = BOJE[u["gt"][krivo]]                          # boja propustene klase

        fig, ax = plt.subplots(1, 4, figsize=(19, 4.1))
        ax[0].imshow(u["rgb"]);      ax[0].set_title("RGB", fontsize=11)
        ax[1].imshow(gt_rgb);        ax[1].set_title("tocna maska", fontsize=11)
        ax[2].imshow(pr_rgb);        ax[2].set_title("predikcija", fontsize=11)
        ax[3].imshow(diff)
        ax[3].set_title(f"pogreske ({100*krivo.mean():.1f} % piksela)", fontsize=11)
        for a in ax:
            a.axis("off")

        dijelovi = []
        for c in range(1, k):
            v = u["iou"][c]
            dijelovi.append(f"{CLASS_NAMES[c]}: " + ("—" if np.isnan(v) else f"{100*v:.1f}"))
        valjane = u["iou"][1:][~np.isnan(u["iou"][1:])]
        naslov = (f"{u['stem']}    |    " + "   ".join(dijelovi) +
                  f"    |    mIoU slike: {100*valjane.mean():.1f}")
        fig.suptitle(naslov, fontsize=11, y=0.99)
        fig.legend(handles=legenda, loc="lower center", ncol=k, frameon=False, fontsize=9)
        fig.tight_layout(rect=[0, 0.06, 1, 0.94])

        put = os.path.join(izlaz, f"{i:03d}_{u['stem']}.png")
        fig.savefig(put, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"Spremljeno {len(za_crtanje)} ploca u {izlaz}/")

    # ── brojke ──────────────────────────────────────────────────────────────
    d = np.diag(hist).astype(float)
    glob = d / (hist.sum(1) + hist.sum(0) - d)

    po_slici_svi = np.array([[0.0 if np.isnan(v) else v for v in u["iou"]] for u in uzorci])
    po_slici_prisutne = []
    for c in range(k):
        v = [u["iou"][c] for u in uzorci if u["ima"][c]]
        po_slici_prisutne.append(np.nanmean(v) if v else np.nan)

    print(f"\n{'klasa':12}{'globalno':>11}{'po slici (sve)':>17}{'po slici (prisutne)':>22}{'n slika':>10}")
    print("-" * 72)
    for c in range(k):
        n = sum(u["ima"][c] for u in uzorci)
        print(f"{CLASS_NAMES[c]:12}{100*glob[c]:11.2f}{100*po_slici_svi[:,c].mean():17.2f}"
              f"{100*po_slici_prisutne[c]:22.2f}{n:10d}")

    print(f"\nmIoU bez pozadine — globalno {100*np.nanmean(glob[1:]):.2f}   "
          f"po slici {100*po_slici_svi[:,1:].mean():.2f}")


if __name__ == "__main__":
    main()
