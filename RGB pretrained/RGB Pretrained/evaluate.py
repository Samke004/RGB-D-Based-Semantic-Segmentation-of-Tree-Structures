"""
evaluate.py — Multi-class evaluation on test set
────────────────────────────────────────────────────────────────────────────
Izlazi:
  1. CSV s metrikama po slici, za svaki model
  2. Zbirna usporedna tablica (CSV + konzola)
  3. Najgorih 10 slika po indeksu tezine
  4. Kvalitativna vizualizacija
  5. Krivulje treniranja

VAZNO — koje brojke idu u rad
────────────────────────────────────────────────────────────────────────────
Stupci glob_* dolaze iz JEDNE matrice zabune preko cijelog testnog skupa,
isto kako racunaju DFormer, mmsegmentation i vecina radova. To su brojke
za rad.

Ostali stupci racunaju se po slici pa se uzima prosjek, cime svaka slika
ima jednaku tezinu bez obzira na broj piksela. Zadrzani su radi usporedbe
i jer pokazuju rasap po slikama, ali nisu usporedivi s literaturom.

Svaka prosjecna metrika ima dvije varijante:
  <ime>           — bez pozadine (klasa 0)
  <ime>_with_bg   — s pozadinom
Pozadina je ~76 % piksela i IoU joj je ~0.99, pa varijanta s pozadinom
izlazi nekoliko bodova visa.
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config as cfg
from dataset import AppleTreeDataset
from augmentations import get_val_transforms
from models.esanet import build_esanet
from models import (build_unet, build_deeplabv3, build_deeplabv3plus, build_manet,
                    build_segnet, build_segformer, Pix2PixGenerator)
from metrics import (compute_all_metrics, aggregate_metrics, aggregate_counts,
                     CLASS_NAMES, prazna_matrica, dodaj_u_matricu,
                     metrike_iz_matrice, imena_globalnih_stupaca)

# Boje za vizualizaciju
CLASS_COLORS = np.array([
    [0,   0,   0  ],   # 0 background — crna
    [255, 0,   0  ],   # 1 trunk      — crvena
    [0,   0,   255],   # 2 branches   — plava
    [0,   255, 0  ],   # 3 support    — zelena
    [255, 255, 0  ],   # 4 grass      — zuta
], dtype=np.uint8)

assert len(CLASS_COLORS) >= cfg.NUM_CLASSES, (
    f"CLASS_COLORS ima {len(CLASS_COLORS)} boja, a NUM_CLASSES je {cfg.NUM_CLASSES}"
)

LEGENDA = "  ".join(
    f"{ime}={boja}" for ime, boja in zip(
        CLASS_NAMES,
        ["black", "red", "blue", "green", "yellow"][:cfg.NUM_CLASSES])
)

# ─────────────────────────────────────────────────────────────────────────────

def _load_model(model, ckpt_path, device, key="model_state"):
    if not os.path.isfile(ckpt_path):
        print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
        return None
    state = torch.load(ckpt_path, map_location=device)
    actual_key = key if key in state else "G_state"
    model.load_state_dict(state[actual_key])
    model.eval()
    return model.to(device)


@torch.no_grad()
def _run_inference(model, test_loader, device):
    results = []
    for batch in tqdm(test_loader, desc="  Inference"):
        images = batch["image"].to(device)
        logits = model(images)                          # [B, C, H, W]
        if isinstance(logits, (list, tuple)):           # neki modeli vracaju i side-izlaze
            logits = logits[0]
        preds = logits.float().argmax(dim=1).cpu().numpy()
        gts = batch["mask"].cpu().numpy()
        stems = batch["stem"]

        occs = batch["occlusion"].squeeze(1).numpy() if "occlusion" in batch else [None]*len(stems)
        deps = batch["depth"].squeeze(1).numpy()     if "depth"     in batch else [None]*len(stems)

        for i, stem in enumerate(stems):
            results.append({
                "stem": stem, "pred": preds[i], "gt": gts[i],
                "occlusion": occs[i], "depth": deps[i],
            })
    return results


def _compute_per_image_metrics(inference_results):
    rows = []
    for r in inference_results:
        m = compute_all_metrics(
            pred=r["pred"], gt=r["gt"],
            occlusion=r["occlusion"], depth=r["depth"],
            boundary_theta=cfg.BOUNDARY_THETA,
        )
        m["stem"] = r["stem"]
        rows.append(m)
    return pd.DataFrame(rows)


def _globalne_metrike(inference_results, k):
    """Jedna matrica zabune preko cijelog skupa; racun je u metrics.py,
    isti koji koriste trainer.py i globalni_miou.py."""
    hist = prazna_matrica(k)
    for r in inference_results:
        dodaj_u_matricu(hist, r["pred"], r["gt"], k)
    return metrike_iz_matrice(hist)


def _visualise(model_name, inference_results, out_dir, n_show=8, seed=None):
    """GT / predikcija / preklop za n_show nasumicno odabranih testnih slika.

    Odabir je reproducibilan (fiksni seed) i namjerno isti za svaki model,
    pa se razliciti modeli mogu usporediti na istim primjerima.
    """
    os.makedirs(out_dir, exist_ok=True)

    seed = cfg.RANDOM_SEED if seed is None else seed
    rng = random.Random(seed)

    n_show = min(n_show, len(inference_results))
    indices = sorted(rng.sample(range(len(inference_results)), n_show))
    samples = [inference_results[i] for i in indices]
    n = len(samples)

    fig, axes = plt.subplots(n, 3, figsize=(12, n * 4))
    if n == 1:
        axes = axes[np.newaxis, :]

    for j, t in enumerate(["Ground Truth", "Prediction", "Overlay (TP/FP/FN branches)"]):
        axes[0, j].set_title(t, fontsize=11, fontweight="bold")

    for i, r in enumerate(samples):
        pred, gt = r["pred"], r["gt"]

        gt_rgb = CLASS_COLORS[gt]
        pred_rgb = CLASS_COLORS[pred]

        overlay = np.zeros((*gt.shape, 3), dtype=np.uint8)
        tp = (pred == 2) & (gt == 2)
        fp = (pred == 2) & (gt != 2)
        fn = (pred != 2) & (gt == 2)
        overlay[tp] = [0,   255, 0  ]
        overlay[fp] = [255, 0,   0  ]
        overlay[fn] = [0,   0,   255]

        axes[i, 0].imshow(gt_rgb);   axes[i, 0].axis("off")
        axes[i, 1].imshow(pred_rgb); axes[i, 1].axis("off")
        axes[i, 2].imshow(overlay);  axes[i, 2].axis("off")
        axes[i, 0].set_ylabel(r["stem"][:20], fontsize=7)

    fig.suptitle(f"{model_name} — rezultati po klasama (nasumican uzorak, seed={seed})\n"
                 f"Boje: {LEGENDA}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{model_name}_qual.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"  Saved: {path}")


def plot_training_curves(histories: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, h in histories.items():
        if "val_iou" in h:
            axes[0].plot(h["val_iou"], label=name)
        if "val_bf1" in h:
            axes[1].plot(h["val_bf1"], label=name)
    for ax, title, ylabel in zip(axes,
                                 ["Validacijski mIoU (globalno, bez pozadine)",
                                  "Validacijski mean Boundary F1 (po slici)"],
                                 ["mIoU", "BF1"]):
        ax.set_title(title); ax.set_xlabel("Epoha")
        ax.set_ylabel(ylabel); ax.legend(); ax.grid(True)
    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"Saved: {path}")


def worst_n_analysis(df, model_name, out_dir, n=10):
    rows = []
    for col, label in [("occlusion_difficulty_index", "Occlusion DI"),
                       ("depth_difficulty_index",     "Depth DI")]:
        if col not in df.columns:
            continue
        sub = df.dropna(subset=[col]).nlargest(n, col)
        for _, row in sub.iterrows():
            rows.append({
                "model": model_name, "difficulty_by": label,
                "stem":  row["stem"], col: row.get(col, float("nan")),
                "branch_recall":          row.get("branch_recall", float("nan")),
                "occluded_branch_recall": row.get("occluded_branch_recall", float("nan")),
            })
    if rows:
        path = os.path.join(out_dir, f"{model_name}_worst{n}.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"  Worst-{n} saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(histories: dict = None):
    device = cfg.DEVICE
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    test_dataset = AppleTreeDataset(
        img_dir   = cfg.TEST_IMG_DIR,
        mask_dir  = cfg.TEST_MASK_DIR,
        transform = get_val_transforms(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH),
        img_size  = (cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH),
        num_classes    = cfg.NUM_CLASSES,
        depth_clip_max = cfg.DEPTH_CLIP_MAX,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=cfg.NUM_WORKERS, pin_memory=False)
    print(f"\nTest set: {len(test_dataset)} images")

    model_configs = [
        ("unet_resnet34",
         build_unet("resnet34", weights=None,
                    in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
         os.path.join(cfg.CHECKPOINT_DIR, "unet_resnet34_best.pth"), "model_state"),

        ("unet_efficientnet_b4",
         build_unet("efficientnet-b4", weights=None,
                    in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
         os.path.join(cfg.CHECKPOINT_DIR, "unet_efficientnet_b4_best.pth"), "model_state"),

        ("deeplabv3plus_resnet50",
         build_deeplabv3(pretrained=False,
                         in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
         os.path.join(cfg.CHECKPOINT_DIR, "deeplabv3plus_resnet50_best.pth"), "model_state"),

        ("deeplabv3plus_mobilenetv2",
         build_deeplabv3plus(encoder="mobilenet_v2", weights=None,
                             in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
         os.path.join(cfg.CHECKPOINT_DIR, "deeplabv3plus_mobilenetv2_best.pth"), "model_state"),

        ("segformer_b1",
         build_segformer(variant=cfg.SEGFORMER_VARIANT, weights=None,
                         in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
         os.path.join(cfg.CHECKPOINT_DIR, "segformer_b1_best.pth"), "model_state"),

        ("segnet_vgg16bn",
         build_segnet(in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES,
                      pretrained=False),
         os.path.join(cfg.CHECKPOINT_DIR, "segnet_vgg16bn_best.pth"), "model_state"),

        ("esanet_resnet34",
         build_esanet(pretrained=False,
                      in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES),
         os.path.join(cfg.CHECKPOINT_DIR, "esanet_resnet34_best.pth"), "model_state"),
    ]

    aggregated_all = {}
    counts_all = {}

    # Stupci tablice: prvo globalni (za rad), onda po slici (za uvid)
    KEY_METRICS = (
        imena_globalnih_stupaca()
        + ["mean_iou", "mean_iou_with_bg",
           "mean_boundary_f1", "mean_boundary_f1_with_bg",
           "mean_precision", "mean_precision_with_bg",
           "mean_recall", "mean_recall_with_bg",
           "mean_f1", "mean_f1_with_bg",
           "overall_accuracy", "pixel_accuracy_fg",
           "frequency_weighted_iou", "frequency_weighted_iou_with_bg",
           "branch_recall"]
        + [f"iou_{c}"           for c in CLASS_NAMES]
        + [f"recall_{c}"        for c in CLASS_NAMES]
        + [f"precision_{c}"     for c in CLASS_NAMES]
        + [f"f1_{c}"            for c in CLASS_NAMES]
        + [f"boundary_f1_{c}"   for c in CLASS_NAMES]
    )

    for model_name, model, ckpt_path, state_key in model_configs:
        print(f"\n── Evaluating: {model_name} ──")
        model = _load_model(model, ckpt_path, device, key=state_key)
        if model is None:
            continue

        inf_results = _run_inference(model, test_loader, device)
        df = _compute_per_image_metrics(inf_results)
        csv_path = os.path.join(cfg.RESULTS_DIR, f"{model_name}_per_image.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Per-image CSV: {csv_path}")

        po_slici = df.drop(columns=["stem"]).to_dict("records")
        agg = aggregate_metrics(po_slici)
        agg.update(_globalne_metrike(inf_results, cfg.NUM_CLASSES))
        aggregated_all[model_name] = agg
        counts_all[model_name] = aggregate_counts(po_slici)

        worst_n_analysis(df, model_name, cfg.RESULTS_DIR)
        _visualise(model_name, inf_results, cfg.RESULTS_DIR)

    if not aggregated_all:
        print("\nNijedan model nije evaluiran — nema checkpointa.")
        return None

    # ── Koliko slika doprinosi kojoj metrici po slici ────────────────────────
    prvi = next(iter(counts_all.values()))
    print(f"\n{'─'*60}")
    print("  Slika koje sadrze klasu (vrijedi za stupce po slici)")
    print(f"{'─'*60}")
    for c in CLASS_NAMES:
        kljuc = f"iou_{c}"
        if kljuc in prvi:
            print(f"  {c:14} {prvi[kljuc]:3} / {len(test_dataset)}")
    print("  (glob_* stupci koriste sve piksele svih slika i ne ovise o ovome)")

    # ── Usporedna tablica ────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("  COMPARISON TABLE")
    print("="*80)

    compare_rows = []
    for mname, agg in aggregated_all.items():
        row = {"model": mname}
        for k in KEY_METRICS:
            row[k] = round(float(agg.get(k, float("nan"))), 4)
        compare_rows.append(row)

    df_compare = pd.DataFrame(compare_rows).set_index("model")

    # na konzolu samo glavne brojke; cijela tablica ide u CSV
    glavni = [c for c in ["glob_mean_iou", "glob_mean_iou_with_bg",
                          "glob_mean_f1", "glob_overall_acc"] if c in df_compare.columns]
    glavni += [f"glob_iou_{c}" for c in CLASS_NAMES if f"glob_iou_{c}" in df_compare.columns]
    print(df_compare[glavni].sort_values("glob_mean_iou", ascending=False).to_string())

    path = os.path.join(cfg.RESULTS_DIR, "comparison_table.csv")
    df_compare.to_csv(path)
    print(f"\nPuna tablica ({len(df_compare.columns)} stupaca): {path}")
    print("Za rad koristi glob_* stupce.")

    if histories:
        plot_training_curves(histories, cfg.RESULTS_DIR)

    return df_compare


if __name__ == "__main__":
    evaluate_all()