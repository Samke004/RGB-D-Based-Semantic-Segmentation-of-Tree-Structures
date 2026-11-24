"""
main.py
────────────────────────────────────────────────────────────────────────────
Vodi cijeli postupak:
  1. Gradi skupove i loadere
  2. Trenira odabrane modele
  3. Evaluira na testnom skupu
  4. Sprema rezultate i vizualizacije

Usporaba
────────
    # svi modeli
    python main.py

    # jedan model
    python main.py --model esanet_resnet34

    # sa sjemenom i vlastitim sufiksom mapa (preporuceno kad ima vise krugova)
    python main.py --model esanet_resnet34 --seed 42 --sufiks split2_seed42

    # potpuni determinizam (sporije ~10-30 %, ali bit-po-bit ponovljivo)
    python main.py --seed 42 --deterministicno

    # samo evaluacija iz postojecih checkpointa
    python main.py --eval_only --sufiks split2_seed42


SJEME
────────────────────────────────────────────────────────────────────────────
Prije je cfg.RANDOM_SEED postojao u configu, ali se nigdje nije koristio —
nijedno pokretanje nije bilo ponovljivo. Sada postavi_seed() pokriva python,
numpy, torch i CUDA, a DataLoader dobiva vlastiti generator i worker_init_fn.

Bez worker_init_fn svaki radni proces DataLoadera nasljeduje isto numpy
stanje forkom, pa dva workera rade iste slucajne izbore u augmentacijama.
"""

import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config as cfg
from dataset import AppleTreeDataset
from augmentations import get_train_transforms, get_val_transforms
from models import (build_unet, build_deeplabv3, build_deeplabv3plus, build_segnet,
                    build_segformer, build_manet, Pix2PixGenerator, Pix2PixDiscriminator)
from models.esanet import build_esanet
from trainer import SegmentationTrainer, Pix2PixTrainer
from evaluate import evaluate_all


# ─────────────────────────────────────────────────────────────────────────────
# Sjeme
# ─────────────────────────────────────────────────────────────────────────────

def _seed_workera(worker_id: int) -> None:
    """Svaki DataLoader worker dobiva vlastito, ali izvedeno sjeme."""
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def postavi_seed(seed: int, deterministicno: bool = False) -> int:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministicno:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:          # stariji pytorch nema warn_only
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    return seed


# ─────────────────────────────────────────────────────────────────────────────
# Loaderi
# ─────────────────────────────────────────────────────────────────────────────

def build_loaders(seed: int = 42):
    train_tf = get_train_transforms(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)
    val_tf = get_val_transforms(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)

    train_ds = AppleTreeDataset(
        img_dir=cfg.TRAIN_IMG_DIR, mask_dir=cfg.TRAIN_MASK_DIR,
        transform=train_tf,
        img_size=(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH),
        num_classes=cfg.NUM_CLASSES,
        depth_clip_max=cfg.DEPTH_CLIP_MAX,
    )
    val_ds = AppleTreeDataset(
        img_dir=cfg.VAL_IMG_DIR, mask_dir=cfg.VAL_MASK_DIR,
        transform=val_tf,
        img_size=(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH),
        num_classes=cfg.NUM_CLASSES,
        depth_clip_max=cfg.DEPTH_CLIP_MAX,
    )

    print(f"Train: {len(train_ds)} images  |  Val: {len(val_ds)} images")

    g = torch.Generator()
    g.manual_seed(seed)

    loader_kw = dict(
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        worker_init_fn=_seed_workera,
    )

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True,
                              generator=g, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Pojedinacni treneri
# ─────────────────────────────────────────────────────────────────────────────

def train_unet(train_loader, val_loader) -> dict:
    model = build_unet(
        encoder=cfg.UNET_ENCODER,
        weights=cfg.UNET_WEIGHTS,
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES,
    )
    return SegmentationTrainer(
        model=model, model_name="unet_resnet34", cfg=cfg,
        train_loader=train_loader, val_loader=val_loader,
        lr=cfg.UNET_LR,
    ).train()


def train_unet_efficientnet(train_loader, val_loader) -> dict:
    model = build_unet(
        encoder=cfg.UNET_EFF_ENCODER,
        weights=cfg.UNET_EFF_WEIGHTS,
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES,
    )
    return SegmentationTrainer(
        model=model, model_name="unet_efficientnet_b4", cfg=cfg,
        train_loader=train_loader, val_loader=val_loader,
        lr=cfg.UNET_EFF_LR,
    ).train()


def train_deeplabv3(train_loader, val_loader) -> dict:
    model = build_deeplabv3(
        pretrained=cfg.DEEPLABV3_PRETRAINED,
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES,
    )
    return SegmentationTrainer(
        model=model, model_name="deeplabv3plus_resnet50", cfg=cfg,
        train_loader=train_loader, val_loader=val_loader,
        lr=cfg.DEEPLABV3_LR,
    ).train()


def train_deeplabv3plus(train_loader, val_loader) -> dict:
    model = build_deeplabv3plus(
        encoder=getattr(cfg, "DEEPLABV3PLUS_ENCODER", "mobilenet_v2"),
        weights="imagenet" if cfg.DEEPLABV3PLUS_PRETRAINED else None,
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES,
    )
    return SegmentationTrainer(
        model=model, model_name="deeplabv3plus_mobilenetv2", cfg=cfg,
        train_loader=train_loader, val_loader=val_loader,
        lr=cfg.DEEPLABV3PLUS_LR,
    ).train()


def train_segformer(train_loader, val_loader) -> dict:
    model = build_segformer(
        variant=cfg.SEGFORMER_VARIANT,
        weights="imagenet",
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES,
    )
    return SegmentationTrainer(
        model=model, model_name=f"segformer_{cfg.SEGFORMER_VARIANT}", cfg=cfg,
        train_loader=train_loader, val_loader=val_loader,
        lr=cfg.SEGFORMER_LR,
    ).train()


def train_manet(train_loader, val_loader) -> dict:
    model = build_manet(
        pretrained=True,
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES,
    )
    return SegmentationTrainer(
        model=model, model_name="manet_resnext50", cfg=cfg,
        train_loader=train_loader, val_loader=val_loader,
        lr=1e-4,
    ).train()


def train_esanet(train_loader, val_loader) -> dict:
    model = build_esanet(
        pretrained=True,
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES,
    )
    return SegmentationTrainer(
        model=model, model_name="esanet_resnet34", cfg=cfg,
        train_loader=train_loader, val_loader=val_loader,
        lr=cfg.UNET_LR,
    ).train()


def train_segnet(train_loader, val_loader) -> dict:
    model = build_segnet(
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES,
        pretrained=cfg.SEGNET_PRETRAINED,
    )
    return SegmentationTrainer(
        model=model, model_name="segnet_vgg16bn", cfg=cfg,
        train_loader=train_loader, val_loader=val_loader,
        lr=cfg.SEGNET_LR,
    ).train()


def train_pix2pix_gan(train_loader, val_loader) -> dict:
    G = Pix2PixGenerator(in_channels=cfg.IN_CHANNELS, out_channels=cfg.NUM_CLASSES)
    D = Pix2PixDiscriminator(in_channels=cfg.IN_CHANNELS)
    return Pix2PixTrainer(
        generator=G, discriminator=D, use_gan=True,
        cfg=cfg, train_loader=train_loader, val_loader=val_loader,
    ).train()


def train_pix2pix_gen(train_loader, val_loader) -> dict:
    G = Pix2PixGenerator(in_channels=cfg.IN_CHANNELS, out_channels=cfg.NUM_CLASSES)
    return Pix2PixTrainer(
        generator=G, discriminator=None, use_gan=False,
        cfg=cfg, train_loader=train_loader, val_loader=val_loader,
    ).train()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

MODEL_TRAINERS = {
    "unet_resnet34":           train_unet,
    "unet_efficientnet_b4":    train_unet_efficientnet,
    "deeplabv3plus_resnet50":      train_deeplabv3,
    "deeplabv3plus_mobilenet": train_deeplabv3plus,
    "segformer_b1":            train_segformer,
    "segnet_vgg16bn":          train_segnet,
    "esanet_resnet34":         train_esanet,
    # neaktivno — vrati redak ako zatreba:
    # "manet_resnext50":       train_manet,
    # "pix2pix_gan":           train_pix2pix_gan,
    # "pix2pix_gen":           train_pix2pix_gen,
}


def main():
    parser = argparse.ArgumentParser(description="Apple Tree Segmentation Pipeline")
    parser.add_argument(
        "--model",
        choices=["all"] + list(MODEL_TRAINERS.keys()),
        default="all",
        help="Koji model trenirati (zadano: all)",
    )
    parser.add_argument(
        "--eval_only", action="store_true",
        help="Preskoci trening, samo evaluiraj iz postojecih checkpointa",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Sjeme; bez njega se uzima cfg.RANDOM_SEED.",
    )
    parser.add_argument(
        "--deterministicno", action="store_true",
        help="Potpuno determinirani cuDNN algoritmi. Sporije (~10-30 %%), "
             "ali isti rezultat na istom stroju.",
    )
    parser.add_argument(
        "--sufiks", type=str, default="",
        help="Dodaje se na imena mapa za checkpointe, rezultate i logove. "
             "Npr. --sufiks split2_seed42",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Nadjacava cfg.DATA_DIR i sve izvedene putanje. "
             "Npr. --data data_split2. Bez njega vrijedi ono iz configa.",
    )
    args = parser.parse_args()

    # ── sjeme prije svega ostalog ───────────────────────────────────────────
    seed = args.seed if args.seed is not None else getattr(cfg, "RANDOM_SEED", 42)
    postavi_seed(seed, deterministicno=args.deterministicno)

    # ── podaci ──────────────────────────────────────────────────────────────
    # DATA_DIR se u configu cita pri definiciji razreda, pa izvedene putanje
    # (TRAIN_IMG_DIR i ostale) treba preracunati rucno.
    if args.data:
        if not os.path.isdir(args.data):
            raise SystemExit(f"Mapa '{args.data}' ne postoji.")
        cfg.DATA_DIR = args.data
        for podskup, kratica in (("train", "TRAIN"), ("val", "VAL"), ("test", "TEST")):
            setattr(cfg, f"{kratica}_IMG_DIR", os.path.join(args.data, podskup, "images"))
            setattr(cfg, f"{kratica}_MASK_DIR", os.path.join(args.data, podskup, "masks"))

    for putanja in (cfg.TRAIN_IMG_DIR, cfg.TRAIN_MASK_DIR,
                    cfg.VAL_IMG_DIR, cfg.VAL_MASK_DIR):
        if not os.path.isdir(putanja):
            raise SystemExit(f"Nema mape '{putanja}'.")

    # ── mape po krugu, da se pokretanja ne gaze ─────────────────────────────
    if args.sufiks:
        cfg.CHECKPOINT_DIR = f"{cfg.CHECKPOINT_DIR}_{args.sufiks}"
        cfg.RESULTS_DIR = f"{cfg.RESULTS_DIR}_{args.sufiks}"
        cfg.LOG_DIR = f"{getattr(cfg, 'LOG_DIR', 'logs')}_{args.sufiks}"

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    print(f"\nDevice  : {cfg.DEVICE}")
    print(f"Sjeme   : {seed}" + ("  (determinirano)" if args.deterministicno else ""))
    print(f"Podaci  : {cfg.DATA_DIR}")
    print(f"Ckpt    : {cfg.CHECKPOINT_DIR}")
    print(f"Rezult. : {cfg.RESULTS_DIR}")
    print(f"Logovi  : {getattr(cfg, 'LOG_DIR', 'logs')}")
    print(f"Epochs  : {cfg.NUM_EPOCHS}")
    print(f"Batch   : {cfg.BATCH_SIZE}")
    print(f"Size    : {cfg.IMAGE_HEIGHT}x{cfg.IMAGE_WIDTH}")
    print(f"Channels: {cfg.IN_CHANNELS}")
    print(f"Klase   : {cfg.NUM_CLASSES}  {list(cfg.CLASS_NAMES)}")

    histories = {}

    if not args.eval_only:
        train_loader, val_loader = build_loaders(seed=seed)

        to_train = (
            list(MODEL_TRAINERS.keys()) if args.model == "all"
            else [args.model]
        )

        for name in to_train:
            print(f"\n{'#'*60}")
            print(f"  MODEL: {name.upper()}   (sjeme {seed})")
            print(f"{'#'*60}")
            # sjeme se vraca prije svakog modela, da redoslijed u petlji
            # ne mijenja inicijalizaciju pojedinog modela
            postavi_seed(seed, deterministicno=args.deterministicno)
            histories[name] = MODEL_TRAINERS[name](train_loader, val_loader)

    # ── Evaluacija ──────────────────────────────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"  EVALUATION")
    print(f"{'#'*60}")
    evaluate_all(histories if histories else None)


if __name__ == "__main__":
    main()