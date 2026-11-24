"""
dataset.py
────────────────────────────────────────────────────────────────────────────
PyTorch Dataset za RGBD segmentaciju stabala.

IZMJENE u odnosu na prethodnu verziju:

1. np.clip(mask, 0, 3) uklonjen.
   Broj klasa dolazi iz num_classes. Prije je trava (indeks 4) tiho
   postajala potpora (indeks 3). Sada se neispravna maska prijavi
   kao greska umjesto da se potiho popravi.

2. Depth vise ne prolazi kroz uint8.
   Prije: float -> uint8 (256 razina preko 0-4 m, korak 1.56 cm) -> float.
   Sada ostaje float32 kroz cijeli put. Augmentacije ga primaju kao
   "mask" target, pa koriste NEAREST interpolaciju i ne diraju ga
   fotometrijski.

3. cv2.resize za depth koristi INTER_NEAREST.
   Depth je rijedak (~88 % nula). Bilinearno mijesanje mjerene vrijednosti
   s nulom susjeda stvara dubine koje senzor nikad nije izmjerio.

4. DEPTH_CLIP_MAX je parametar, ne konstanta.
   Provjeri stvarni raspon svojih podataka - ako je dubina do ~1.9 m,
   dijeljenje s 4.0 trosi vise od pola ulaznog raspona na prazno.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEPTH_CLIP_MAX_ZADANO = 4.0


class AppleTreeDataset(Dataset):

    def __init__(self, img_dir, mask_dir, transform=None, img_size=(480, 640),
                 num_classes=5, depth_clip_max=DEPTH_CLIP_MAX_ZADANO):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_size = img_size
        self.num_classes = num_classes
        self.depth_clip_max = float(depth_clip_max)

        img_stems = {Path(f).stem.replace("_rgbd", "")
                     for f in os.listdir(img_dir) if f.endswith("_rgbd.npy")}
        mask_stems = {Path(f).stem.replace("_mask", "")
                      for f in os.listdir(mask_dir) if f.endswith("_mask.png")}

        self.stems = sorted(img_stems & mask_stems)

        samo_slike = img_stems - mask_stems
        samo_maske = mask_stems - img_stems
        if samo_slike or samo_maske:
            print(f"UPOZORENJE [{img_dir}]: {len(samo_slike)} slika bez maske, "
                  f"{len(samo_maske)} maski bez slike - nesparene se preskacu")

        if len(self.stems) == 0:
            raise FileNotFoundError(
                f"No matching pairs found.\n  images: {img_dir}\n  masks: {mask_dir}"
            )
        print(f"Dataset loaded: {len(self.stems)} samples from {img_dir} "
              f"({self.num_classes} klasa, depth clip {self.depth_clip_max} m)")

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]

        # ── Ucitaj RGBD ──────────────────────────────────────────────────────
        rgbd = np.load(os.path.join(self.img_dir, f"{stem}_rgbd.npy"))

        H, W = self.img_size
        if rgbd.shape[:2] != (H, W):
            rgb = cv2.resize(rgbd[:, :, :3], (W, H), interpolation=cv2.INTER_LINEAR)
            # NEAREST: depth je rijedak, bilinearno bi izmislilo medu-vrijednosti
            depth = cv2.resize(rgbd[:, :, 3], (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            rgb = rgbd[:, :, :3]
            depth = rgbd[:, :, 3]

        rgb = np.ascontiguousarray(rgb, dtype=np.float32)      # [H, W, 3] u [0,1]
        depth = np.ascontiguousarray(depth, dtype=np.float32)  # [H, W] u metrima

        # ── Ucitaj masku ─────────────────────────────────────────────────────
        put_maske = os.path.join(self.mask_dir, f"{stem}_mask.png")
        mask = cv2.imread(put_maske, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Maska se ne moze procitati: {put_maske}")
        if mask.shape[:2] != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        najveci = int(mask.max())
        if najveci >= self.num_classes:
            raise ValueError(
                f"{stem}: maska sadrzi indeks klase {najveci}, a num_classes="
                f"{self.num_classes}. Ocekujem 0..{self.num_classes - 1}. "
                f"Provjeri je li konverzija skupa koristila isti broj klasa."
            )
        mask = mask.astype(np.uint8)

        # ── Augmentacija - RGB + Depth + Mask zajedno ────────────────────────
        if self.transform is not None:
            # RGB u uint8 za albumentations (fotometrijske transformacije to traze).
            rgb_uint8 = (rgb * 255.0).clip(0, 255).astype(np.uint8)

            # Depth ide kao "mask" target: ostaje float32, nearest interpolacija,
            # fotometrijske transformacije ga ne diraju.
            rezultat = self.transform(
                image=rgb_uint8,
                mask=mask,
                depth=depth,
            )

            rgb = rezultat["image"].astype(np.float32) / 255.0
            mask = rezultat["mask"]
            depth = np.asarray(rezultat["depth"], dtype=np.float32)

        # ── Normalizacija ────────────────────────────────────────────────────
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        depth = np.clip(depth, 0.0, self.depth_clip_max) / self.depth_clip_max

        # ── Tenzori ──────────────────────────────────────────────────────────
        rgb_t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float()
        depth_t = torch.from_numpy(np.ascontiguousarray(depth)).unsqueeze(0).float()
        mask_t = torch.from_numpy(np.ascontiguousarray(mask).astype(np.int64)).long()

        image_t = torch.cat([rgb_t, depth_t], dim=0)   # [4, H, W]

        return {"image": image_t, "mask": mask_t, "stem": stem}