"""
augmentations.py
────────────────────────────────────────────────────────────────────────────
Augmentacije za RGBD segmentaciju.

IZMJENE u odnosu na prethodnu verziju:

1. depth je deklariran kao "mask", ne kao "image".
   Posljedica: geometrijske transformacije koriste NEAREST interpolaciju
   (nema izmisljenih dubina na rubovima praznina), a fotometrijske
   (brightness/contrast, blur, CLAHE) se na depth NE primjenjuju uopce.
   Prije je komentar tvrdio da se ne primjenjuju, ali jesu.

2. RandomRotate90 izbacen.
   Na nekvadratnim slikama (480x640) rotacija za 90 stupnjeva daje 640x480,
   pa se batch ne moze posloziti. Na kvadratu je bio bezopasan.

3. VerticalFlip izbacen (vidi NAPOMENA nize).

Resize se NE radi ako su slike vec u ciljanoj velicini - A.Resize je tada
no-op, pa ga drzimo radi sigurnosti.
"""

import albumentations as A

# Stabla imaju gravitaciju: deblo raste prema gore, potporni stupovi su
# okomiti, trava je uvijek pri dnu kadra. Okretanje po vertikali stvara
# prizore koje senzor nikad ne vidi i ucenju odnosa gore/dolje samo smeta.
# Ako ga ipak zelis, vrati A.VerticalFlip(p=0.2) medu geometrijske.
KORISTI_VERTICAL_FLIP = False


def _geometrijske():
    t = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1,
            rotate_limit=15, border_mode=0, p=0.5,
        ),
        A.ElasticTransform(alpha=120, sigma=6.0, p=0.3),
        A.GridDistortion(p=0.2),
    ]
    if KORISTI_VERTICAL_FLIP:
        t.insert(1, A.VerticalFlip(p=0.2))
    return t


def _fotometrijske():
    # Primjenjuju se SAMO na RGB - depth je "mask" target pa ih albumentations
    # na njega ne primjenjuje. Ovaj put je to doista tocno.
    return [
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.CLAHE(clip_limit=4.0, p=0.2),
    ]


def get_train_transforms(height: int = 480, width: int = 640) -> A.Compose:
    return A.Compose(
        [A.Resize(height, width)] + _geometrijske() + _fotometrijske(),
        additional_targets={"depth": "mask"},
    )


def get_val_transforms(height: int = 480, width: int = 640) -> A.Compose:
    return A.Compose(
        [A.Resize(height, width)],
        additional_targets={"depth": "mask"},
    )