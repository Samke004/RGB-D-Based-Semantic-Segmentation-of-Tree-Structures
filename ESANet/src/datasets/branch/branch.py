# -*- coding: utf-8 -*-
"""
Osnovni opis BRANCH skupa za sluzbeni ESANet okvir.

VAZNO — pomak oznaka za +1.

ESANet ocekuje da je klasa 0 "void" (nema oznake) i u gubitku je ignorira:
u CrossEntropyLoss2d stoji `targets_m -= 1` uz `ignore_index=-1`.

U BRANCH skupu klasa 0 je POZADINA, dakle stvarna klasa koja pokriva oko
76 % piksela, a ne odsustvo oznake. Zato se pri ucitavanju sve oznake
pomicu za +1:

    izvorno            ESANet
    0 pozadina    ->   1
    1 deblo       ->   2
    2 grane       ->   3
    3 potpora     ->   4
    4 trava       ->   5
    (void = 0 se nikad ne pojavljuje)

Posljedica za metrike: njihov mIoU preskace samo void, pa prosjecuje preko
svih pet stvarnih klasa UKLJUCUJUCI pozadinu. Rezultati koji se usporeduju
s ostatkom projekta moraju se preracunati iz IoU po klasi, bez pozadine.
"""


class BranchBase:
    SPLITS = ['train', 'valid', 'test']

    # u skupu stoje kao train.txt / val.txt / test.txt
    SPLIT_FILELIST_FILENAMES = {
        'train': 'train.txt',
        'valid': 'val.txt',
        'test': 'test.txt',
    }

    RGB_DIR = 'RGB'
    DEPTH_DIR = 'Depth'
    LABELS_DIR = 'Label'

    N_CLASSES = [5]          # bez voida

    CLASS_NAMES = [
        'void',
        'background',
        'trunk',
        'branches',
        'support',
        'grass',
    ]

    CLASS_COLORS = [
        (0, 0, 0),            # void
        (40, 40, 40),         # pozadina
        (220, 50, 47),        # deblo
        (38, 139, 210),       # grane
        (70, 200, 90),        # potpora
        (240, 200, 40),       # trava
    ]
