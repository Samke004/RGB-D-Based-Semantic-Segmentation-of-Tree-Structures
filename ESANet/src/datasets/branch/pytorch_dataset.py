# -*- coding: utf-8 -*-
"""
BRANCH skup za sluzbeni ESANet okvir (TUI-NICR).

Razlike u odnosu na njihove ugradene skupove:

1. Ravna struktura mapa. NYUv2 ima <korijen>/<split>/<RGB|Depth|Label>/, a
   BRANCH ima <korijen>/<RGB|Depth|Label>/ uz train.txt/val.txt/test.txt
   koji nabrajaju imena. _load je prilagoden tome.

2. Oznake se pomicu za +1 (vidi branch.py) jer je klasa 0 ovdje pozadina,
   a ne void.

3. Dubina je uint16 u milimetrima i rijetka (~78 % nula), pa je ispravan
   nacin rada 'raw' — on nakon normalizacije vraca nevazece piksele na nulu.
   Pokreni trening s --raw_depth.

4. depth_mean/std se racunaju iz train splita i spremaju u pickle pokraj ove
   datoteke. Val i test ih citaju odande; ako pickle ne postoji, racunaju se
   iz privremene train instance.
"""

import os

import cv2
import numpy as np

from ..dataset_base import DatasetBase
from .branch import BranchBase


class Branch(BranchBase, DatasetBase):
    def __init__(self,
                 data_dir=None,
                 n_classes=5,
                 split='train',
                 depth_mode='raw',
                 with_input_orig=False,
                 _bez_statistike=False):
        super(Branch, self).__init__()
        assert split in self.SPLITS, f'split mora biti jedan od {self.SPLITS}'
        assert n_classes in self.N_CLASSES
        assert depth_mode in ['refined', 'raw']

        self._n_classes = n_classes
        self._split = split
        self._depth_mode = depth_mode
        self._with_input_orig = with_input_orig
        self._cameras = ['asus_xtion_pro']

        if data_dir is not None:
            data_dir = os.path.expanduser(data_dir)
            assert os.path.exists(data_dir), f'Nema mape: {data_dir}'
            self._data_dir = data_dir

            fp = os.path.join(self._data_dir,
                              self.SPLIT_FILELIST_FILENAMES[self._split])
            assert os.path.exists(fp), f'Nema popisa: {fp}'
            # redci mogu biti goli korijeni ili "RGB/x.png Label/x.png"
            imena = []
            with open(fp) as fh:
                for red in fh:
                    red = red.strip()
                    if not red:
                        continue
                    prvi = red.split()[0]
                    imena.append(os.path.splitext(os.path.basename(prvi))[0])
            self._filenames = np.array(imena, dtype=str)
        else:
            self._data_dir = None
            print(f'Loaded {self.__class__.__name__} dataset without files')

        self._class_names = list(self.CLASS_NAMES)
        self._class_colors = np.array(self.CLASS_COLORS, dtype='uint8')

        if _bez_statistike or self._data_dir is None:
            self._depth_mean, self._depth_std = 0.0, 1.0
        else:
            self._depth_mean, self._depth_std = self._statistika_dubine()

    # ── statistika dubine ───────────────────────────────────────────────────

    def _put_statistike(self):
        return os.path.join(self.source_path,
                            f'depth_{self._depth_mode}_mean_std.pickle')

    def _statistika_dubine(self):
        import pickle
        put = self._put_statistike()

        if os.path.exists(put):
            s = pickle.load(open(put, 'rb'))
            return float(s['mean']), float(s['std'])

        if self._split == 'train':
            racunam = self
        else:
            # val/test ne smiju racunati statistiku iz sebe
            racunam = Branch(data_dir=self._data_dir, n_classes=self._n_classes,
                             split='train', depth_mode=self._depth_mode,
                             _bez_statistike=True)

        print('Racunam mean i std dubine iz train splita...')
        zbroj, broj = np.float64(0), np.uint64(0)
        for i in range(len(racunam)):
            d = racunam.load_depth(i)
            v = d[d > 0] if self._depth_mode == 'raw' else d.flatten()
            zbroj += np.sum(v, dtype=np.float64)
            broj += np.uint64(v.size)
        mean = float(zbroj / broj)

        kvadrati = np.float64(0)
        for i in range(len(racunam)):
            d = racunam.load_depth(i)
            v = d[d > 0] if self._depth_mode == 'raw' else d.flatten()
            kvadrati += np.sum((v.astype(np.float64) - mean) ** 2)
        std = float(np.sqrt(kvadrati / broj))

        pickle.dump({'mean': mean, 'std': std}, open(put, 'wb'))
        print(f'Dubina: mean {mean:.2f} mm, std {std:.2f} mm  -> {put}')
        return mean, std

    # ── svojstva koja okvir trazi ───────────────────────────────────────────

    @property
    def cameras(self):
        return self._cameras

    @property
    def class_names(self):
        return self._class_names

    @property
    def class_names_without_void(self):
        return self._class_names[1:]

    @property
    def class_colors(self):
        return self._class_colors

    @property
    def class_colors_without_void(self):
        return self._class_colors[1:]

    @property
    def n_classes(self):
        return self._n_classes + 1          # +1 za void

    @property
    def n_classes_without_void(self):
        return self._n_classes

    @property
    def split(self):
        return self._split

    @property
    def depth_mode(self):
        return self._depth_mode

    @property
    def depth_mean(self):
        return self._depth_mean

    @property
    def depth_std(self):
        return self._depth_std

    @property
    def source_path(self):
        return os.path.abspath(os.path.dirname(__file__))

    @property
    def with_input_orig(self):
        return self._with_input_orig

    # ── ucitavanje ──────────────────────────────────────────────────────────

    def _load(self, directory, filename):
        """Ravna struktura: <korijen>/<directory>/<filename>.png — bez splita."""
        fp = os.path.join(self._data_dir, directory, f'{filename}.png')
        im = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if im is None:
            raise FileNotFoundError(f'Ne mogu procitati: {fp}')
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

    def load_image(self, idx):
        return self._load(self.RGB_DIR, self._filenames[idx])

    def load_depth(self, idx):
        d = self._load(self.DEPTH_DIR, self._filenames[idx])
        if d.ndim == 3:
            d = d[..., 0]
        return d

    def load_label(self, idx):
        lab = self._load(self.LABELS_DIR, self._filenames[idx])
        if lab.ndim == 3:
            lab = lab[..., 0]
        najveci = int(lab.max())
        if najveci >= self._n_classes:
            raise ValueError(
                f'{self._filenames[idx]}: oznaka {najveci}, a n_classes='
                f'{self._n_classes}. Ocekujem 0..{self._n_classes - 1}.'
            )
        # pomak za +1: void ostaje rezerviran za 0 i nikad se ne pojavljuje
        return (lab.astype(np.uint8) + 1)

    def __len__(self):
        return len(self._filenames)
