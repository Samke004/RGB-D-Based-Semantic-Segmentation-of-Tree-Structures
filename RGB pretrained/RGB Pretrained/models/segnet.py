"""
models/segnet.py
────────────────────────────────────────────────────────────────────────────
SegNet (Badrinarayanan et al., 2017) s VGG16-BN enkoderom, ImageNet tezine.

Zasto bas SegNet za grane: za razliku od U-Neta koji prenosi cijele mape
znacajki preskocnim vezama, SegNet pamti SAMO indekse max-poolinga i njima
vraca vrijednosti na tocne pozicije pri odpoolavanju. Rubovi tankih
struktura ostaju ostri, a memorija je bitno manja nego kod U-Neta.

Prilagodbe za ovaj skup:
  - 4 ulazna kanala (RGB + depth): prvi konvolucijski sloj se prosiruje,
    cetvrti kanal dobiva prosjek RGB tezina umjesto nasumicnih vrijednosti,
    pa depth od pocetka ima smislen odziv umjesto suma.
  - broj izlaznih klasa je parametar.
"""

import torch
import torch.nn as nn
from torchvision.models import vgg16_bn


def _blok(ulaz, izlaz):
    return nn.Sequential(
        nn.Conv2d(ulaz, izlaz, kernel_size=3, padding=1),
        nn.BatchNorm2d(izlaz),
        nn.ReLU(inplace=True),
    )


class SegNet(nn.Module):

    # broj konvolucija po stupnju, kao u VGG16
    STUPNJEVI = [2, 2, 3, 3, 3]
    SIRINE = [64, 128, 256, 512, 512]

    def __init__(self, in_channels=4, num_classes=5, pretrained=True,
                 drop_rate=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # ── Enkoder ─────────────────────────────────────────────────────────
        enk = []
        prethodna = in_channels
        for n, sirina in zip(self.STUPNJEVI, self.SIRINE):
            slojevi = []
            for i in range(n):
                slojevi.append(_blok(prethodna if i == 0 else sirina, sirina))
            enk.append(nn.Sequential(*slojevi))
            prethodna = sirina
        self.enkoder = nn.ModuleList(enk)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)

        # ── Dekoder - zrcalo enkodera ───────────────────────────────────────
        self.unpool = nn.MaxUnpool2d(2, 2)
        dek = []
        ulazne = list(reversed(self.SIRINE))   # 512 512 256 128 64
        izlazne = [512, 256, 128, 64, 64]      # zadnja konvolucija svakog stupnja
        for n, u, i in zip(reversed(self.STUPNJEVI), ulazne, izlazne):
            slojevi = []
            for k in range(n):
                zadnji = (k == n - 1)
                slojevi.append(_blok(u, i if zadnji else u))
            dek.append(nn.Sequential(*slojevi))
        self.dekoder = nn.ModuleList(dek)

        self.drop = nn.Dropout2d(drop_rate) if drop_rate > 0 else nn.Identity()
        self.glava = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)

        self._inicijaliziraj()
        if pretrained:
            self._ucitaj_vgg()

    # ── inicijalizacija ─────────────────────────────────────────────────────

    def _inicijaliziraj(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _ucitaj_vgg(self):
        """Prenosi VGG16-BN ImageNet tezine u enkoder."""
        try:
            vgg = vgg16_bn(weights="IMAGENET1K_V1")
        except TypeError:                      # starije torchvision verzije
            vgg = vgg16_bn(pretrained=True)

        vgg_conv = [m for m in vgg.features if isinstance(m, nn.Conv2d)]
        vgg_bn = [m for m in vgg.features if isinstance(m, nn.BatchNorm2d)]

        moj_conv, moj_bn = [], []
        for stupanj in self.enkoder:
            for blok in stupanj:
                moj_conv.append(blok[0])
                moj_bn.append(blok[1])

        preneseno = 0
        for i, (src_c, dst_c, src_b, dst_b) in enumerate(
                zip(vgg_conv, moj_conv, vgg_bn, moj_bn)):

            w = src_c.weight.data
            if i == 0 and self.in_channels != 3:
                if self.in_channels < 3:
                    w = w[:, :self.in_channels]
                else:
                    # dodatni kanali (depth) dobivaju prosjek RGB tezina,
                    # skaliran da ukupni odziv ostane slican izvornom
                    dodatnih = self.in_channels - 3
                    prosjek = w.mean(dim=1, keepdim=True).repeat(1, dodatnih, 1, 1)
                    w = torch.cat([w, prosjek], dim=1) * (3.0 / self.in_channels)

            if w.shape == dst_c.weight.shape:
                dst_c.weight.data.copy_(w)
                if src_c.bias is not None and dst_c.bias is not None:
                    dst_c.bias.data.copy_(src_c.bias.data)
                dst_b.load_state_dict(src_b.state_dict())
                preneseno += 1

        print(f"SegNet: preneseno {preneseno}/{len(moj_conv)} VGG16-BN slojeva "
              f"(ulaznih kanala: {self.in_channels})")

    # ── prolaz ──────────────────────────────────────────────────────────────

    def forward(self, x):
        indeksi, velicine = [], []

        for stupanj in self.enkoder:
            x = stupanj(x)
            velicine.append(x.shape)
            x, idx = self.pool(x)
            indeksi.append(idx)

        x = self.drop(x)

        for stupanj, idx, vel in zip(self.dekoder,
                                     reversed(indeksi),
                                     reversed(velicine)):
            x = self.unpool(x, idx, output_size=vel[-2:])
            x = stupanj(x)

        return self.glava(x)


def build_segnet(in_channels=4, num_classes=5, pretrained=True, drop_rate=0.0):
    """Potpis usklađen s ostalim build_* funkcijama u models/."""
    return SegNet(in_channels=in_channels, num_classes=num_classes,
                  pretrained=pretrained, drop_rate=drop_rate)
