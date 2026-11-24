# Segmentacija grana krušaka iz RGB-D snimaka

Semantička segmentacija stabala jabuke iz RGB-D snimaka, 5 klasa:

| indeks | klasa        | opis               |
|:------:|--------------|--------------------|
|   0  | `background` | pozadina / ostalo  |
|   1  | `trunk`      | deblo              |
|   2  | `branches`   | grane              |
|   3  | `support`    | potpora (armatura) |
|   4  | `grass`      | trava              |

Uspoređuje se sedam arhitektura (U-Net ×2, DeepLabv3+ ×2, SegFormer-B1, SegNet, ESANet)
na dva nezavisna splita podataka i dva nasumična sjemena.

---

## 1. Okruženje

```bash
conda create -n branchseg python=3.11
conda activate branchseg

# PyTorch — verzija prema CUDA-i na stroju, vidi pytorch.org
pip install torch torchvision

pip install segmentation-models-pytorch transformers albumentations \
            opencv-python numpy scipy pandas scikit-learn \
            matplotlib pillow tqdm
```

Testirano na Pythonu 3.11, CUDA GPU. Radi i na CPU-u (`config.DEVICE` se sam
prebacuje), ali trening je u tom slučaju nepraktično spor.

---

## 2. Format podataka

Očekuje se sljedeća struktura:

```
data/
├── train/
│   ├── images/   <ime>_rgbd.npy    float32 (H, W, 4) — RGB u 0..1 + dubina u metrima
│   └── masks/    <ime>_mask.png    uint8  (H, W)     — indeksi klasa 0..4
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

Slike i maske se uparuju po imenu bez sufiksa (`X_rgbd.npy` ↔ `X_mask.png`).
Datoteka bez para se preskače i prijavljuje.

Priprema podataka — dvije ulazne točke, ovisno o izvoru:

```bash
# A) iz vlastitih anotacija: split po stablu (bez curenja podataka između splitova)
python prepare_data.py --anotirane /put/do/Anotirane --out_dir data

# B) iz DFormer skupa (RGB/ + Depth/ + Label/ + train|val|test.txt)
#    Podjela se preuzima doslovno iz .txt datoteka, da usporedba s DFormerom bude poštena.
python dformer_u_chen.py --dformer /put/do/BranchDataset --izlaz data

# provjera: broj piksela po klasi, po splitu
python datasetstats.py --data_root data
```

---

## 3. Trening

Jedan model:

```bash
python main.py --model esanet_resnet34 --data data --seed 42 --sufiks split1_seed42
```

Argumenti `main.py`:

| argument            | značenje                                                                  |
|---------------------|---------------------------------------------------------------------------|
| `--model`           | `all` (zadano) ili jedan iz tablice ispod                                 |
| `--data`            | nadjačava `Config.DATA_DIR` i sve izvedene putanje                        |
| `--seed`            | sjeme; bez njega vrijedi `Config.RANDOM_SEED` (42)                        |
| `--sufiks`          | dodaje se na `checkpoints_`, `results_`, `logs_` da se pokretanja ne gaze |
| `--eval_only`       | preskače trening, evaluira iz postojećih checkpointa                      |
| `--deterministicno` | potpuno determinirani cuDNN algoritmi; ~10–30 % sporije                   |

Dostupni modeli (`MODEL_TRAINERS` u `main.py`):

```
unet_resnet34            unet_efficientnet_b4
deeplabv3plus_resnet50   deeplabv3plus_mobilenet
segformer_b1             segnet_vgg16bn
esanet_resnet34
```

MANet i Pix2Pix su implementirani i dostupni u `models/`, ali su zakomentirani
u `MODEL_TRAINERS` — vrati redak ako zatrebaju.

### Cijela kampanja

`kampanja.sh` vrti sve što ulazi u rad: 2 splita × 7 modela sa sjemenom 42,
pa 2 splita × 3 modela sa sjemenom 1337 (procjena varijance sjemena).

```bash
chmod +x kampanja.sh
nohup ./kampanja.sh > kampanja.txt 2>&1 &
tail -f kampanja.txt
```

Svaki model ide kao vlastiti proces: ako jedan padne (OOM, loše težine),
ostali nastave, a VRAM se potpuno oslobodi između modela. Sjeme se vraća prije
svakog modela, pa je rezultat isti kao da je sve išlo u jednom pozivu.
Prekinuta pa ponovno pokrenuta skripta preskače modele koji već imaju
`<model>_best.pth`.

---

## 4. Izlazi

```
checkpoints_<sufiks>/<model>_best.pth      najbolje težine po val mIoU
results_<sufiks>/
├── comparison_table.csv                   glavna tablica — glob_* stupci prvi, pa po-slici
├── <model>_per_image.csv                  metrike po pojedinoj slici
├── <model>_worst<N>.csv                   najgore slike po mIoU
└── *.png                                  krivulje treninga, usporedni grafovi
log/<sufiks>__<model>.txt                  cijeli ispis treninga
```

---

## 5. Analiza i vizualizacija

Glavne brojke već izlaze iz `main.py` — ne treba pokretati ništa dodatno.
Skripte ispod su kontrola i slike.

```bash
# kontrola: isti globalni račun u nezavisnoj implementaciji, uz usporedbu s po-slici
python globalni_miou.py --csv usporedba_globalno.csv

# kontrola: pet nezavisnih implementacija istog računa za jedan model
python provjera_metrika.py --model esanet_resnet34

# vrijeme inferencije po modelu (batch=1, uz warmup i cuda.synchronize) → CSV
python benchmark_inference.py

# 4-panelne ploče: RGB | točna maska | predikcija | greške (bojom propuštene klase)
python vizualizacija_predikcija.py --model esanet_resnet34 --n 15
python vizualizacija_predikcija.py --sortiraj grass        # najgore prvo

# čiste slike bez teksta i okvira, za LaTeX subfigure — za sve modele odjednom
./run_all_vis.sh
```

### Dvije vrste stupaca u tablici

`comparison_table.csv` sadrži obje mjere, globalne prve:

`glob_*` **— globalni račun.** Jedna matrica zabune preko cijelog skupa, IoU
tek na kraju, pa svaki piksel nosi jednaku težinu. Tako računaju DFormer,
mmsegmentation i Cityscapes. **U rad idu ove brojke i samo su one usporedive s
DFormerovima.** Implementacija je u `metrics.py` (`prazna_matrica`,
`dodaj_u_matricu`, `metrike_iz_matrice`), a koriste je i `trainer.py` i
`evaluate.py`. Po globalnom mIoU bez pozadine bira se i najbolji checkpoint
tijekom treninga.

**Ostali stupci — prosjek po slici.** Metrika se računa za svaku sliku zasebno
pa se uzme prosjek, dakle svaka slika ima jednaku težinu bez obzira na broj
piksela. Korisno za uvid u raspršenost i za `*_per_image.csv` / `*_worst<N>.csv`.
Kad klase nema ni u istini ni u predikciji, metrika vraća `nan`, a
`aggregate_metrics` ga izbacuje — ta slika ne ulazi u prosjek za tu klasu.
(Ranije se vraćala nula, pa su rijetke klase — potpora, trava — bile sustavno
podcijenjene.)

`globalni_miou.py` i `provjera_metrika.py` nisu izvor brojki nego nezavisna
provjera da je taj račun ispravan: `provjera_metrika.py` slaže pet
implementacija istog IoU-a jednu uz drugu, uključujući DFormerovu formulu
doslovno i `sklearn.metrics.confusion_matrix`.

---

## 6. Struktura koda

```
main.py                        ulazna točka: trening + evaluacija
config.py                      sve postavke na jednom mjestu
dataset.py                     AppleTreeDataset — učitavanje RGB-D + maski
augmentations.py               albumentations transformacije (train / val)
trainer.py                     petlja treninga, early stopping, AMP
losses.py                      CrossEntropy + Dice, GAN gubici za Pix2Pix
metrics.py                     IoU, Dice, boundary metrike; po slici + globalna matrica zabune
evaluate.py                    evaluacija na testu, tablice i grafovi

models/
├── __init__.py                registar modela
├── unet.py                    U-Net (smp, ResNet34 / EfficientNet-B4)
├── deeplabv3.py               DeepLabv3
├── deeplabv3plus.py           DeepLabv3+
├── segformer.py               SegFormer-B1 (HuggingFace transformers)
├── segnet.py                  SegNet (VGG16-BN)
├── esanet.py                  ESANet — dvogranski RGB-D
├── manet.py                   MANet (neaktivno)
└── pix2pix.py                 Pix2Pix generator + diskriminator (neaktivno)

prepare_data.py                split po stablu
dformer_u_chen.py              konverzija DFormer skupa u ovaj format
datasetstats.py                statistika piksela po klasi
globalni_miou.py               kontrolni globalni mIoU, nezavisno od evaluate.py
provjera_metrika.py            kontrolni izračun u pet implementacija
benchmark_inference.py         mjerenje brzine inferencije
vizualizacija_predikcija.py    ploče s analizom grešaka
visualize_multidepth_clean.py  slike za rad
kampanja.sh                    cijela serija pokretanja
run_all_vis.sh                 vizualizacija za sve modele
```

---

## 7. Napomene

- **Dubina.** `Config.IN_CHANNELS = 4`, dubina je spakirana u `_rgbd.npy` kao četvrti kanal. Zastavice `USE_DEPTH` i `USE_OCCLUSION` odnose se na stare odvojene direktorije i trenutno su `False`. Dubina kroz cijeli put ostaje `float32`, mijenja veličinu s `INTER_NEAREST` (rijetka je, ~88 % nula, pa bi bilinearno miješanje stvorilo dubine koje senzor nikad nije izmjerio) i augmentacijama se predaje kao `mask` target, da je fotometrijske transformacije ne diraju.
- **Odabir checkpointa.** `<model>_best.pth` je model s najboljim globalnim mIoU bez pozadine na validacijskom skupu, isti kriterij kakav koristi DFormer — ne s najmanjim gubitkom.
- **Maske se ne „popravljaju“ tiho.** Maska s indeksom izvan `num_classes` prijavi se kao greška umjesto da se stisne na raspon. U ranijoj verziji je trava (4) tiho postajala potpora (3).
- **Veličina ulaza** je 480 × 640, izvorna veličina projekcije iz PLY-a.
- `ESANet` se uvozi izravno iz `models.esanet`, dok ostali idu kroz `models/__init__.py`.
- `Config.DATA_DIR` je u datoteci postavljen na `data_split2`; `--data` nadjačava i njega i sve izvedene putanje, pa je za pokretanje dovoljno proslijediti taj argument.