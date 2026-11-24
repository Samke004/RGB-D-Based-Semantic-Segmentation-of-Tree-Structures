# Segmentacija grana voćaka iz RGB-D slika — DFormer / DFormerv2

Prilagodba službene implementacije **DFormera** (ICLR 2024) i **DFormerv2** (CVPR 2025) na
vlastiti skup podataka **BranchDataset** — RGB-D snimke stabala jabuke s pet semantičkih
klasa: `ostalo`, `deblo`, `grane`, `potpora`, `trava`.

Izvorni repozitorij: <https://github.com/VCIP-RGBD/DFormer>
Ovdje je zadržana njegova struktura (`models/`, `utils/`, `local_configs/`), a dodano je
sve što se tiče BranchDataseta: konfiguracije, priprema podataka, podjela po stablima,
skripte za trening i evaluaciju te vlastito računanje metrika.

---

## 1. Što je promijenjeno u odnosu na izvorni DFormer

| Promjena                                                                     | Datoteke                                         |
|------------------------------------------------------------------------------|--------------------------------------------------|
| Nova konfiguracija skupa podataka (5 klasa, 480×640)                         | `local_configs/_base_/datasets/BranchDataset.py` |
| Konfiguracije po modelu za BranchDataset                                     | `local_configs/BranchDataset/*.py`               |
| mIoU samo po klasama prvog plana + dodatne metrike                           | `utils/metrics_new.py`                           |
| Zapis metrika po epohama u CSV tijekom treninga                              | `utils/train.py` (funkcija `log_to_csv`)         |
| Pretvorba podataka iz `.npy` formata prethodnog pipelinea u DFormerov format | `data_prepare_DFormer.py`                        |
| Podjela train/val/test **po stablima**, bez propuštanja                      | `split_po_stablima.py`                           |
| Skripta za trening s potpunom arhivom svakog pokretanja                      | `train_all.sh`                                   |
| Skripte za evaluaciju više modela uzastopno                                  | `eval_all.sh`, `eval_all_v2.sh`                  |
| Izvlačenje rezultata iz dovršenih treninga                                   | `rezultati.py`                                   |
| `weights_only=False` u svim pozivima `torch.load`                            | `models/encoders/*.py`, `utils/*.py`             |

Zadnja promjena je samo kompatibilnost: od PyTorcha 2.6 `torch.load` po zadanom odbija
učitati checkpointe koji nisu čisti tenzori, a DFormerove predtrenirane težine to nisu.

---

## 2. Okruženje

```bash
conda create -n dformer python=3.10 -y
conda activate dformer

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118
pip install mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
pip install "numpy<2"
```

Ostali paketi koje kod traži:

```
mmengine  mmsegmentation  easydict  timm  tensorboardX
opencv-python  matplotlib  tqdm  tabulate  pyyaml
```

Napomene:

- `numpy<2` je obavezan — `mmcv` 2.1.0 je preveden za NumPy 1.x.
- `mmsegmentation` je potreban zbog `from mmseg.ops import resize` u `models/decoders/ham_head.py` i `models/decoders/decode_head.py`.
- Točne verzije svih paketa korištene u pojedinom treningu zapisane su u `checkpoints/<model>/INFO.txt` (odjeljak `--- PAKETI ---`), pa je svaki rezultat moguće rekonstruirati bez pogađanja.

---

## 3. Podaci

### Format

Kod očekuje skup podataka u `datasets/BranchDataset/`:

```
datasets/BranchDataset/
├── RGB/      <ime>.png    8-bit RGB, 480×640
├── Depth/    <ime>.png    16-bit jednokanalna dubina u milimetrima
├── Label/    <ime>.png    8-bit maska, vrijednosti 0–4, 255 = ignoriraj
├── train.txt              imena bez ekstenzije, jedno po retku
├── val.txt
└── test.txt
```

Klase (`C.class_names` u `local_configs/_base_/datasets/BranchDataset.py`):

| Oznaka | Klasa                          |
|--------|--------------------------------|
| 0      | ostalo (pozadina)              |
| 1      | deblo                          |
| 2      | grane                          |
| 3      | potpora                        |
| 4      | trava                          |
| 255    | ignoriraj — ne ulazi u metrike |

### Pretvorba iz prethodnog pipelinea

`data_prepare_DFormer.py` pretvara `*_rgbd.npy` (float32 `[H,W,4]`, RGB u 0–1,
dubina u metrima) iz Chenovog pipelinea u gornju strukturu: RGB se skalira na uint8,
dubina množi s 1000 i sprema kao 16-bitni PNG, maske se prepisuju.

Putevi su na početku datoteke, treba ih prilagoditi:

```python
src_base = ".../Pokusaj_Chen_MultiDepthV2/data"
dst_base = ".../DFormer/datasets/BranchDataset"
```

```bash
python data_prepare_DFormer.py
```

### Podjela po stablima

Slike istog stabla su međusobno vrlo slične, pa slučajna podjela po slikama propušta
informaciju iz treninga u test. `split_po_stablima.py` grupira slike po identifikatoru
`DrvoNN` iz imena datoteke i dodjeljuje **cijela stabla** skupovima 70/15/15, počevši od
najvećih stabala, pa provjerava da se nijedno stablo ne pojavljuje u dva skupa.

```bash
cd datasets/BranchDataset
python ../../split_po_stablima.py 42      # 42 = sjeme; ispisuje "PREKLAPANJE STABALA: nema"
```

Prepisuje `train.txt`, `val.txt`, `test.txt` u radnoj mapi.

Trenutna podjela u konfiguraciji: **464 slike za trening, 102 za validaciju**
(`C.num_train_imgs`, `C.num_eval_imgs`).

---

## 4. Predtrenirane težine

Potrebne su težine iz izvornog repozitorija (DFormer je predtreniran na RGB-D podacima,
ne na ImageNetu):

<https://drive.google.com/drive/folders/1YuW7qUtnguUFkhC-sfqGySrerjK0rZJX>

Putevi koje konfiguracije očekuju:

```
checkpoints/pretrained/DFormer_Small.pth.tar
checkpoints/pretrained/DFormer_Base.pth.tar
checkpoints/pretrained/DFormer_Large.pth.tar
checkpoints/pretrained/DFormerv2/pretrained/DFormerv2_Small_pretrained.pth
checkpoints/pretrained/DFormerv2/pretrained/DFormerv2_Base_pretrained.pth
checkpoints/pretrained/DFormerv2/pretrained/DFormerv2_Large_pretrained.pth
```

Put se mijenja preko `C.pretrained_model` u `local_configs/BranchDataset/<model>.py`.

---

## 5. Treniranje

> ### ⚠ PRIJE SVAKOG POKRETANJA: provjeri na kojem je skupu `C.eval_source`
>
> U `local_configs/_base_/datasets/BranchDataset.py` stoje dva para redaka od kojih je
> uvijek jedan zakomentiran. **Kod ne provjerava je li odabrano ono što treba** — samo
> učita ono što nađe i mirno nastavi, pa se greška vidi jedino po broju slika u logu.
>
> **Za TRENING** — aktivan `val.txt`, test zakomentiran:
>
> ```python
> C.eval_source = osp.join(C.dataset_path, "val.txt")
> #C.eval_source = osp.join(C.dataset_path, "test.txt")
> C.num_eval_imgs = 102
> #C.num_eval_imgs = 77
> ```
>
> **Za TESTIRANJE** (`utils/eval.py`) — aktivan `test.txt`, val zakomentiran:
>
> ```python
> #C.eval_source = osp.join(C.dataset_path, "val.txt")
> C.eval_source = osp.join(C.dataset_path, "test.txt")
> #C.num_eval_imgs = 102
> C.num_eval_imgs = 77
> ```
>
> `C.eval_source` i `C.num_eval_imgs` moraju se mijenjati **zajedno** i broj mora
> odgovarati broju redaka u odabranoj `.txt` datoteci (`wc -l`).
>
> Zašto je to važno: ako se trenira s aktivnim `test.txt`, checkpointi se biraju po
> rezultatu na testnom skupu, pa test prestaje biti test i prijavljeni rezultat je
> previsok. Ako se testira s aktivnim `val.txt`, prijavljeni rezultat je rezultat na
> validaciji, a ne na testu.

### Svi modeli uzastopno

```bash
tmux new -s trening          # obavezno kad se radi preko SSH-a ili AnyDeska
bash train_all.sh
```

Korisne opcije:

```bash
bash train_all.sh --proba                              # simulacija, ne trenira ništa
bash train_all.sh --modeli "DFormer_Base DFormerv2_B"  # samo odabrani modeli
bash train_all.sh --dataset /put/do/BranchDataset
bash train_all.sh --izlaz /put/do/checkpoints
```

Prije prvog pokretanja treba u skripti postaviti listu `MODELI`. Dostupne konfiguracije
za BranchDataset: `DFormer_Small`, `DFormer_Base`, `DFormer_Large`,
`DFormerv2_S`, `DFormerv2_B`, `DFormerv2_L`.

### ⚠ DFormer-Large: bez `--val_amp`

Large se ponašao drugačije od ostalih modela — mixed precision u validaciji
(`--val_amp`) rušio mu je mIoU. `train_all.sh` taj prekidač uvijek uključuje, pa se
Large **ne trenira kroz** `train_all.sh`, nego ručno, bez njega:

```bash
export CUDA_VISIBLE_DEVICES=0
PYTHONPATH=".:..:$PYTHONPATH" torchrun --nproc_per_node=1 --master_port=29210 \
    utils/train.py --config=local_configs.BranchDataset.DFormer_Large --gpus=1 \
    --no-sliding --no-compile --no-amp --no-val_amp --no-use_seed
```

Još jedna razlika: u tom slučaju DFormer checkpointe zapisuje izravno u mapu pokretanja,
a ne u podmapu `checkpoint/`, pa ih treba tražiti rekurzivno. Ako mIoU tijekom treninga
padne na nulu dok loss ostaje zdrav, trening je u raspadu i nema smisla ga dovršavati.

### Ručno, jedan model

```bash
export CUDA_VISIBLE_DEVICES=0
PYTHONPATH=".:..:$PYTHONPATH" torchrun --nproc_per_node=1 --master_port=29158 \
    utils/train.py --config=local_configs.BranchDataset.DFormer_Base --gpus=1 \
    --no-sliding --no-compile --no-amp --val_amp --no-use_seed
```

Što ti prekidači znače:

| Prekidač        | Značenje                                                           |
|-----------------|--------------------------------------------------------------------|
| `--no-sliding`  | cijela slika odjednom, bez kliznog okna (480×640 stane u memoriju) |
| `--no-compile`  | bez `torch.compile`                                                |
| `--no-amp`      | trening u punoj preciznosti                                        |
| `--val_amp`     | mixed precision u validaciji (ubrzava; kod Largea izostaviti)      |
| `--no-use_seed` | bez fiksiranja sjemena                                             |

### Hiperparametri

Jednaki za sve modele, u `local_configs/BranchDataset/<model>.py`:

```
optimizator        AdamW
learning rate      6e-5,  polinomni raspad (power 0.9)
weight decay       0.01
epohe              200
warm-up            5 epoha
batch size         4
skaliranje         [0.5, 0.75, 1.0, 1.25, 1.5]
dekoder            ham (LightHamHead), embed_dim 512
checkpoint         od 10. epohe, svakih 10 epoha
```

---

## 6. Što trening ostavlja za sobom

Za svaki model `train_all.sh` složi mapu `checkpoints/<model>/`:

```
checkpoints/DFormer_Base/
├── checkpoint/          .pth datoteke, imenovane epoch-<N>_miou_<X>.pth
├── tb/                  TensorBoard
├── log/                 svi logovi treninga
├── konfiguracija/       kopije .py konfiguracija korištenih u ovom treningu
├── INFO.txt             vrijeme, trajanje, naredba, GPU, verzije paketa, cijeli config
├── metrike.csv           mIoU po epohama, izvučen iz loga
├── metrics_log.csv      metrike po epohama koje zapisuje utils/train.py
├── checkpoints.txt      svi checkpointi, najbolji na vrhu
└── dataset.txt          stanje skupa podataka i raspodjela klasa po pikselima
```

Plus `checkpoints/SAZETAK.csv` — jedan redak po modelu
(`model, pocetak, trajanje_min, status, najbolji_miou, najbolji_checkpoint, mapa`).

Postojeća mapa istog imena ne prepisuje se, nego se premješta u `<model>_staro_<datum>`.

---

## 7. Evaluacija

Evaluacija se pokreće nad konkretnim checkpointom. Primjer:

```bash
export CUDA_VISIBLE_DEVICES=0
PYTHONPATH=".:..:$PYTHONPATH" torchrun --nproc_per_node=1 --master_port=29158 \
    utils/eval.py --config=local_configs.BranchDataset.DFormer_Base \
    --gpus=1 --no-sliding --no-compile --no-amp \
    --continue_fpath="checkpoints/DFormer_Base/checkpoint/epoch-165_miou_88.24.pth"
```

Mijenjaju se dvije stvari: `--config` da odgovara modelu i `--continue_fpath` da pokazuje
na checkpoint koji se ocjenjuje. Najbolji checkpoint po modelu stoji na vrhu
`checkpoints/<model>/checkpoints.txt`, a `--master_port` treba promijeniti samo ako se
pokreće više evaluacija paralelno.

Gotove skripte koje to rade za više modela uzastopno:

| Skripta          | Modeli                         |
|------------------|--------------------------------|
| `eval_all.sh`    | DFormer-Small / Base / Large   |
| `eval_all_v2.sh` | DFormerv2-Small / Base / Large |

**Putanje do checkpointa u njima odnose se na konkretna pokretanja i treba ih zamijeniti
vlastitima.** U `eval_all_v2.sh` je putanja za Large placeholder
(`..._RRRRMMDD-HHMMSS/epoch-NNN_miou_XX.XX.pth`) — taj model još nije istreniran.

### ⚠ Prvo prebaci konfiguraciju na test skup

Evaluacija čita isti `C.eval_source` kao i trening. **Prije pokretanja** `utils/eval.py`
u `local_configs/_base_/datasets/BranchDataset.py` mora biti aktivan `test.txt`, a
`val.txt` zakomentiran:

```python
#C.eval_source = osp.join(C.dataset_path, "val.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
#C.num_eval_imgs = 102
C.num_eval_imgs = 77
```

Nakon testiranja vrijedi vratiti na `val.txt` prije sljedećeg treninga — puni opis i
razlog u uvodu odjeljka 5.

Provjera da je prebacivanje uspjelo: broj slika koji evaluacija ispiše na početku mora
odgovarati `wc -l datasets/BranchDataset/test.txt`.

### Vizualizacija predikcija

Uz `--save_path output/` evaluacija sprema obojene maske predikcija. Paleta se čita iz
`utils/nyucmap.npy`.

---

## 8. Analiza dovršenih treninga

`rezultati.py` ne trenira ništa — prođe kroz `checkpoints/<model>/`, pročita checkpointe
i logove i napiše sažetke:

```bash
python rezultati.py
python rezultati.py --mapa checkpoints
python rezultati.py --cisti      # obriše sve checkpointe osim najboljeg
```

Rezultat: `checkpoints/SAZETAK.csv`, `checkpoints/<model>/krivulja.csv` (mIoU po
epohama, za grafove) i `checkpoints/<model>/checkpoints.txt`.

---

## 9. Kako se računa mIoU

Ovo je važno za usporedbu s brojkama iz izvornog rada, pa je izdvojeno.

`utils/metrics_new.py` akumulira **jednu globalnu matricu zabune** preko cijelog skupa
(`Metrics.update`), a metrike računa na kraju iz nje — ne uprosječuje po slikama. Taj dio
je nepromijenjen iz izvornog DFormera. Promijenjeno je **što se iz te matrice izvlači**:

```
IoU_k    = TP_k / (TP_k + FP_k + FN_k)        iz globalne matrice, jednako kao upstream
mIoU     = srednji IoU klasa 1–4              ← ovdje: BEZ pozadine
                                                upstream: prosjek preko svih klasa
```

Prijavljeni **mIoU je prosjek samo po klasama prvog plana** — `deblo`, `grane`,
`potpora`, `trava`. Pozadina (`ostalo`) zauzima najveći dio svake slike i IoU joj je oko
98 %, pa bi njezino uključivanje podiglo brojku za oko tri postotna poena i učinilo je
manje informativnom.

**Posljedica: mIoU iz ovog repozitorija nije izravno usporediv s mIoU iz DFormerovog
rada** (NYUv2, SUNRGB-D), gdje pozadina ulazi u prosjek. Pri svakoj usporedbi s
literaturom treba navesti koji je skup klasa u prosjeku.

Razlike u detalju, u odnosu na izvorni `utils/metrics_new.py`:

|                           | izvorni DFormer                              | ovdje                                            |
|---------------------------|----------------------------------------------|--------------------------------------------------|
| akumulacija matrice       | globalna, preko cijelog skupa                | nepromijenjeno                                   |
| mIoU                      | prosjek preko svih klasa                     | prosjek klasa 1–4, bez pozadine                  |
| klasa bez piksela u skupu | `ious[isnan] = 0` → ulazi u prosjek kao nula | `+ 1e-6` u djelitelju, isti efekt bez NaN-a      |
| `compute_pixel_acc`       | srednja točnost po klasama (mAcc)            | ukupna točnost po pikselima (OA)                 |
| dodatne metrike           | —                                            | `pixel_acc_fg`, `fwiou`, `fwiou_fg`, `cl_recall` |

Pikseli s oznakom 255 isključeni su iz svega.

Dvije nedosljednosti koje treba imati na umu pri pisanju tablica:

- `compute_pixel_acc` **ne vraća mAcc**, nego ukupnu točnost po pikselima (`diag.sum() / total`). `utils/train.py` i `utils/eval.py` tu vrijednost ispisuju pod imenom `macc`, pa u logu stoji `macc` uz brojku koja je ukupna točnost. U tablicama taj stupac treba označiti kao **OA (ukupna točnost)**, ne mAcc. Srednji recall po klasama dostupan je u `cl_recall` iz `compute_stats`.
- `compute_f1` **uprosječuje preko svih klasa, uključujući pozadinu**, pa mF1 i mIoU nisu računati nad istim skupom klasa.

---

## 11. Struktura repozitorija

```
.
├── local_configs/
│   ├── _base_/
│   │   ├── __init__.py                  osnovna konfiguracija (C, putevi, sjeme)
│   │   └── datasets/
│   │       ├── BranchDataset.py         ← naš skup: klase, putevi, veličine
│   │       ├── NYUDepthv2.py            iz izvornog repozitorija
│   │       └── SUNRGBD.py               iz izvornog repozitorija
│   ├── BranchDataset/                   ← konfiguracije po modelu za naš skup
│   ├── NYUDepthv2/, SUNRGBD/            iz izvornog repozitorija
│   └── template/
│
├── models/
│   ├── builder.py                       spaja enkoder i dekoder prema konfiguraciji
│   ├── net_utils.py
│   ├── encoders/
│   │   ├── DFormer.py                   DFormer Tiny/Small/Base/Large
│   │   └── DFormerv2.py                 DFormerv2 S/B/L
│   ├── decoders/
│   │   ├── ham_head.py                  ← dekoder koji koristimo (C.decoder = "ham")
│   │   ├── decode_head.py, fcnhead.py   potrebni ham_headu
│   │   └── MLPDecoder.py, UPernet.py, deeplabv3plus.py, nl_head.py, LMLPDecoder.py
│   │                                    alternative iz izvornog repozitorija
│   └── losses/                          funkcije gubitka (cross-entropy je u upotrebi)
│
├── utils/
│   ├── train.py                         trening (ulazna točka)
│   ├── eval.py                          evaluacija (ulazna točka)
│   ├── val_mm.py                        validacijska petlja, jedno- i više-skalna
│   ├── metrics_new.py                   ← metrike iz globalne matrice zabune
│   ├── dataloader/                      RGBXDataset + dataloaderi
│   ├── transforms.py                    augmentacije
│   ├── engine/                          engine.py (stanje treninga), logger.py
│   ├── init_func.py, lr_policy.py       inicijalizacija težina, raspored lr-a
│   ├── pyt_utils.py, load_utils.py      distribuirani pomoćnici, učitavanje težina
│   └── nyucmap.npy                      paleta za vizualizaciju
│
├── data_prepare_DFormer.py              .npy → RGB/Depth/Label PNG
├── split_po_stablima.py                 podjela train/val/test po stablima
├── rezultati.py                         izvlačenje rezultata iz dovršenih treninga
│
├── train_all.sh                         trening modela + arhiviranje pokretanja
├── eval_all.sh                          evaluacija DFormer Small / Base / Large
├── eval_all_v2.sh                       evaluacija DFormerv2 Small / Base / Large
│
├── train.sh, eval.sh, infer.sh          iz izvornog repozitorija, pokazuju na NYUv2
├── test_models.sh, test_v2.sh           kratki dimni testovi (pokrenu i prekinu trening)
├── train_v2.sh                          starija varijanta train_all.sh
│
├── README.md
└── LICENSE                              iz izvornog repozitorija
```

Mapa `checkpoints/` ne postoji unaprijed — stvara se pri prvom pokretanju. `datasets/`
sadrži BranchDataset u strukturi iz odjeljka 3; predtrenirane težine treba dobaviti kako
je opisano u odjeljku 4.

---

## 12. Na što paziti

- `C.eval_source` **se prebacuje ručno između** `val.txt` **i** `test.txt` i kod to ne provjerava. Trening mora ići s `val.txt`, evaluacija s `test.txt`, a `C.num_eval_imgs` mijenja se zajedno s putem. Detaljno u uvodu odjeljka 5. Ovo je najlakša greška u cijelom pipelineu i ne prijavljuje se nikakvom porukom — vidi se samo po broju slika u logu.
- **CSV zapis metrika pokriva samo četiri klase.** `log_to_csv` u `utils/train.py` ima stupce `iou_background`, `iou_trunk`, `iou_branches`, `iou_potpora` — IoU za klasu `trava` (indeks 4) ne zapisuje se u `metrics_log.csv`. Sam mIoU u stupcu `mean_iou`**jest** računat preko svih pet klasa, pa su mIoU i stupci po klasama nedosljedni. Potpune vrijednosti po klasama ostaju u logu treninga.
- `mean_boundary_f1` **u CSV-u je uvijek 0** — nije implementiran, stupac je rezerviran.
- **Protokol evaluacije nije jednak za sve modele.** `DFormer_Base.py` ima `C.eval_scale_array = [0.5, 0.75, 1, 1.25, 1.5]`, svi ostali `[1]`. Više-skalna evaluacija daje bolji rezultat, pa brojke nisu izravno usporedive ako se evaluira s `--mst`. Prije usporedne tablice vrijedi izjednačiti tu postavku.
- **Large se ne trenira kroz** `train_all.sh` — treba mu `--no-val_amp`, vidi odjeljak 5.
- **Checkpointi zauzimaju puno prostora** — svakih 10 epoha od 10. do 200., po modelu. `python rezultati.py --cisti` ostavlja samo najbolji.
- **Reproducibilnost.** Skripte se pokreću s `--no-use_seed`, pa pokretanja nisu bit-po-bit ponovljiva. Za ponovljivost treba izostaviti taj prekidač (sjeme je `C.seed = 12345` u `local_configs/_base_/__init__.py`).
- **Konfiguracije za NYUv2 i SUNRGB-D** (`local_configs/NYUDepthv2/`, `local_configs/SUNRGBD/`) ostavljene su iz izvornog repozitorija kao referenca. Za ovaj rad relevantne su samo one u `local_configs/BranchDataset/`.
- `train.sh`**,** `eval.sh`**,** `infer.sh` su iz izvornog repozitorija i pokazuju na NYUv2 / SUNRGB-D konfiguracije; `test_models.sh` i `test_v2.sh` su dimni testovi koji pokrenu trening i prekinu ga nakon minute-dvije; `train_v2.sh` je starija varijanta `train_all.sh`. Nijedna nije dio rezultata ovog rada.

---

## 13. Licenca i citiranje

Kod je izveden iz službene implementacije DFormera; licenca izvornog repozitorija je u
`LICENSE` i primjenjuje se na sve preuzete datoteke.

```bibtex
@inproceedings{yin2024dformer,
  title={DFormer: Rethinking RGBD Representation Learning for Semantic Segmentation},
  author={Yin, Bowen and Zhang, Xuying and Li, Zhong-Yu and Liu, Li and Cheng, Ming-Ming and Hou, Qibin},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}

@inproceedings{yin2025dformerv2,
  title={DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation},
  author={Yin, Bo-Wen and Cao, Jiao-Long and Cheng, Ming-Ming and Hou, Qibin},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```