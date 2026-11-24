# Segmentacija grana voćaka iz RGB-D slika — službeni ESANet

Službena implementacija **ESANeta** (TUI-NICR, IEEE ICRA 2021) s ugrađenim vlastitim
skupom podataka **BRANCH** — RGB-D snimke stabala jabuke s pet semantičkih klasa:
pozadina, deblo, grane, potpora, trava.

Izvorni repozitorij: <https://github.com/TUI-NICR/ESANet>

## Čemu ovaj repozitorij postoji

U projektu postoje dvije ESANet varijante i one odgovaraju na različita pitanja:

|                                           | arhitektura  | predtreniranje                        | pipeline     |
|-------------------------------------------|--------------|---------------------------------------|--------------|
| vlastita implementacija (`RGBPretrained`) | vlastita     | ImageNet                              | vlastiti     |
| **ovaj repozitorij**                      | **službena** | **ImageNet ili NYUv2+SceneNet RGB-D** | **službeni** |

Smisao je **izolirati učinak predtreniranja**. Ako se usporede vlastita ESANet
implementacija s ImageNet težinama i službena ESANet s RGB-D težinama, razlikuju se dvije
stvari odjednom — arhitektura i izvor predtreniranja — i ne zna se koja je proizvela
razliku. Zato se ovdje unutar **istog** repozitorija, istog pipelinea i iste podjele
podataka pokreću dva treninga koja se razlikuju **samo po početnim težinama**.

Naredbe za ta dva treninga su u odjeljku 5, a pokretanja koja su izvedena popisana su u
odjeljku 8. Brojke su u radu, ne ovdje.

---

## 1. Što je promijenjeno u odnosu na izvorni ESANet

Promijenjeno je **pet datoteka**, dodana je **jedna mapa** i **jedna skripta**. Sve
ostalo je nepromijenjeno. Originali prve četiri datoteke spremljeni su kao
`*.prije_branch`, pa je svaka izmjena provjerljiva s `diff`.

| Datoteka                   | Izmjena                                                           | Kako je napravljena |
|----------------------------|-------------------------------------------------------------------|---------------------|
| `src/datasets/branch/`     | **nova mapa** — definicija BRANCH skupa                           | ručno               |
| `src/datasets/__init__.py` | izvozi `Branch`                                                   | `ugradi_branch.py`  |
| `src/prepare_data.py`      | grana za `--dataset branch` (`n_classes=5`, `valid_set='valid'`)  | `ugradi_branch.py`  |
| `src/args.py`              | `'branch'` dodan u dopuštene vrijednosti `--dataset`              | `ugradi_branch.py`  |
| `src/build_model.py`       | filtrirano učitavanje `--finetune` + uvjetno zamrzavanje enkodera | **ručno**           |
| `eval.py`                  | `--split` + ispis IoU po klasi i mIoU bez pozadine                | **ručno**           |

`ugradi_branch.py` je skripta koja prve tri izmjene radi idempotentno — provjeri je li
već ugrađeno, pa je sigurno pokrenuti dvaput:

```bash
python ugradi_branch.py
```

`src/build_model.py` **i** `eval.py` **ona ne dira** — te su dvije izmjene ručne i opisane su
niže (odjeljci 1.2 i 1.3), jer su bitne za ispravnost rezultata.

### 1.1 Pomak oznaka za +1

Ovo je najvažnija pojedinost cijele integracije.

ESANet pretpostavlja da je **klasa 0 "void"** — odsustvo oznake — i u gubitku je ignorira
(`CrossEntropyLoss2d` radi `targets_m -= 1` uz `ignore_index=-1`).

U BRANCH skupu je klasa 0 **pozadina**, dakle stvarna klasa koja pokriva oko 76 % piksela.
Da se ne bi tiho izgubila, `Branch.load_label` pomiče sve oznake za +1:

| u datoteci | u ESANetu                         |
|------------|-----------------------------------|
| —          | 0 = void (nikad se ne pojavljuje) |
| 0 pozadina | 1                                 |
| 1 deblo    | 2                                 |
| 2 grane    | 3                                 |
| 3 potpora  | 4                                 |
| 4 trava    | 5                                 |

`n_classes` je 5 (bez voida); `Branch.n_classes` vraća 6, `n_classes_without_void` vraća 5.

**Posljedica za metrike je u odjeljku 7** i mora se pročitati prije bilo kakve usporedbe
brojki s ostatkom projekta.

### 1.2 `src/build_model.py` — dva ispravka

**Filtrirano učitavanje** `--finetune`**.** Original je radio:

```python
checkpoint = torch.load(args.finetune)
model.load_state_dict(checkpoint['state_dict'])          # strict=True
```

To pada na NYUv2 checkpointu jer je njegova izlazna glava za 40 klasa, a naša za 5.
Sada se preuzimaju samo tenzori koji se poklapaju imenom i oblikom, uz `strict=False`,
a prefiks `module.` (od `DataParallel`) se skida:

```python
uzeto = {k: v for k, v in sd.items()
         if k in trenutno and trenutno[k].shape == v.shape}
model.load_state_dict(uzeto, strict=False)
```

Skripta ispiše koliko je preuzeto. Za NYUv2+SceneNet checkpoint:

```
Loaded weights for finetuning: trained_models/nyuv2/r34_NBt1D_scenenet.pth
  preuzeto 886/898 tenzora, preskoceno 12
    decoder.conv_out.bias
    decoder.conv_out.weight
    decoder.decoder_module_1.side_output.bias
    decoder.decoder_module_1.side_output.weight
    decoder.decoder_module_2.side_output.bias
    decoder.decoder_module_2.side_output.weight
```

Preskočeno je točno ono što treba biti preskočeno — izlazne glave i pomoćni izlazi — i te
težine ostaju nasumično inicijalizirane. **Ovaj ispis treba provjeriti pri svakom
pokretanju s **`--finetune`**:** ako broj padne znatno niže od 886, nešto se u modelu
promijenilo i prijenos težina više nije onakav kakav se misli.

**Uvjetno zamrzavanje enkodera.** Original je enkodere zamrzavao bezuvjetno kad je
zadan `--finetune`:

```python
print('Freeze the encoder(s).')
for name, param in model.named_parameters():
    if 'encoder_rgb' in name or 'encoder_depth' in name or 'se_layer' in name:
        param.requires_grad = False
```

Problem je što ih `train.py` odmrzava samo pod uvjetom `args.finetune is None`, pa bi se
s `--finetune` **trenirao samo dekoder, kroz sve epohe**. Sada je zamrzavanje vezano na
`--freeze`:

```python
if args.freeze > 0:
    print('Freeze the encoder(s).')
    ...
```

Zadano je `--freeze 0`, dakle bez zamrzavanja. Ako se u ispisu pojavi
`Freeze the encoder(s).`, a nije zadan `--freeze`, nešto nije u redu.

### 1.3 `eval.py` — dva dodatka

- `--split {valid,test}`, zadano `test`. Bez toga `prepare_data` uzima `valid_set`, koji je za branch `'valid'`, pa bi se mjerilo na validacijskom skupu.
- **Ispis IoU po klasi te mIoU u dvije varijante** — po njihovoj konvenciji (s pozadinom) i bez pozadine. Vidi odjeljak 7.

---

## 2. Okruženje

U repozitoriju su tri opisa okruženja, a **nijedan nije onaj u kojem su priložena
pokretanja izvedena** — logovi pokazuju conda okruženje `treepartnet` s Pythonom 3.8.
Prije objave to treba uskladiti.

| Datoteka                  | Što je                                    | Python / torch |
|---------------------------|-------------------------------------------|----------------|
| `rgbd_segmentation.yaml`  | izvorno okruženje autora ESANeta          | 3.7 / 1.3.1    |
| `env_esanet2025.yaml`     | novije okruženje iz njihovog repozitorija | 3.12 / 2.9.1   |
| `env_esanet2025_mac.yaml` | isto, za macOS                            | 3.12 / 2.9.1   |
| `requirements_jetson.txt` | za NVIDIA Jetson                          | —              |

```bash
conda env create -f env_esanet2025.yaml
conda activate esanet2025
```

Zavisnosti koje kod stvarno traži: `torch`, `torchvision`, `numpy`, `opencv-python`,
`tensorboardX`, `tqdm`, `matplotlib`. `cityscapesscripts` traži samo Cityscapes skup —
zato `ugradi_branch.py` u `prepare_data.py` uvozi `Branch` izravno iz podmodula, da uvoz
ne povuče Cityscapes i ne padne ako taj paket nije instaliran.

---

## 3. Podaci

### Format

Skup ima **ravnu strukturu** — različito od njihovih ugrađenih skupova, koji imaju
`<korijen>/<split>/<RGB|Depth|Label>/`:

```
BranchDataset/
├── RGB/      <ime>.png    8-bit RGB
├── Depth/    <ime>.png    16-bit jednokanalna dubina u MILIMETRIMA
├── Label/    <ime>.png    8-bit maska, vrijednosti 0–4
├── train.txt              imena, jedno po retku
├── val.txt                (split 'valid' ga čita pod tim imenom)
└── test.txt
```

`Branch._load` je prilagođen tome, a `_filenames` prihvaća i gole korijene imena i
retke oblika `RGB/x.png Label/x.png` — uzima prvi stupac i odbacuje ekstenziju.

Imena splitova u kodu i imena datoteka se razlikuju: split `valid` čita `val.txt`
(`BranchBase.SPLIT_FILELIST_FILENAMES`).

### Klase

| Indeks nakon pomaka | CLASS_NAMES  | Hrvatski                   |
|---------------------|--------------|----------------------------|
| 0                   | `void`       | — (nikad se ne pojavljuje) |
| 1                   | `background` | pozadina                   |
| 2                   | `trunk`      | deblo                      |
| 3                   | `branches`   | grane                      |
| 4                   | `support`    | potpora                    |
| 5                   | `grass`      | trava                      |

`load_label` provjerava da najveća oznaka u datoteci nije ≥ `n_classes` i javlja grešku s
imenom datoteke ako jest — tako se pokvarena maska otkrije odmah, a ne kroz čudne metrike.

### Dubina

Dubina je uint16 u milimetrima i **rijetka — oko 78 % piksela je nula** (nema mjerenja).
Zato se trenira s `--raw_depth`: taj način nakon normalizacije vraća nevažeće piksele
na nulu, umjesto da model uči predviđati interpolirane vrijednosti koje nisu izmjerene.

Statistika dubine računa se **samo iz train splita** i sprema u pickle pokraj definicije
skupa. Val i test je čitaju odande; ako pickle ne postoji, stvara se privremena train
instanca da ga izračuna — val i test nikad ne računaju statistiku iz sebe.

`src/datasets/branch/depth_raw_mean_std.pickle`:

```
mean = 1266.95 mm
std  =  466.25 mm
```

### Težine klasa

ESANet otežava gubitak po klasama. Pickle datoteke su u `src/datasets/branch/` i kod ih
koristi ako postoje, inače ih izračuna i spremi.

| Datoteka                                      | Što sadrži                                                | Gdje se koristi                                            |
|-----------------------------------------------|-----------------------------------------------------------|------------------------------------------------------------|
| `weighting_median_frequency_1+5_train.pickle` | težine klasa po `median_frequency` shemi, iz train splita | `CrossEntropyLoss2d` u treningu                            |
| `weighting_linear_1+5_valid.pickle`           | broj piksela po klasi u validacijskom skupu               | normalizacija validacijskog gubitka, nije težina u gubitku |

`median_frequency` shema (`dataset_base.compute_class_weights`) računa frekvenciju klase
kao broj njezinih piksela podijeljen s brojem piksela slika u kojima se ta klasa pojavljuje,
pa težinu kao medijan tih frekvencija podijeljen s frekvencijom klase. Rijetke klase tako
dobiju veću težinu, a pozadina znatno manju od 1.

**Oba pickle-a vrijede za konkretnu podjelu podataka.** Ako se podjela promijeni, treba ih
obrisati da se preračunaju — inače se tiho koriste stare težine. Ime datoteke sadrži samo
`1+5` i ime splita, ne i otisak podjele, pa se neslaganje ne može otkriti automatski.

---

## 4. Predtrenirane težine

| Datoteka                                      | Što je                                                 | Koristi se za                      |
|-----------------------------------------------|--------------------------------------------------------|------------------------------------|
| `trained_models/imagenet/r34_NBt1D.pth`       | ResNet-34 s NonBottleneck1D blokovima, ImageNet        | zadano početno stanje enkodera     |
| `trained_models/nyuv2/r34_NBt1D_scenenet.pth` | cijeli ESANet, predtreniran na SceneNet RGB-D pa NYUv2 | `--finetune`, RGB-D predtreniranje |
| `trained_models/imagenet_r34_NBt1D.tar.gz`    | arhiva ImageNet težina                                 | — (raspakirana verzija je gore)    |

Preuzimaju se s njihovog repozitorija. ImageNet težine enkodera učitavaju se **uvijek**,
osim uz `--no_imagenet_pretraining`; `--finetune` se primjenjuje **nakon** toga i
prepisuje ono što se poklapa. Zato i u logu RGB-D treninga na početku stoji
`Loaded r34 ... pretrained on ImageNet` — to nije greška, nego redoslijed učitavanja.

---

## 5. Treniranje

### ImageNet predtreniranje (kontrola)

```bash
python train.py \
  --dataset branch \
  --dataset_dir /put/do/BranchDataset \
  --results_dir ./results_branch_imagenet \
  --raw_depth \
  --height 480 --width 640 \
  --batch_size 4 \
  --epochs 200 \
  --modality rgbd
```

### RGB-D predtreniranje (NYUv2 + SceneNet)

Ista naredba, dodan `--finetune`:

```bash
python train.py \
  --dataset branch \
  --dataset_dir /put/do/BranchDataset \
  --results_dir ./results_branch_rgbd \
  --finetune trained_models/nyuv2/r34_NBt1D_scenenet.pth \
  --raw_depth \
  --height 480 --width 640 \
  --batch_size 4 \
  --epochs 200 \
  --modality rgbd
```

> **Za valjanu usporedbu te dvije naredbe smiju se razlikovati samo po** `--finetune` **i**`--results_dir`**.** Sve ostalo — podjela podataka, broj epoha, batch, otežavanje klasa,
> razlučivost — mora biti identično, podjela podataka prije svega: dva modela trenirana na
> različitim podjelama ne smiju se naći u istoj tablici.

### Hiperparametri

Zadane vrijednosti iz `src/args.py`, uz ono što je u priloženim pokretanjima doista
korišteno (očitano iz logova):

|                             | Vrijednost                                            | Izvor                                                    |
|-----------------------------|-------------------------------------------------------|----------------------------------------------------------|
| optimizator                 | SGD, momentum 0.9                                     | zadano                                                   |
| learning rate               | **0.005**                                             | `0.01 × batch/8` — kod ga sam prilagodi batchu i to javi |
| lr raspored                 | OneCycleLR, `div_factor=25`, `pct_start=0.1`          | zadano                                                   |
| weight decay                | 1e-4                                                  | zadano                                                   |
| epohe                       | 200                                                   | `--epochs 200` (zadano je 500)                           |
| batch size                  | 4                                                     | `--batch_size 4` (zadano je 8)                           |
| otežavanje klasa            | `median_frequency`                                    | zadano                                                   |
| enkoder                     | resnet34, blok NonBottleneck1D                        | zadano                                                   |
| kontekstni modul            | `ppm`                                                 | zadano                                                   |
| fuzija dubine u RGB enkoder | `SE-add`                                              | zadano                                                   |
| dekoder                     | `decreasing` kanali, `learned-3x3-zeropad` upsampling | zadano                                                   |
| razlučivost                 | 480 × 640                                             | zadano                                                   |
| augmentacija                | slučajno skaliranje 1.0–1.4                           | zadano                                                   |

`--channels_decoder` se ignorira kad je `--decoder_channels_mode decreasing`, pa
upozorenje o tome u logu nije problem.

---

## 6. Što trening ostavlja za sobom

Mapa `<results_dir>/branch/checkpoints_<DD_MM_YYYY-HH_MM_SS-mikrosekunde>/`:

```
├── args.json                 svi argumenti, serijalizirani
├── argsv.txt                 točna naredba kako je pokrenuta  ← za reproducibilnost
├── logs.csv                  gubitak i mIoU po epohama
├── confusion_matrices/        matrice zabune
├── depth_mean_std.pickle     statistika dubine za ovo pokretanje
├── ckpt_epoch_<N>.pth        snimljen SAMO kad mIoU nadmaši dotad najbolji, i od 10. epohe
├── ckpt_latest.pth           prepisuje se svaku epohu; sadrži best_miou i best_miou_epoch
└── finished.txt              najbolji mIoU i epoha u kojoj je postignut
```

Dva su korisna svojstva: `argsv.txt` čuva točnu naredbu, pa se pokretanje može ponoviti
bez pamćenja; a `ckpt_epoch_<N>.pth` postoji samo za epohe koje su bile rekord, pa je
najveći `N` u mapi istovremeno i najbolji model.

**Checkpointi zauzimaju puno prostora.** Korisno je zadržati samo najbolji i jedan
rezervni:

```bash
ls -v <ckpt_dir>/ckpt_epoch_*.pth | head -n -2 | xargs -r rm --
```

---

## 7. Metrike

`src/metric.py` (`MeanIntersectionOverUnion`) akumulira **globalnu matricu zabune** preko
cijelog skupa i IoU računa na kraju iz nje. Ne uprosječuje po slikama.

```
IoU_k = TP_k / (TP_k + FP_k + FN_k)       iz globalne matrice
```

Dvije stvari u ponašanju te klase vrijedi znati:

**Klase bez piksela u označenim podacima se preskaču.** `compute()` maskira klase s
`sum_gt == 0` prije prosjeka, pa odsutna klasa ne ulazi u prosjek kao nula. Isto radi
`np.nanmean` u `eval.py`.

**Prosjek po zadanom ide preko svih klasa osim voida.** `ignore_first_class` ostaje
`False`, a oznake su već pomaknute za +1 (odjeljak 1.1), pa void nikad ne postoji i
prosjek pokriva svih pet stvarnih klasa — **uključujući pozadinu**, koja pokriva oko 76 %
piksela i ima IoU blizu 1.

Zato se razlikuju dvije brojke:

|                       | Konvencija       | Gdje se pojavljuje                       |
|-----------------------|------------------|------------------------------------------|
| mIoU tijekom treninga | **s pozadinom**  | `logs.csv`, `finished.txt`, ispis u logu |
| mIoU iz evaluacije    | **bez pozadine** | ispis `eval.py`                          |

mIoU koji trening ispisuje i po kojem bira najbolji checkpoint **uključuje pozadinu** —
okvir ne zna o tome da je pozadina kod nas stvarna klasa, a kriterij odabira checkpointa
nije mijenjan. `eval.py` je izmijenjen da pozadinu izbaci, pa je brojka iz evaluacije ona
koja se prijavljuje:

```
IoU po klasi:
  background      ...
  trunk           ...
  branches        ...
  support         ...
  grass           ...

mIoU (sve klase, njihova konvencija): ...
mIoU BEZ pozadine (za usporedbu):     ...
```

### Evaluacija

```bash
python eval.py \
  --dataset branch \
  --dataset_dir /put/do/BranchDataset \
  --ckpt_path <results_dir>/branch/checkpoints_<...>/ckpt_epoch_<N>.pth \
  --split test \
  --raw_depth \
  --height 480 --width 640 \
  --modality rgbd
```

`--raw_depth`, `--height`, `--width` i `--modality` moraju odgovarati onima iz treninga,
inače se model evaluira na drugačije pripremljenim ulazima. Točne vrijednosti stoje u
`args.json` tog pokretanja. Najveći `N` u mapi je ujedno i najbolji checkpoint
(odjeljak 6).

---

## 8. Izvedena pokretanja

Logovi treninga priloženi su u korijenu repozitorija. Brojke iz njih nisu ovdje navedene
— nalaze se u radu; ovaj odjeljak opisuje **što je pokrenuto i kako**, da se postavka može
rekonstruirati.

| Log                     | Predtreniranje         | Napomena                         |
|-------------------------|------------------------|----------------------------------|
| `log_orig_imagenet.txt` | ImageNet               | kontrolno pokretanje             |
| `log_orig_rgbd.txt`     | NYUv2 + SceneNet RGB-D | `--finetune`                     |
| `log_orig.txt`          | ImageNet               | starija, veća podjela podataka   |
| `log_s1_rgbd.txt`       | NYUv2 + SceneNet RGB-D | **pokretanje prekinuto greškom** |

**Prva dva loga su parna usporedba.** Ista podjela podataka, isti pipeline, ista
razlučivost, isti hiperparametri, isti broj epoha — razlika je **samo u** `--finetune`.
To je cijela svrha ovog repozitorija (vidi uvod): izolirati izvor predtreniranja od
arhitekture.

`log_orig.txt` **nije dio te usporedbe.** Radi se na starijoj i većoj podjeli podataka, s
drugim brojem validacijskih slika, pa se ne smije staviti u istu tablicu s prva dva.

`log_s1_rgbd.txt` **je pokretanje koje je palo**, na drugoj podjeli podataka. Prekida se s
`FileNotFoundError` na datoteci
`.../branchtrening12/RGB/E_tree_1_V_0174_tree_1_V_0174_14_pred.png` — ime u popisu splita
ne odgovara datoteci na disku, dio imena je udvostručen. Treba popraviti popis ili imena
datoteka pa ponoviti.

### Što za svako pokretanje ostaje zapisano

Uz log, svako pokretanje ostavlja `args.json` i `argsv.txt` u svojoj mapi (odjeljak 6),
pa se točna naredba i svi argumenti mogu pročitati bez pamćenja. Pri svakom pokretanju s
`--finetune` log sadrži i ispis `preuzeto N/898 tenzora` — provjera da je prijenos težina
prošao kako treba (odjeljak 1.2).

Kod ne fiksira sjeme, pa pokretanja nisu bit-po-bit ponovljiva; svaki je uvjet pokrenut
jednom.

---

## 9. Struktura repozitorija

Zvjezdicom je označeno ono što nije iz izvornog repozitorija.

```
.
├── train.py                              trening (ulazna točka)
├── eval.py                             * + --split, IoU po klasi, mIoU bez pozadine
├── ugradi_branch.py                    * registracija BRANCH skupa u njihov kod
│
├── src/
│   ├── args.py                         * 'branch' među dopuštenim skupovima
│   ├── build_model.py                  * filtrirani --finetune + uvjetni freeze
│   ├── prepare_data.py                 * grana za --dataset branch
│   ├── preprocessing.py                  augmentacije i normalizacija
│   ├── metric.py                         mIoU iz globalne matrice zabune
│   ├── utils.py                          gubitak, snimanje checkpointa
│   ├── logger.py                         CSV logger
│   ├── models/
│   │   ├── model.py                      ESANet
│   │   ├── model_one_modality.py         varijanta samo RGB ili samo dubina
│   │   ├── resnet.py                     enkoder, NonBottleneck1D blok
│   │   ├── rgb_depth_fusion.py           SE-add fuzija
│   │   ├── context_modules.py            PPM / APPM
│   │   └── external_code/                zakrpe za tuđe modele iz njihovog rada
│   └── datasets/
│       ├── dataset_base.py               bazna klasa, težine klasa, statistika dubine
│       ├── branch/                     * NAŠ SKUP
│       │   ├── branch.py               *   imena klasa, boje, mape, pomak +1
│       │   ├── pytorch_dataset.py      *   učitavanje, ravna struktura, statistika
│       │   ├── depth_raw_mean_std.pickle           *
│       │   ├── weighting_median_frequency_1+5_train.pickle *
│       │   └── weighting_linear_1+5_valid.pickle   *
│       ├── nyuv2/, sunrgbd/, cityscapes/, scenenetrgbd/   iz izvornog repozitorija
│       └── *.prije_branch              * originali zakrpanih datoteka
│
├── trained_models/
│   ├── imagenet/r34_NBt1D.pth          * ImageNet težine enkodera
│   └── nyuv2/r34_NBt1D_scenenet.pth    * RGB-D predtrenirani ESANet
│
├── log_orig_imagenet.txt               * log ImageNet treninga
├── log_orig_rgbd.txt                   * log RGB-D treninga
├── log_orig.txt                        * log na starijoj podjeli podataka
├── log_s1_rgbd.txt                     * log pokretanja koje je palo
│
├── inference_samples.py                  demo na primjerima — TREBA mapu samples/
├── inference_dataset.py                  inferencija nad skupom
├── inference_time*.py / .sh              mjerenje brzine
├── model_to_onnx.py                      izvoz u ONNX
├── imagenet_pretraining.py               predtreniranje enkodera na ImageNetu
│
├── rgbd_segmentation.yaml                izvorno okruženje autora
├── env_esanet2025*.yaml                  novije okruženje iz njihovog repozitorija
├── requirements_jetson.txt
├── README.md
└── LICENSE                               iz izvornog repozitorija
```

---

## 10. Na što paziti

- **Pickle datoteke s težinama klasa vrijede za konkretnu podjelu podataka.** Promijeni li se podjela, treba ih obrisati iz `src/datasets/branch/` da se preračunaju. Ime ne sadrži otisak podjele, pa se neslaganje ne otkriva automatski. Isto za `depth_raw_mean_std.pickle`.
- `--raw_depth` **nije zadano.** Bez njega se koristi `refined` način, a dubina je ovdje rijetka (~78 % nula), pa model uči predviđati vrijednosti koje nisu izmjerene. Sva priložena pokretanja su s `--raw_depth`.
- **Ispis** `preuzeto N/898 tenzora` **treba provjeriti pri svakom** `--finetune`**.** Ako N znatno padne, prijenos težina nije onakav kakav se misli, a trening će se ipak izvršiti do kraja i dati brojku.
- **Ako se u ispisu pojavi** `Freeze the encoder(s).` **bez zadanog** `--freeze`, verzija `build_model.py` nije zakrpana i trenira se samo dekoder.
- **mIoU iz treninga uključuje pozadinu**, mIoU za rad je onaj bez pozadine iz `eval.py`. Odjeljak 7.
- **Docstring u** `eval.py` **navodi konkretan broj validacijskih i testnih slika** — te se brojke odnose na stariju podjelu podataka i ne odgovaraju trenutnoj.
- **Reproducibilnost.** Kod ne fiksira sjeme, pa pokretanja nisu ponovljiva. Točna naredba svakog pokretanja ostaje u `argsv.txt`, ali sjeme ne.
- `inference_samples.py` **neće raditi** — traži mapu `samples/` iz izvornog repozitorija, koja ovdje nije uključena.
- **Mapa** `datasets/` iz izvornog repozitorija (prazna, za simboličke veze na skupove) također nije uključena; put do skupa se zadaje preko `--dataset_dir`.

---

## 11. Licenca i citiranje

Kod je izveden iz službene implementacije ESANeta; licenca izvornog repozitorija je u
`LICENSE` i primjenjuje se na sve preuzete datoteke.

```bibtex
@inproceedings{esanet2021icra,
  title={Efficient RGB-D Semantic Segmentation for Indoor Scene Analysis},
  author={Seichter, Daniel and K{\"o}hler, Mona and Lewandowski, Benjamin
          and Wengefeld, Tim and Gross, Horst-Michael},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  pages={13525--13531},
  year={2021}
}
```