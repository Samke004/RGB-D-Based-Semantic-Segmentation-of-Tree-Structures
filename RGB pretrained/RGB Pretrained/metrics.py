"""
metrics.py — Multi-class segmentation metrics
────────────────────────────────────────────────────────────────────────────
Klase se citaju iz configa (CLASS_NAMES), trenutno:
  0=background, 1=trunk, 2=branches, 3=support, 4=grass

Sve funkcije primaju:
  pred : [H, W] int array  — argmax izlaza modela, vrijednosti 0..C-1
  gt   : [H, W] int array  — tocne oznake,          vrijednosti 0..C-1


VAZNO — sto ove metrike jesu, a sto nisu
────────────────────────────────────────────────────────────────────────────
Ovdje se sve racuna ZA JEDNU SLIKU. aggregate_metrics zatim uzima prosjek
preko slika, pa svaka slika ima jednaku tezinu bez obzira na broj piksela.

To NIJE isto kao globalni racun (jedna matrica zabune preko cijelog skupa),
koji koriste DFormer, mmsegmentation i vecina radova. Globalni racun je u
trainer.py (_globalni_iou) i evaluate.py (_globalne_metrike), i daje
stupce glob_*. U rad idu glob_* brojke.

Kad klase na slici nema ni u istini ni u predikciji, metrika za tu klasu
vraca nan, a ne nulu. aggregate_metrics nan izbacuje, pa ta slika ne ulazi
u prosjek za tu klasu. Prije se vracala nula, cime su rijetke klase
(potpora, trava) bile sustavno podcijenjene — slika bez trave upisivala je
travi IoU 0.
"""

from typing import Dict, List, Optional
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

from config import Config as _cfg

NUM_CLASSES = _cfg.NUM_CLASSES
CLASS_NAMES = list(_cfg.CLASS_NAMES)

# Indeks klase "grane" — koristi se za branch_recall i indekse tezine.
# Izvodi se iz imena da ne ostane tvrdo zapisan ako se shema promijeni.
try:
    BRANCH_IDX = CLASS_NAMES.index("branches")
except ValueError:
    BRANCH_IDX = 2


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _boundary(mask: np.ndarray) -> np.ndarray:
    eroded = binary_erosion(mask.astype(bool), iterations=1)
    return mask.astype(bool) & ~eroded


def _sigurno_dijeli(brojnik, nazivnik) -> float:
    """Dijeljenje koje vraca nan umjesto nule kad nazivnik ne postoji.

    Prije je svugdje stajalo `/ (nazivnik + 1e-10)`, sto je za nazivnik 0
    davalo 0.0 — a 0.0 je legitimna vrijednost metrike, pa se nije moglo
    razlikovati "model je sve promasio" od "klase nema na slici".
    """
    return float(brojnik / nazivnik) if nazivnik > 0 else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Overall Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def overall_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(pred == gt))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-class IoU + Mean IoU
# ─────────────────────────────────────────────────────────────────────────────

def per_class_iou(pred: np.ndarray, gt: np.ndarray,
                  num_classes: int = NUM_CLASSES) -> List[float]:
    """IoU po klasi. nan kad klase nema ni u istini ni u predikciji."""
    ious = []
    for c in range(num_classes):
        p = pred == c
        g = gt == c
        presjek = np.logical_and(p, g).sum()
        unija = np.logical_or(p, g).sum()
        ious.append(_sigurno_dijeli(presjek, unija))
    return ious


def mean_iou(pred: np.ndarray, gt: np.ndarray, ignore_bg: bool = True) -> float:
    """Prosjek IoU preko klasa; klase kojih na slici nema se preskacu.

    nanmean, ne mean — inace bi jedan nan ucinio cijelu sliku nan-om,
    pa bi aggregate_metrics izbacio tu sliku i prosjek bi se racunao samo
    preko slika koje sadrze SVE klase.
    """
    ious = per_class_iou(pred, gt)
    start = 1 if ignore_bg else 0
    valjane = ious[start:]
    if all(np.isnan(v) for v in valjane):
        return float("nan")
    return float(np.nanmean(valjane))


def frequency_weighted_iou(pred: np.ndarray, gt: np.ndarray,
                           num_classes: int = NUM_CLASSES) -> float:
    """IoU tezinski po ucestalosti klase. Odsutne klase se preskacu."""
    ious = per_class_iou(pred, gt, num_classes)
    total = gt.size
    fw, tezina = 0.0, 0.0
    for c in range(num_classes):
        freq = float((gt == c).sum()) / total
        if freq > 0 and not np.isnan(ious[c]):
            fw += freq * ious[c]
            tezina += freq
    return _sigurno_dijeli(fw, tezina) if tezina > 0 else float("nan")


def pixel_accuracy_fg(pred: np.ndarray, gt: np.ndarray) -> float:
    """Tocnost po pikselima samo na prednjim klasama (bez pozadine)."""
    fg = gt != 0
    if fg.sum() == 0:
        return float("nan")
    return float(np.mean(pred[fg] == gt[fg]))


def frequency_weighted_iou_fg(pred: np.ndarray, gt: np.ndarray,
                              num_classes: int = NUM_CLASSES) -> float:
    """fwIoU samo za prednje klase (1..C-1), frekvencije renormalizirane."""
    ious = per_class_iou(pred, gt, num_classes)
    total_fg = (gt != 0).sum()
    if total_fg == 0:
        return float("nan")
    fw, tezina = 0.0, 0.0
    for c in range(1, num_classes):
        freq = float((gt == c).sum()) / total_fg
        if freq > 0 and not np.isnan(ious[c]):
            fw += freq * ious[c]
            tezina += freq
    return _sigurno_dijeli(fw, tezina) if tezina > 0 else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Per-class Precision, Recall, F1
# ─────────────────────────────────────────────────────────────────────────────

def per_class_precision_recall_f1(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> Dict[str, List[float]]:
    """Preciznost, odziv i F1 po klasi.

    precision = nan kad model tu klasu nije predvidio nigdje
    recall    = nan kad klase nema u istini
    """
    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = np.logical_and(pred == c, gt == c).sum()
        fp = np.logical_and(pred == c, gt != c).sum()
        fn = np.logical_and(pred != c, gt == c).sum()

        p = _sigurno_dijeli(tp, tp + fp)
        r = _sigurno_dijeli(tp, tp + fn)

        if np.isnan(p) or np.isnan(r) or (p + r) == 0:
            f = float("nan") if (np.isnan(p) or np.isnan(r)) else 0.0
        else:
            f = 2 * p * r / (p + r)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
    return {"precision": precisions, "recall": recalls, "f1": f1s}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Boundary F1 (per class)
# ─────────────────────────────────────────────────────────────────────────────

def boundary_f1_per_class(
    pred: np.ndarray,
    gt: np.ndarray,
    theta: int = 2,
    num_classes: int = NUM_CLASSES,
) -> List[float]:
    """Boundary F1 po klasi.

    Klase kojih nema ni u istini ni u predikciji vracaju nan.
    Prije je vracano 1.0 — savrsen rezultat za klasu koje nema — sto je
    dizalo mean_boundary_f1 na slikama bez trave i potpore.
    """
    f1s = []
    struct = np.ones((2 * theta + 1, 2 * theta + 1), dtype=bool)
    for c in range(num_classes):
        pred_b = _boundary(pred == c)
        gt_b = _boundary(gt == c)

        if pred_b.sum() == 0 and gt_b.sum() == 0:
            f1s.append(float("nan"))      # klase nema — nema sto mjeriti
            continue
        if pred_b.sum() == 0 or gt_b.sum() == 0:
            f1s.append(0.0)               # jedna strana prazna — promasaj
            continue

        pred_dil = binary_dilation(pred_b, structure=struct)
        gt_dil = binary_dilation(gt_b, structure=struct)
        prec = _sigurno_dijeli(np.logical_and(pred_b, gt_dil).sum(), pred_b.sum())
        rec = _sigurno_dijeli(np.logical_and(gt_b, pred_dil).sum(), gt_b.sum())
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return f1s


def mean_boundary_f1(pred, gt, theta=2, ignore_bg=True) -> float:
    f1s = boundary_f1_per_class(pred, gt, theta)
    start = 1 if ignore_bg else 0
    valjane = f1s[start:]
    if all(np.isnan(v) for v in valjane):
        return float("nan")
    return float(np.nanmean(valjane))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Branch Recall + Occluded Branch Recall
# ─────────────────────────────────────────────────────────────────────────────

def branch_recall(pred: np.ndarray, gt: np.ndarray) -> float:
    """Odziv za klasu grana. nan ako grana nema na slici."""
    tp = np.logical_and(pred == BRANCH_IDX, gt == BRANCH_IDX).sum()
    fn = np.logical_and(pred != BRANCH_IDX, gt == BRANCH_IDX).sum()
    return _sigurno_dijeli(tp, tp + fn)


def occluded_branch_recall(
    pred: np.ndarray,
    gt: np.ndarray,
    occlusion: np.ndarray,
) -> float:
    """Odziv za piksele grana koji leze u zaklonjenim podrucjima."""
    occ_branch_gt = (gt == BRANCH_IDX) & occlusion.astype(bool)
    n = occ_branch_gt.sum()
    if n == 0:
        return float("nan")
    tp = np.logical_and(pred == BRANCH_IDX, occ_branch_gt).sum()
    return float(tp / n)


# ─────────────────────────────────────────────────────────────────────────────
# 6 & 7. Indeksi tezine
# ─────────────────────────────────────────────────────────────────────────────

def occlusion_difficulty_index(gt: np.ndarray, occlusion: np.ndarray) -> float:
    branch_pixels = (gt == BRANCH_IDX).sum()
    if branch_pixels == 0:
        return float("nan")
    return float(np.logical_and(gt == BRANCH_IDX,
                                occlusion.astype(bool)).sum() / branch_pixels)


def depth_difficulty_index(depth: np.ndarray, gt: np.ndarray) -> float:
    branch_mask = gt == BRANCH_IDX
    if branch_mask.sum() == 0:
        return float("nan")
    return float(np.std(depth[branch_mask]))


# ─────────────────────────────────────────────────────────────────────────────
# Glavna funkcija
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    occlusion: Optional[np.ndarray] = None,
    depth: Optional[np.ndarray] = None,
    boundary_theta: int = 2,
) -> Dict:
    """Sve metrike za JEDNU sliku. pred / gt: [H, W] s indeksima 0..C-1."""
    ious = per_class_iou(pred, gt)
    prf = per_class_precision_recall_f1(pred, gt)
    bf1s = boundary_f1_per_class(pred, gt, theta=boundary_theta)

    results = {
        "overall_accuracy": overall_accuracy(pred, gt),
        "pixel_accuracy": overall_accuracy(pred, gt),
        "pixel_accuracy_fg": pixel_accuracy_fg(pred, gt),
        "frequency_weighted_iou_with_bg": frequency_weighted_iou(pred, gt),
        "frequency_weighted_iou": frequency_weighted_iou_fg(pred, gt),
        "frequency_weighted_iou_fg": frequency_weighted_iou_fg(pred, gt),
        "mean_iou": mean_iou(pred, gt, ignore_bg=True),
        "mean_iou_with_bg": mean_iou(pred, gt, ignore_bg=False),
        "mean_boundary_f1": mean_boundary_f1(pred, gt, boundary_theta),
        "mean_boundary_f1_with_bg": mean_boundary_f1(pred, gt, boundary_theta,
                                                     ignore_bg=False),
        "branch_recall": branch_recall(pred, gt),
    }

    # prosjeci preciznosti / odziva / F1 po slici, obje varijante
    for naziv, niz in (("precision", prf["precision"]),
                       ("recall", prf["recall"]),
                       ("f1", prf["f1"])):
        bez = [v for v in niz[1:] if not np.isnan(v)]
        sve = [v for v in niz if not np.isnan(v)]
        results[f"mean_{naziv}"] = float(np.mean(bez)) if bez else float("nan")
        results[f"mean_{naziv}_with_bg"] = float(np.mean(sve)) if sve else float("nan")

    for i, name in enumerate(CLASS_NAMES):
        results[f"iou_{name}"] = ious[i]
        results[f"precision_{name}"] = prf["precision"][i]
        results[f"recall_{name}"] = prf["recall"][i]
        results[f"f1_{name}"] = prf["f1"][i]
        results[f"boundary_f1_{name}"] = bf1s[i]

    if occlusion is not None:
        results["occluded_branch_recall"] = occluded_branch_recall(pred, gt, occlusion)
        results["occlusion_difficulty_index"] = occlusion_difficulty_index(gt, occlusion)

    if depth is not None:
        results["depth_difficulty_index"] = depth_difficulty_index(depth, gt)

    return results



# ─────────────────────────────────────────────────────────────────────────────
# GLOBALNE metrike iz matrice zabune
# ─────────────────────────────────────────────────────────────────────────────
#
# Ovo je racun koji koriste DFormer, mmsegmentation, Cityscapes i vecina
# radova: pikseli svih slika idu u JEDNU matricu, a metrike se racunaju tek
# na kraju. Razlikuje se od svega iznad, gdje se racuna po slici pa uzima
# prosjek — ondje svaka slika ima jednaku tezinu bez obzira na broj piksela.
#
# U rad idu brojke odavde.

def prazna_matrica(num_classes: int = NUM_CLASSES) -> np.ndarray:
    return np.zeros((num_classes, num_classes), dtype=np.int64)


def dodaj_u_matricu(hist: np.ndarray, pred: np.ndarray, gt: np.ndarray,
                    num_classes: int = NUM_CLASSES) -> np.ndarray:
    """hist[i, j] += broj piksela istinite klase i predvidenih kao j."""
    g = np.asarray(gt).ravel()
    p = np.asarray(pred).ravel()
    vazeci = (g >= 0) & (g < num_classes)
    hist += np.bincount(g[vazeci] * num_classes + p[vazeci],
                        minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


def metrike_iz_matrice(hist: np.ndarray,
                       class_names: Optional[List[str]] = None,
                       prefiks: str = "glob") -> Dict[str, float]:
    """Sve globalne metrike iz jedne matrice zabune.

    Svaka prosjecna metrika dolazi u dvije varijante:
      <prefiks>_mean_X          — bez pozadine (klasa 0), kako racuna DFormer
      <prefiks>_mean_X_with_bg  — s pozadinom

    Pozadina je kod BRANCH skupa oko 76 % piksela i IoU joj je ~0.99, pa
    varijanta s pozadinom izlazi nekoliko bodova visa. Za usporedbu s
    DFormerom koristi se ona bez pozadine.
    """
    imena = list(class_names) if class_names is not None else list(CLASS_NAMES)
    h = np.asarray(hist, dtype=np.float64)
    k = h.shape[0]

    tp = np.diag(h)
    po_istini = h.sum(1)        # koliko piksela klasa stvarno ima
    po_predikciji = h.sum(0)    # koliko ih je model predvidio
    ukupno = h.sum()

    with np.errstate(divide="ignore", invalid="ignore"):
        unija = po_istini + po_predikciji - tp
        iou = np.where(unija > 0, tp / unija, np.nan)
        # F1 po klasi = 2TP / (predvideni + stvarni); matematicki isto sto i Dice
        f1 = np.where((po_predikciji + po_istini) > 0,
                      2 * tp / (po_predikciji + po_istini), np.nan)
        recall = np.where(po_istini > 0, tp / po_istini, np.nan)
        precision = np.where(po_predikciji > 0, tp / po_predikciji, np.nan)

    def _sred(v, s):
        dio = v[s:]
        return float(np.nanmean(dio)) if not np.all(np.isnan(dio)) else float("nan")

    out: Dict[str, float] = {}
    for i in range(k):
        ime = imena[i] if i < len(imena) else f"klasa_{i}"
        out[f"{prefiks}_iou_{ime}"] = float(iou[i])
        out[f"{prefiks}_f1_{ime}"] = float(f1[i])
        out[f"{prefiks}_recall_{ime}"] = float(recall[i])
        out[f"{prefiks}_precision_{ime}"] = float(precision[i])

    for naziv, niz in (("iou", iou), ("f1", f1),
                       ("recall", recall), ("precision", precision)):
        out[f"{prefiks}_mean_{naziv}"] = _sred(niz, 1)          # bez pozadine
        out[f"{prefiks}_mean_{naziv}_with_bg"] = _sred(niz, 0)  # s pozadinom

    # tocnost po pikselima
    out[f"{prefiks}_overall_acc"] = float(tp.sum() / ukupno) if ukupno > 0 else float("nan")
    fg_ukupno = po_istini[1:].sum()
    out[f"{prefiks}_pixel_acc_fg"] = (float(tp[1:].sum() / fg_ukupno)
                                      if fg_ukupno > 0 else float("nan"))

    # frekvencijski tezinski IoU
    with np.errstate(divide="ignore", invalid="ignore"):
        freq = po_istini / ukupno if ukupno > 0 else np.zeros(k)
    ok = (freq > 0) & ~np.isnan(iou)
    out[f"{prefiks}_fwiou_with_bg"] = float((freq[ok] * iou[ok]).sum()) if ok.any() else float("nan")

    if fg_ukupno > 0:
        freq_fg = po_istini[1:] / fg_ukupno
        ok_fg = (freq_fg > 0) & ~np.isnan(iou[1:])
        out[f"{prefiks}_fwiou"] = (float((freq_fg[ok_fg] * iou[1:][ok_fg]).sum())
                                   if ok_fg.any() else float("nan"))
    else:
        out[f"{prefiks}_fwiou"] = float("nan")

    return out


def imena_globalnih_stupaca(class_names: Optional[List[str]] = None,
                            prefiks: str = "glob") -> List[str]:
    """Popis kljuceva koje vraca metrike_iz_matrice — za zaglavlje CSV-a."""
    imena = list(class_names) if class_names is not None else list(CLASS_NAMES)
    st = []
    for naziv in ("iou", "f1", "recall", "precision"):
        st += [f"{prefiks}_{naziv}_{ime}" for ime in imena]
    for naziv in ("iou", "f1", "recall", "precision"):
        st += [f"{prefiks}_mean_{naziv}", f"{prefiks}_mean_{naziv}_with_bg"]
    st += [f"{prefiks}_overall_acc", f"{prefiks}_pixel_acc_fg",
           f"{prefiks}_fwiou", f"{prefiks}_fwiou_with_bg"]
    return st


def aggregate_metrics(metric_list: list) -> Dict:
    """Prosjek preko slika; nan vrijednosti se izbacuju.

    Posljedica koju treba imati na umu: stupac za rijetku klasu racuna se
    preko manje slika nego stupac za cestu klasu. Koliko tocno — vidi
    aggregate_counts.
    """
    if not metric_list:
        return {}
    kljucevi = set()
    for m in metric_list:
        kljucevi.update(m.keys())

    agg = {}
    for k in sorted(kljucevi):
        vals = []
        for m in metric_list:
            if k not in m:
                continue
            v = m[k]
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if not np.isnan(v):
                vals.append(v)
        agg[k] = float(np.mean(vals)) if vals else float("nan")
    return agg


def aggregate_counts(metric_list: list) -> Dict[str, int]:
    """Koliko je slika doprinijelo svakoj metrici (bez nan)."""
    if not metric_list:
        return {}
    kljucevi = set()
    for m in metric_list:
        kljucevi.update(m.keys())

    broj = {}
    for k in sorted(kljucevi):
        n = 0
        for m in metric_list:
            if k not in m:
                continue
            try:
                v = float(m[k])
            except (TypeError, ValueError):
                continue
            if not np.isnan(v):
                n += 1
        broj[k] = n
    return broj