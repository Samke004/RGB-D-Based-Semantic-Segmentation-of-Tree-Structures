#!/bin/bash
# =============================================================================
# train_all.sh — trenira svih 6 DFormer modela na BranchDatasetu
#
# Za svaki model radi mapu  checkpoints/<ime_modela>/  u koju sprema:
#     checkpoint/           — .pth datoteke (kako ih DFormer zapiše)
#     tb/                   — TensorBoard zapisi
#     log/                  — svi logovi iz treninga
#     konfiguracija/        — KOPIJE svih .py configa korištenih u treningu
#     INFO.txt              — sve o ovom treningu (vrijeme, GPU, verzije, ...)
#     metrike.csv           — mIoU po epohama, izvučen iz loga
#     checkpoints.txt       — popis checkpointa s njihovim mIoU
#     dataset.txt           — stanje dataseta u trenutku treninga
#
# Na kraju piše i  checkpoints/SAZETAK.csv  — jedan redak po modelu.
#
# UPORABA
#   tmux new -s trening            # OBAVEZNO ako radiš preko AnyDeska/SSH-a
#   bash train_all.sh
#
#   bash train_all.sh --proba            # simulacija, ne trenira ništa
#   bash train_all.sh --modeli "DFormer_Base DFormerv2_L"
#   bash train_all.sh --dataset /put/do/BranchDataset
# =============================================================================

cd "$(dirname "$0")" || exit 1
BASE="$(pwd)"
export PYTHONPATH="$BASE:$BASE/..:$PYTHONPATH"

GPUS=2
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NNODES=1
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PORT=29200

CKPT="$BASE/checkpoints"
DATASET="$BASE/datasets/BranchDataset"
PROBA=0

MODELI=(
    "DFormer_Small"
    "DFormer_Base"
    "DFormer_Large"
    "DFormerv2_S"
    "DFormerv2_B"
    "DFormerv2_L"
)

# ── Argumenti ────────────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        --proba)    PROBA=1; shift ;;
        --modeli)   read -r -a MODELI <<< "$2"; shift 2 ;;
        --dataset)  DATASET="$2"; shift 2 ;;
        --izlaz)    CKPT="$2"; shift 2 ;;
        -h|--help)  sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "Nepoznat argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$CKPT"
SAZETAK="$CKPT/SAZETAK.csv"
[ -f "$SAZETAK" ] || echo "model,pocetak,trajanje_min,status,najbolji_miou,najbolji_checkpoint,mapa" > "$SAZETAK"

# ── Stanje dataseta (računa se jednom, kopira se u svaku mapu) ───────────────
DATASET_INFO="$(mktemp)"
{
    echo "DATASET: $DATASET"
    echo "snimljeno: $(date '+%Y-%m-%d %H:%M:%S')"
    echo
    for s in train val test; do
        if [ -f "$DATASET/$s.txt" ]; then
            echo "$s.txt: $(wc -l < "$DATASET/$s.txt") redaka"
        fi
    done
    for d in RGB Depth Label; do
        [ -d "$DATASET/$d" ] && echo "$d/: $(ls -1 "$DATASET/$d" 2>/dev/null | wc -l) datoteka"
    done
    echo
    echo "--- raspodjela klasa po pikselima (Label/) ---"
    python3 - "$DATASET" <<'PYEOF' 2>/dev/null || echo "(ne mogu izračunati — nedostaje numpy/PIL)"
import sys, os, glob
import numpy as np
from PIL import Image
NAZIV = {0: "pozadina", 1: "deblo", 2: "grane", 3: "potpora", 4: "trava", 255: "ignore"}
mape = sorted(glob.glob(os.path.join(sys.argv[1], "Label", "*.png")))
if not mape:
    print("(nema Label/*.png)"); raise SystemExit
zbroj = {}
for f in mape:
    u, c = np.unique(np.array(Image.open(f)), return_counts=True)
    for k, n in zip(u.tolist(), c.tolist()):
        zbroj[k] = zbroj.get(k, 0) + n
uk = sum(zbroj.values())
for k in sorted(zbroj):
    print(f"  {NAZIV.get(k, '?'):<10} (labela {k:>3}) {zbroj[k]:>14,}  ({100*zbroj[k]/uk:6.2f}%)")
stvarne = [k for k in zbroj if k != 255]
print(f"\n  num_classes mora biti {max(stvarne)+1}")
PYEOF
} > "$DATASET_INFO" 2>&1

echo "══════════════════════════════════════════════════════════════════════"
echo "  Modeli:  ${MODELI[*]}"
echo "  Dataset: $DATASET"
echo "  Izlaz:   $CKPT"
[ "$PROBA" = "1" ] && echo "  NAČIN:   PROBA — ne trenira se ništa"
echo "══════════════════════════════════════════════════════════════════════"
sed -n '/raspodjela/,$p' "$DATASET_INFO"
echo

# ── Petlja po modelima ───────────────────────────────────────────────────────
for MODEL in "${MODELI[@]}"; do
    PORT=$((PORT + 1))
    CFG="local_configs.BranchDataset.$MODEL"
    CFG_FILE="$BASE/local_configs/BranchDataset/$MODEL.py"
    ODREDISTE="$CKPT/$MODEL"
    LOG="$CKPT/${MODEL}_trening.log"

    echo ""
    echo "──────────────────────────────────────────────────────────────────────"
    echo "  $MODEL   ($(date '+%H:%M:%S'))"
    echo "──────────────────────────────────────────────────────────────────────"

    if [ ! -f "$CFG_FILE" ]; then
        echo "  [PRESKAČEM] nema configa: $CFG_FILE"
        echo "$MODEL,$(date '+%Y-%m-%d %H:%M'),0,nema_configa,,,"  >> "$SAZETAK"
        continue
    fi

    # Popis mapa PRIJE treninga, da poslije znamo koja je nova
    PRIJE="$(mktemp)"
    ls -1 "$CKPT" > "$PRIJE" 2>/dev/null

    POCETAK_S=$(date +%s)
    POCETAK="$(date '+%Y-%m-%d %H:%M:%S')"

    NAREDBA="torchrun --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR \
--nproc_per_node=$GPUS --master_port=$PORT utils/train.py --config=$CFG --gpus=$GPUS \
--no-sliding --no-compile --no-amp --val_amp --no-use_seed"

    if [ "$PROBA" = "1" ]; then
        # Simulacija: napravi mapu kakvu bi DFormer napravio, s par lažnih checkpointa
        LAZNA="$CKPT/BranchDataset_${MODEL}_$(date '+%Y%m%d-%H%M%S')"
        mkdir -p "$LAZNA/checkpoint" "$LAZNA/tb"
        for e in 10 20 30; do
            : > "$LAZNA/checkpoint/epoch-${e}_miou_8${e}.00.pth"
        done
        {
            echo "PROBA — ovo nije pravi trening"
            echo "Epoch 10 miou 80.00"
            echo "Epoch 20 miou 82.00"
            echo "Epoch 30 miou 83.00"
        } > "$LAZNA/log_proba.log"
        cp "$LAZNA/log_proba.log" "$LOG"
        STATUS="proba"
    else
        eval "$NAREDBA" 2>&1 | tee "$LOG"
        if [ "${PIPESTATUS[0]}" -eq 0 ]; then STATUS="ok"; else STATUS="pao"; fi
    fi

    KRAJ_S=$(date +%s)
    TRAJANJE=$(( (KRAJ_S - POCETAK_S) / 60 ))

    # Koja je mapa nastala?
    NOVA="$(ls -1 "$CKPT" | grep -vxF -f "$PRIJE" | grep -v '^SAZETAK' | head -n1)"
    rm -f "$PRIJE"

    if [ -z "$NOVA" ]; then
        echo "  [UPOZORENJE] trening nije napravio novu mapu u checkpoints/"
        echo "$MODEL,$POCETAK,$TRAJANJE,${STATUS}_bez_mape,,,"  >> "$SAZETAK"
        continue
    fi

    # Staru istoimenu mapu makni u stranu umjesto da je pregazimo
    if [ -d "$ODREDISTE" ]; then
        STARA="${ODREDISTE}_staro_$(date '+%Y%m%d-%H%M%S')"
        echo "  $ODREDISTE već postoji → premještam u $(basename "$STARA")"
        mv "$ODREDISTE" "$STARA"
    fi
    mv "$CKPT/$NOVA" "$ODREDISTE"

    # ── Slaganje mape ────────────────────────────────────────────────────────
    mkdir -p "$ODREDISTE/log" "$ODREDISTE/konfiguracija"
    mv "$LOG" "$ODREDISTE/log/" 2>/dev/null
    find "$ODREDISTE" -maxdepth 1 -name "*.log" -exec mv {} "$ODREDISTE/log/" \; 2>/dev/null
    cp "$DATASET_INFO" "$ODREDISTE/dataset.txt"

    # Kompletna konfiguracija — izvorni .py, točno kakvi su bili pri treningu
    cp "$CFG_FILE" "$ODREDISTE/konfiguracija/" 2>/dev/null
    for f in "$BASE/local_configs/_base_/datasets/BranchDataset.py" \
             "$BASE/local_configs/_base_/__init__.py" \
             "$BASE/local_configs/__init__.py"; do
        [ -f "$f" ] && cp "$f" "$ODREDISTE/konfiguracija/$(basename "$(dirname "$f")")_$(basename "$f")"
    done

    # Popis checkpointa s mIoU, najbolji na vrhu
    : > "$ODREDISTE/checkpoints.txt"
    NAJBOLJI=""; NAJMIOU=""
    if ls "$ODREDISTE"/checkpoint/*miou_*.pth >/dev/null 2>&1; then
        ls -1 "$ODREDISTE"/checkpoint/*miou_*.pth \
            | sed 's/.*miou_\([0-9.]*\)\.pth/\1 &/' \
            | sort -k1 -nr > "$ODREDISTE/checkpoints.txt"
        NAJMIOU="$(head -n1 "$ODREDISTE/checkpoints.txt" | awk '{print $1}')"
        NAJBOLJI="$(basename "$(head -n1 "$ODREDISTE/checkpoints.txt" | awk '{print $2}')")"
    fi

    # mIoU po epohama iz loga
    {
        echo "izvor,redak"
        grep -hiE "miou|mean_iou|epoch" "$ODREDISTE"/log/*.log 2>/dev/null \
            | sed 's/,/;/g' | awk '{print "log,"$0}'
    } > "$ODREDISTE/metrike.csv"

    # ── INFO.txt ─────────────────────────────────────────────────────────────
    {
        echo "═══════════════════════════════════════════════════════════════"
        echo " MODEL: $MODEL"
        echo "═══════════════════════════════════════════════════════════════"
        echo
        echo "Početak:   $POCETAK"
        echo "Kraj:      $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Trajanje:  $TRAJANJE min"
        echo "Status:    $STATUS"
        echo "Najbolji:  ${NAJBOLJI:-nema}  (mIoU ${NAJMIOU:-?})"
        echo "Izvorna mapa koju je DFormer napravio: $NOVA"
        echo
        echo "--- NAREDBA ---"
        echo "$NAREDBA"
        echo
        echo "--- OKRUŽENJE ---"
        echo "host:        $(hostname)"
        echo "korisnik:    $(whoami)"
        echo "radna mapa:  $BASE"
        echo "conda env:   ${CONDA_DEFAULT_ENV:-nepoznato}"
        echo "python:      $(python3 --version 2>&1)"
        echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        python3 -c "import torch;print(f'torch:       {torch.__version__}  cuda {torch.version.cuda}  dostupno {torch.cuda.is_available()}')" 2>/dev/null
        python3 -c "import torch;print(f'GPU:         {torch.cuda.get_device_name(0)}')" 2>/dev/null
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
            | sed 's/^/nvidia-smi:  /'
        if [ -d "$BASE/.git" ]; then
            echo "git commit:  $(git -C "$BASE" rev-parse --short HEAD 2>/dev/null)"
            echo "git grana:   $(git -C "$BASE" rev-parse --abbrev-ref HEAD 2>/dev/null)"
            IZMJENE=$(git -C "$BASE" status --porcelain 2>/dev/null | wc -l)
            echo "nespremljenih izmjena u repou: $IZMJENE"
        fi
        echo
        echo "--- DATASET ---"
        cat "$DATASET_INFO"
        echo
        echo "--- KONFIGURACIJA (${MODEL}.py) ---"
        cat "$CFG_FILE"
        echo
        echo "--- KONFIGURACIJA (_base_/datasets/BranchDataset.py) ---"
        cat "$BASE/local_configs/_base_/datasets/BranchDataset.py" 2>/dev/null
        echo
        echo "--- SVI CHECKPOINTI (mIoU, putanja) ---"
        cat "$ODREDISTE/checkpoints.txt"
        echo
        echo "--- ZADNJIH 40 REDAKA LOGA ---"
        tail -n 40 "$ODREDISTE"/log/*.log 2>/dev/null
        echo
        echo "--- PAKETI (pip freeze) ---"
        pip freeze 2>/dev/null
    } > "$ODREDISTE/INFO.txt"

    echo "$MODEL,$POCETAK,$TRAJANJE,$STATUS,${NAJMIOU:-},${NAJBOLJI:-},$ODREDISTE" >> "$SAZETAK"

    echo "  gotovo: $STATUS   $TRAJANJE min   najbolji mIoU ${NAJMIOU:-?}"
    echo "  spremljeno u: $ODREDISTE"
done

rm -f "$DATASET_INFO"

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "  SVE GOTOVO"
echo "══════════════════════════════════════════════════════════════════════"
column -s, -t < "$SAZETAK" 2>/dev/null || cat "$SAZETAK"
echo ""
echo "Sažetak: $SAZETAK"