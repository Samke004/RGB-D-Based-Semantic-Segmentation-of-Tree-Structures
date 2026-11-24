#!/usr/bin/env bash
#
# kampanja.sh — cijela serija pokretanja, jedno za drugim.
#
#   chmod +x kampanja.sh
#   nohup ./kampanja.sh > kampanja.txt 2>&1 &
#   tail -f kampanja.txt
#
# Svaki model se pokrece kao vlastiti proces. To je namjerno:
#  - ako jedan model padne (OOM, greska u tezinama), ostali nastave
#  - main.py vraca sjeme prije svakog modela, pa je rezultat isti kao
#    da je sve islo u jednom pozivu s --model all
#  - VRAM se potpuno oslobodi izmedu modela
#
# Logovi: log/<sufiks>__<model>.txt
# Ako skriptu prekines i pokrenes ponovno, modeli koji su vec gotovi
# se preskacu (provjerava se postoji li <model>_best.pth).

set -u

SVI_MODELI=(
  unet_resnet34
  unet_efficientnet_b4
  deeplabv3plus_resnet50
  deeplabv3plus_mobilenet
  segformer_b1
  segnet_vgg16bn
  esanet_resnet34
)

# za drugo sjeme samo podskup — dovoljno za procjenu varijance sjemena
PODSKUP=(unet_resnet34 segformer_b1 esanet_resnet34)

mkdir -p log

pokreni_krug () {
  local data="$1" sufiks="$2" seed="$3"; shift 3
  local modeli=("$@")

  echo ""
  echo "============================================================"
  echo "  KRUG: $sufiks   (podaci $data, sjeme $seed)"
  echo "  modeli: ${modeli[*]}"
  echo "  pocetak: $(date '+%F %T')"
  echo "============================================================"

  for m in "${modeli[@]}"; do
    local ckpt="checkpoints_${sufiks}/${m}_best.pth"
    if [[ -f "$ckpt" ]]; then
      echo "[$(date '+%T')] preskacem $m — $ckpt vec postoji"
      continue
    fi

    echo "[$(date '+%T')] >>> $sufiks / $m"
    local t0=$SECONDS
    python main.py \
        --model  "$m" \
        --data   "$data" \
        --seed   "$seed" \
        --sufiks "$sufiks" \
        > "log/${sufiks}__${m}.txt" 2>&1
    local status=$?
    local min=$(( (SECONDS - t0) / 60 ))

    if [[ $status -eq 0 ]]; then
      echo "[$(date '+%T')] <<< $m gotov za ${min} min"
      grep -oE "mIoU=[0-9.]+" "log/${sufiks}__${m}.txt" | tail -1
    else
      echo "[$(date '+%T')] !!! $m PAO (izlazni kod $status) nakon ${min} min"
      echo "    zadnjih 15 redaka log/${sufiks}__${m}.txt:"
      tail -15 "log/${sufiks}__${m}.txt" | sed 's/^/    /'
    fi
  done
}

# ── prvo sjeme, oba splita: puna mreza 2 x 7 ───────────────────────────────
pokreni_krug data        split1_seed42   42   "${SVI_MODELI[@]}"
pokreni_krug data_split2 split2_seed42   42   "${SVI_MODELI[@]}"

# ── drugo sjeme, oba splita: podskup ───────────────────────────────────────
pokreni_krug data        split1_seed1337 1337 "${PODSKUP[@]}"
pokreni_krug data_split2 split2_seed1337 1337 "${PODSKUP[@]}"

echo ""
echo "============================================================"
echo "  SVE GOTOVO: $(date '+%F %T')"
echo "============================================================"
ls -d results_split*_seed* 2>/dev/null