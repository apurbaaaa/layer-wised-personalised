#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Fetch the ISIC 2019 *training* split (the only data the federated code uses)
# into a target directory, then verify the expected image count.
#
#   bash fetch_isic.sh [dest_dir]     # default: /root/data
#
# Designed to be run detached (nohup) — it is idempotent and writes a
# .fetch_done marker on success so a poller can tell when it has finished.
# ---------------------------------------------------------------------------
set -euo pipefail

DEST="${1:-/root/data}"
BASE="https://isic-archive.s3.amazonaws.com/challenges/2019"
EXPECTED=25331

mkdir -p "$DEST"
cd "$DEST"
rm -f .fetch_done .fetch_failed

echo "[fetch] dest=$DEST"

for f in ISIC_2019_Training_GroundTruth.csv ISIC_2019_Training_Metadata.csv; do
    if [ -s "$f" ]; then
        echo "[fetch] $f already present"
    else
        echo "[fetch] downloading $f"
        wget -q "$BASE/$f"
    fi
done

if [ -d ISIC_2019_Training_Input ] && \
   [ "$(ls ISIC_2019_Training_Input | wc -l)" -ge "$EXPECTED" ]; then
    echo "[fetch] images already present"
else
    if [ ! -s ISIC_2019_Training_Input.zip ]; then
        echo "[fetch] downloading images (~9 GB) ..."
        wget -q "$BASE/ISIC_2019_Training_Input.zip"
    fi
    echo "[fetch] unzipping ..."
    unzip -q -o ISIC_2019_Training_Input.zip
    rm -f ISIC_2019_Training_Input.zip
fi

COUNT=$(ls ISIC_2019_Training_Input | wc -l)
echo "[fetch] image count: $COUNT (expected $EXPECTED)"

if [ "$COUNT" -lt "$EXPECTED" ]; then
    echo "[fetch] FAILED: incomplete image set"
    touch .fetch_failed
    exit 1
fi

for f in ISIC_2019_Training_GroundTruth.csv ISIC_2019_Training_Metadata.csv; do
    [ -s "$f" ] || { echo "[fetch] FAILED: missing $f"; touch .fetch_failed; exit 1; }
done

touch .fetch_done
echo "[fetch] OK — complete"
