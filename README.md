# DATN — Ho tro chan doan thoai hoa khop goi (X-quang)

Pipeline: tien xu ly X-quang goi (grayscale, crop ROI theo nhan YOLO), huan luyen phan loai muc Kellgren-Lawrence (KL0–KL4), danh gia va xuat bao cao (confusion matrix, CSV du doan, JSON).

**Luu y:** Day la cong cu ho tro, khong thay the bac si; quyet dinh lam sang thuoc bac si.

## Yeu cau

- Python 3.10+ (khuyen nghi 3.11)
- GPU (tuy chon, tang toc train)

## Cai dat

```bash
cd DATN_PTH_V2
pip install -r requirements.txt
```

## Cau truc du lieu

- `data/Images_E/` — anh goc (vi du `.jpg`)
- `data/Labels_E/` — nhan YOLO: moi file `.txt` cung ten stem voi anh, moi dong `class cx cy w h` (chuan hoa 0–1). Class `0…4` tuong ung KL0…KL4; **nhan anh** khi train lay **max** class trong file (heuristic).

## Cau truc ma (`src/`)

Thu tu so trong ten file goi y luong xu ly:

1. `1_run_logging.py` — logging ra file
2. `2_preprocess.py` — doc anh, grayscale, crop, preview
3. `3_labels.py` — suy ra KL tu file nhan
4. `4_manifest.py` — gom mau, chia **train / val / test** (mac dinh **7-2-1**)
5. `5_dataset.py` — PyTorch `Dataset`
6. `6_model.py` — ResNet18 mot ken + pretrained
7. `7_train.py` — huan luyen, checkpoint tot nhat theo **val**
8. `8_eval_run.py` — danh gia lai checkpoint
9. `9_eval_report.py` — xuat bao cao (PNG/CSV/JSON)

`main.py` o thu muc goc la diem vao CLI.

## Chay nhanh

### 1. Tien xu ly (batch)

```bash
python main.py preprocess --images data/Images_E --labels data/Labels_E --out output/preprocessed
```

Tao `output/preprocessed/gray_crop/` (PNG 1 ken), `preview/`, `logs/preprocess.log`.

Tuy chon: `--limit N` (chi N anh dau, thu nghiem), `--pattern "*.jpg"`.

### 2. Huan luyen

```bash
python main.py train --gray-dir output/preprocessed/gray_crop --labels-dir data/Labels_E --out output/run_latest
```

Mac dinh: `--epochs 30`, `--batch-size 16`, chia **70% train / 20% val / 10% test**, checkpoint tot nhat theo accuracy tren **val**.

Tuy chinh ti le:

```bash
python main.py train ... --train-frac 0.7 --val-frac 0.2 --test-frac 0.1
```

**Artifacts chinh:**

- `output/run_latest/best_model.pt` — trong so mo hinh
- `output/run_latest/manifest.json` — danh sach train/val/test
- `output/run_latest/history.csv` — loss/accuracy theo epoch
- `output/run_latest/train_config.json` — sieu tham so
- `output/run_latest/reports/validation/` — bao cao tren tap validation
- `output/run_latest/reports/test/` — bao cao tren tap test (giu lai, khong chon checkpoint)

### Danh gia lai (tuy chon)

```bash
python main.py evaluate --manifest output/run_latest/manifest.json --checkpoint output/run_latest/best_model.pt --out output/eval_out --split test
```

`--split val` hoac `test` (mac dinh `test`).

### Mot anh + mot nhan

```bash
python main.py preprocess-one --image path/to.jpg --label path/to.txt --out output/preprocessed
```

## Bao cao trong `reports/`

Moi thu muc con (`validation/`, `test/`) gom:

- `classification_report.txt`
- `predictions.csv` — `stem`, `y_true`, `y_pred`, xac suat `prob_KL0`…`prob_KL4`
- `confusion_matrix.csv`, `confusion_matrix.png`
- `eval_summary.json`

## Ghi chu

- Tap **test** khong dung de chon `best_model.pt`; chi danh gia sau cung.
- Dataset nho hoac lop hiem: co the log canh bao va chia ngau nhien thay vi stratify.
- Import module co tien to so (`2_preprocess`, …): dung `importlib.import_module("src.2_preprocess")` (xem `main.py`).

## Du lieu y te

Chi dung du lieu dung quy dinh va da duoc phep.
