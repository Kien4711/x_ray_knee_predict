# DATN — Ho tro chan doan thoai hoa khop goi (X-quang)

Pipeline: tien xu ly X-quang goi (grayscale, crop ROI theo nhan YOLO), huan luyen phan loai muc Kellgren-Lawrence (KL0–KL4), danh gia va xuat bao cao (confusion matrix, CSV du doan, JSON). Co the chon backbone (ResNet50, EfficientNet-B0, MobileNetV2) va so sanh nhieu lan train.

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
6. `6_model.py` — backbone tuy chon: **resnet50**, **efficientnet_b0**, **mobilenet_v2**, conv dau 1 ken + pretrained ImageNet (RGB trung binh sang grayscale)
7. `7_train.py` — huan luyen, checkpoint tot nhat theo **val**
8. `8_eval_run.py` — danh gia lai checkpoint
9. `9_eval_report.py` — xuat bao cao (PNG/CSV/JSON)
10. `10_compare.py` — so sanh >=2 run da train (metrics + bieu do)

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

Mac dinh: `--model resnet50`, `--epochs 30`, `--batch-size 16`, chia **70% train / 20% val / 10% test**, checkpoint tot nhat theo accuracy tren **val**.

**Chon backbone** (`--model`):

| Gia tri | Mo ta |
|---------|--------|
| `resnet50` | ResNet-50 (mac dinh) |
| `efficientnet_b0` | EfficientNet-B0 |
| `mobilenet_v2` | MobileNetV2 |

Vi du:

```bash
python main.py train --gray-dir output/preprocessed/gray_crop --labels-dir data/Labels_E --out output/run_efficientnet --model efficientnet_b0
python main.py train --gray-dir output/preprocessed/gray_crop --labels-dir data/Labels_E --out output/run_mobilenet --model mobilenet_v2
```

Tuy chinh ti le:

```bash
python main.py train ... --train-frac 0.7 --val-frac 0.2 --test-frac 0.1
```

**Artifacts chinh:**

- `output/run_latest/best_model.pt` — trong so mo hinh (gom `model`, `image_size`, `num_classes`, …)
- `output/run_latest/manifest.json` — danh sach train/val/test
- `output/run_latest/history.csv` — loss/accuracy theo epoch
- `output/run_latest/train_config.json` — sieu tham so (gom `model`)
- `output/run_latest/reports/validation/` — bao cao tren tap validation
- `output/run_latest/reports/test/` — bao cao tren tap test (giu lai, khong chon checkpoint)

### Danh gia lai (tuy chon)

```bash
python main.py evaluate --manifest output/run_latest/manifest.json --checkpoint output/run_latest/best_model.pt --out output/eval_out --split test
```

`--split val` hoac `test` (mac dinh `test`). Neu checkpoint cu khong co truong `model`, mac dinh dung `resnet50`. Co the ghi de bang `--model mobilenet_v2` (phai khop kien truc voi trong so).

### So sanh nhieu mo hinh (`compare`)

Sau khi da train **it nhat hai** lan (hai thu muc output khac nhau), co the so sanh tren **cung mot tap** (test hoac val): cac `manifest.json` phai co **cung danh sach mau** (stem + KL) trong split duoc chon — thuc te: cung `gray_dir`, `labels_dir`, `seed`, va cac ti le train/val/test.

```bash
python main.py compare --runs output/run_resnet50 output/run_mobilenet --out output/compare_latest --split test
```

**Ket qua** (trong `--out`):

- `comparison_summary.json` — accuracy, F1 (macro/weighted), best val acc theo `history.csv`
- `comparison_metrics.csv` — bang tom tat
- `figures/metrics_bar.png` — cot accuracy / F1
- `figures/val_acc_curves.png` — duong val acc theo epoch (neu co `history.csv`)
- `logs/compare.log`

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
- Doi backbone: can train lai; checkpoint cu khong tai duoc sang kien truc khac.
- Checkpoint / `train_config` ghi `efficientnet_b4` hoac `mobilenet_v3` (phien cu): khong con ho tro; dung `efficientnet_b0` / `mobilenet_v2` va train lai.

## Du lieu y te

Chi dung du lieu dung quy dinh va da duoc phep.
