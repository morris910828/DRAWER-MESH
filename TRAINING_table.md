# `data/table` 完整訓練流程（data → mesh → texture_mesh → Gaussian Splat）

這份文件把 `data/table` 這張桌子從原始資料一路做到可以用 SIBR viewer 打開的
Gaussian Splat 的**所有指令**整理起來。分成四個階段：

| 階段 | 產物 | 執行環境 |
|---|---|---|
| 0. 前置 | `transforms.json`、`marigold_ft/`、`gs_masks/masks/` | `drawer_sdf` |
| 1. Stage 1 – SDF 重建 | `table_sdf_ds1/{config.yml, sdfstudio_models/, mesh_table.ply}` | `drawer_sdf` |
| 2. Mesh 後處理 → `texture_mesh/` | `texture_mesh/mesh_table-only-400k*.obj` `.mtl` `.png` | `drawer_sdf` |
| 3. 遮罩準備 | `data/table/mask/`（50 張全解析度 PNG）| — |
| 4. Stage 4 – Gaussian Splat | `table_gs28/export_ply/splat.ply` | `drawer_splat` |
| 5.（本機）SIBR viewer | `D:\DRAWER\output\table_gs28\` | Windows |

> **路徑約定**：訓練在 Linux 機器（DGX / RTX）上跑，repo 根目錄為
> `/workspace/DRAWER-MESH`，兩個 conda 環境 `drawer_sdf`、`drawer_splat`。
> 下面所有指令的工作目錄都是 repo 根目錄，除非另外標註。
> 本機 Windows 只用來跑 viewer。

> **重要前提**：`data/table` 是 5568×4176 的全解析度資料，50 張影像。
> 全解析度是硬性要求，不可降（見 `project_full_res_training.md`）。
> 下面所有階段都用 `downscale_factor 1`。

---

## 階段 0：前置資料（已完成，僅供重建參考）

`data/table` 目前已經有 `transforms.json`、`sparse/`、`marigold_ft/`、
`gs_masks/masks/`，所以正常情況**這一段可以跳過**。要從零重跑時才需要：

### 0a. COLMAP → transforms.json

```bash
conda activate drawer_sdf

python scripts/colmap_to_drawer.py \
    --data_dir  /workspace/DRAWER-MESH/data/table \
    --downscale 1
```

需要 `data/table/images/`（原始 JPG）。輸出 `data/table/transforms.json` 與
`data/table/sparse/`。

### 0b. Marigold 單目深度 / 法線先驗

```bash
cd marigold
conda activate drawer_sdf

python run.py \
    --checkpoint "GonzaloMG/marigold-e2e-ft-depth" \
    --modality depth \
    --input_rgb_dir /workspace/DRAWER-MESH/data/table/images \
    --output_dir    /workspace/DRAWER-MESH/data/table/marigold_ft

python run.py \
    --checkpoint "GonzaloMG/marigold-e2e-ft-normals" \
    --modality normals \
    --input_rgb_dir /workspace/DRAWER-MESH/data/table/images \
    --output_dir    /workspace/DRAWER-MESH/data/table/marigold_ft

python read_marigold.py --data_dir /workspace/DRAWER-MESH/data/table/marigold_ft
cd ..
```

`run_table_sdf.sh` 會自己建立 `depth/` 與 `normal/` 這兩個 symlink 指向
`marigold_ft/`，不用手動做。

---

## 階段 1：Stage 1 – SDF 重建

一支腳本包辦 BakedSDF 訓練 + 抽桌子 mesh + 抽全景 mesh：

```bash
conda activate drawer_sdf
cd /workspace/DRAWER-MESH

bash scripts/run_table_sdf.sh 1
```

參數 `1` = downscale factor 1（全解析度）。腳本內容：

1. 建 `depth/`、`normal/` symlink。
2. **BakedSDF 訓練** 50k 步 →
   `data/table/table_sdf_ds1/{config.yml, sdfstudio_models/step-00050000.ckpt}`。
   已有 `step-00050000.ckpt` 會自動跳過；有中途 checkpoint 會自動續訓。
   全解析度下週期性 eval 會 OOM，所以腳本在 downscale 1 時把 eval 關掉，
   幾何品質改用抽出來的 mesh 檢查。
3. **抽桌子 mesh** → `table_sdf_ds1/mesh_table.ply`
   bounding box（訓練座標，已含小 margin，量自 COLMAP sparse cloud）：

   ```
   BB_MIN = (-0.71, -0.96, -0.96)
   BB_MAX = ( 0.79,  0.55,  0.41)
   ```

   `--resolution 2048 --marching_cube_threshold 0.0035 --simplify-mesh False`

4. **抽全景 mesh**（備援 / 對照）→ `table_sdf_ds1/mesh_full.ply`，bbox `-1..1`。

輸出重點：`table_sdf_ds1/config.yml`（後面 texture bake 要用）與
`table_sdf_ds1/mesh_table.ply`（原始，約 2、3 千萬面，未簡化）。

---

## 階段 2：Mesh 後處理 → `texture_mesh/`

目標是產出 GS 階段實際吃的檔案：

```
data/table/table_sdf_ds1/texture_mesh/
├── mesh_table-only-400k.obj          ← texture bake 後的 mesh（training space）
├── mesh_table-only-400k.mtl
├── mesh_table-only-400k.png          ← baked 紋理圖
└── mesh_table-only-400k_world.obj    ← 轉回 COLMAP world 座標（GS 訓練用這個）
```

> ⚠️ **這條「400k 鏈」目前是手動跑的**，`run_table_post.sh` 那支腳本描述的是
> 另一套命名（`mesh_table-clean-simplify`、100k 面），產物在硬碟上不存在
> （見 `table_gs_experiments.md` §5.6）。下面是重現「400k 鏈」的指令。

所有指令環境：`conda activate drawer_sdf`，工作目錄 repo 根。
設一個變數少打字：

```bash
SDF=/workspace/DRAWER-MESH/data/table/table_sdf_ds1
```

### 2a. 裁出中間那張桌子（去掉鄰桌碎片與 SDF 飛點）

```bash
python scripts/crop_table.py \
    --input  ${SDF}/mesh_table.ply \
    --output ${SDF}/mesh_table-only.ply
```

`crop_table.py` 用的是量自 mesh 本身的**斜向 box**（預設值就是中間桌），
再只留最大的連通元件——桌子連同桌上所有東西是同一塊（佔 99.4%），
其餘飛點與被 box 切到的鄰桌薄片都會被丟掉。四支桌腳完整保留。

### 2b.（可選）丟掉內部看不到的面

```bash
python scripts/filter_interior_mesh.py \
    --input  ${SDF}/mesh_table-only.ply \
    --output ${SDF}/mesh_table-only-clean.ply \
    --n_samples 30
```

若跳過這步，下一步的 `--input` 直接用 `mesh_table-only.ply`。

### 2c. 簡化到約 40 萬面

```bash
python scripts/simplify_mesh.py \
    --input  ${SDF}/mesh_table-only-clean.ply \
    --output ${SDF}/mesh_table-only-400k.ply \
    --target_faces 400000 \
    --min_component_faces 200
```

`--min_component_faces 200` 很關鍵：預設 10000 在 40 萬面的 mesh 上會把
桌腳這種小元件整支刪掉。

### 2d. Bake 紋理（需要 GPU，會載入訓練好的 SDF 模型）

```bash
mkdir -p ${SDF}/texture_mesh

cd /workspace/DRAWER-MESH/sdf
python scripts/texture.py \
    --load-config ${SDF}/config.yml \
    --output-dir  ${SDF}/texture_mesh \
    --input_mesh_filename ${SDF}/mesh_table-only-400k.ply \
    --target_num_faces 400000
cd /workspace/DRAWER-MESH
```

輸出 `texture_mesh/mesh_table-only-400k.{obj,mtl,png}`。
輸出檔名 = 輸入 PLY 的 basename，所以 PLY 一定要叫 `mesh_table-only-400k.ply`。
`--target_num_faces` 設成不小於 mesh 面數（400000），才不會又被砍一次。

### 2e. 轉回 COLMAP world 座標

```bash
python scripts/mesh_to_world.py \
    --input     ${SDF}/texture_mesh/mesh_table-only-400k.obj \
    --transform /workspace/DRAWER-MESH/data/table/transforms.json \
    --output    ${SDF}/texture_mesh/mesh_table-only-400k_world.obj \
    --extra-scale 1.0
```

`_world.obj` 的 `mtllib` 仍指向原本的 `.mtl` / `.png`，不用改。
這一步結束後 `texture_mesh/` 就齊了。

> 如果 SDF 重新訓練 / mesh 重新 bake 過，**遮罩必須重切**（見階段 3）。

---

## 階段 3：遮罩準備

GS 訓練一定要帶遮罩（桌子剪影），否則背景會灌爆 SSIM loss、splat 全糊
（見 `feedback_fg_mask_needs_tight_silhouette.md`）。

`gs_masks/masks/` 應該已有 50 張全解析度 PNG。GS 腳本的 pre-flight 會自動
把 `data/table/mask/` symlink 過去，並檢查數量：

```bash
ls -1 /workspace/DRAWER-MESH/data/table/gs_masks/masks/*.png | wc -l   # 應為 50
ls -1 /workspace/DRAWER-MESH/data/table/images/*.JPG        | wc -l   # 應為 50
```

若 mesh 重新 bake 過、遮罩需要用**當前的 400k mesh** 重切：

```bash
conda run -n drawer_splat python scripts/save_mesh_depth_masks.py \
    --data_dir      /workspace/DRAWER-MESH/data/table \
    --mesh_path     ${SDF}/texture_mesh/mesh_table-only-400k.obj \
    --dataparser_tf /workspace/DRAWER-MESH/data/table/<某次 GS run>/dataparser_transforms.json \
    --out_dir       /workspace/DRAWER-MESH/data/table/gs_masks
```

`--mesh_path` 用 **training space** 的 `.obj`（沒有 `_world`），
`--dataparser_tf` 來自任一次 GS run 的 `dataparser_transforms.json`。
輸出 `gs_masks/masks/`（二值 PNG）與 `gs_masks/overlay/`（疊圖確認用）。

---

## 階段 4：Stage 4 – Gaussian Splat on Mesh

目前最新的實驗腳本是 `run_table_gs28.sh`（gs27 + `base_opacity_floor 0.65`）。
每一輪都是獨立腳本，改參數就複製一份改號碼。

```bash
conda activate drawer_splat
cd /workspace/DRAWER-MESH

bash scripts/run_table_gs28.sh                # 訓練 + 匯出
# bash scripts/run_table_gs28.sh --export-only  # 只從最後的 checkpoint 匯出
```

腳本會：

1. **Pre-flight**：確認遮罩存在、數量對，並 dry-run dataparser，
   解析不到遮罩就中止（避免又在無遮罩下訓練）。
2. **訓練** 30k 步，吃的 mesh 是
   `${SDF}/texture_mesh/mesh_table-only-400k_world.obj`
   （腳本裡 `SDF_EXP=table_sdf_ds1`、`AREA=1e-4`）。
   log 寫到 `data/table/gs_logs/table_gs28_<時間>.log`；
   額外資訊寫到 `data/table/gs_extra_info/table_gs28.pt`
   （匯出時要用來還原 `is_base` 等 per-Gaussian 狀態）。
   checkpoint 在 `data/table/table_gs28/nerfstudio_models/`，有就自動續訓。
3. **匯出** →
   `data/table/table_gs28/export_ply/splat.ply`

### 訓練後的 PASS / FAIL 檢查（每次改參數都要一起回報）

依 `project_table_gs_hard_constraints.md`：覆蓋率必須 100%，外觀品質不能犧牲，
兩個數字要同時報。

```bash
# 1. 空隙覆蓋率 @ 0.01（硬性要求 100.000%）
python scripts/analyze_gap_coverage.py \
    data/table/table_gs28/export_ply/splat.ply --samples 16

# 2. 分層塗抹 / 每面 face-based Gaussian 數（中位要 >= 2）
python scripts/analyze_layer_smear.py \
    data/table/table_gs28/export_ply/splat.ply \
    --transforms data/table/table_gs28/dataparser_transforms.json \
    --frame GOPR0236

# 3. 遮罩內 / 外 PSNR（跟前一輪比）
python scripts/compare_masked_psnr.py \
    --data data/table \
    --exp  table_gs28 \
    --out  data/table/psnr_compare/gs28 \
    --splits eval train
```

---

## 階段 5：（本機 Windows）用 SIBR viewer 打開

把 `data/table/table_gs28/` 從訓練機拉回本機 `D:\DRAWER\output\table_gs28\`，
至少要有 `export_ply/splat.ply`。把 `splat.ply` 放到 model 目錄根：

```powershell
# D:\DRAWER\output\table_gs28\splat.ply  ← 從 export_ply\splat.ply 複製上來
```

產生 SIBR 需要的目錄結構（`cameras.json` / `cfg_args` /
`point_cloud/iteration_30000/point_cloud.ply`）：

```powershell
cd D:\DRAWER
python setup_sibr_model.py --model D:\DRAWER\output\table_gs28 --data D:\DRAWER\data\table --force
```

> 你先前用過的版本有 `--link --no-transform` 兩個旗標
> （`python setup_sibr_model.py --model D:\DRAWER\output\table_gs27 --data D:\DRAWER\data\table --link --force --no-transform`）。
> 目前 repo 內的 `setup_sibr_model.py` 只有 `--model / --data / --white-background /
> --black-background / --force`；splat 已經在 world 座標，本來就不需要再 transform。

第一次要先 build viewer：

```powershell
cd D:\DRAWER\SIBR_viewers
cmake --build build --config Release --target install
```

打開：

```powershell
cd D:\DRAWER\SIBR_viewers\install\bin
.\SIBR_gaussianViewer_app -m D:\DRAWER\output\table_gs28
```

---

## 一頁速查

```bash
# ── Linux 訓練機，repo 根 = /workspace/DRAWER-MESH ──
SDF=/workspace/DRAWER-MESH/data/table/table_sdf_ds1

# 1. SDF
conda activate drawer_sdf
bash scripts/run_table_sdf.sh 1

# 2. mesh → texture_mesh（手動 400k 鏈）
python scripts/crop_table.py           --input ${SDF}/mesh_table.ply         --output ${SDF}/mesh_table-only.ply
python scripts/filter_interior_mesh.py --input ${SDF}/mesh_table-only.ply    --output ${SDF}/mesh_table-only-clean.ply --n_samples 30
python scripts/simplify_mesh.py        --input ${SDF}/mesh_table-only-clean.ply --output ${SDF}/mesh_table-only-400k.ply --target_faces 400000 --min_component_faces 200
( cd sdf && python scripts/texture.py --load-config ${SDF}/config.yml --output-dir ${SDF}/texture_mesh --input_mesh_filename ${SDF}/mesh_table-only-400k.ply --target_num_faces 400000 )
python scripts/mesh_to_world.py --input ${SDF}/texture_mesh/mesh_table-only-400k.obj --transform /workspace/DRAWER-MESH/data/table/transforms.json --output ${SDF}/texture_mesh/mesh_table-only-400k_world.obj --extra-scale 1.0

# 3. + 4. GS（腳本自己處理遮罩 symlink 與匯出）
conda activate drawer_splat
bash scripts/run_table_gs28.sh

# 檢查
python scripts/analyze_gap_coverage.py data/table/table_gs28/export_ply/splat.ply --samples 16
python scripts/compare_masked_psnr.py --data data/table --exp table_gs28 --out data/table/psnr_compare/gs28

# ── 本機 Windows ──
# copy data/table/table_gs28 → D:\DRAWER\output\table_gs28 ,  export_ply\splat.ply → splat.ply
python setup_sibr_model.py --model D:\DRAWER\output\table_gs28 --data D:\DRAWER\data\table --force
.\SIBR_viewers\install\bin\SIBR_gaussianViewer_app -m D:\DRAWER\output\table_gs28
```
