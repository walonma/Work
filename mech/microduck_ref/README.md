# Microduck 原廠模型（參考用）

把 Pollen Robotics 公開的 Microduck 模擬模型（MJCF ＋ 38 個網格）轉成可以審視的 3D 頁面，當作「一模一樣」的外型與機構基準。

- 來源：[pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) `src/mjlab_microduck/robot/microduck/`（由 Onshape 經 onshape-to-robot 匯出）
- 授權：軟體 Apache 2.0；**硬體設計檔（網格）CC BY-NC-SA**：可以非商業使用與改作，須標示來源、以相同授權分享，**不可商業使用**。
- 本資料夾只放轉換工具和精簡過的審圖頁，不放原始 STL。

## 內容

| 檔案 | 內容 |
| --- | --- |
| `viewer/microduck_viewer.html` | 互動審圖頁：14 個模擬關節滑桿、原廠的初始／站立／坐下／收折姿勢、5 類料件切換 |
| `tools/mjcf_scan.py` | 解析 MJCF：連桿樹、關節、每個可見零件的網格與位置 |
| `tools/export_viewer.py` | 讀 STL、精簡網格（每件 ≤ 9000 面）、輸出 `viewer/data.json` |
| `tools/build_viewer.py` | 把資料嵌進 `viewer/template.html` |
| `tools/export_step.py` | 輸出拆件 STEP：每個零件一個檔（`STEP/parts/*.stp`）＋指定姿勢的組立檔＋`BOM_microduck.csv` |
| `tools/verify_step.py` | 讀回組立檔，檢查 70 個零件的位置與原廠模型一致 |
| `BOM_microduck.csv` | 38 個料號、70 件：中文名稱、分類、數量、外形尺寸、對應檔案 |

## 拆件 STEP

原廠只公開網格（STL），沒有公開 Onshape 的實體 CAD，所以 STEP 用 **AP242 網格（tessellated）** 格式輸出：保留原廠網格的完整精度，不做任何修改，檔案也夠小（38 個零件約 30 MB）。SolidWorks、Fusion、Creo、NX 都能匯入，但匯入後是網格本體，不是可以改特徵的實體；要修改時需要在 CAD 裡轉成實體，或重新建模。

組立檔依連桿分層（身體、左右腿各 5 節、脖子、頭），每個零件只存一份，多處使用時只記位置。`STAND`（站立）姿勢的零件放在原廠位置上；`FOLD`（收折）姿勢以身體為基準輸出，沒有移到地面上。

```bash
python tools/export_step.py SRC STAND     # 或 INIT / SIT / FOLD
python tools/verify_step.py SRC STAND
```

`STEP/` 和打包檔不進 git，需要時用上面的指令重新產生。

## 從原廠模型讀到的機構

- 15 顆 XL330-M288 伺服：每條腿 5 顆；脖子俯仰、頭部俯仰、轉向、側歪各 1 顆；嘴巴 1 顆（模擬中固定）。
- 14 顆軸承：22 × 16 × 4 有 11 顆，踝部用的另一種規格有 3 顆。
- 結構件：身體骨架、電池座、主板固定座、髖部轉向→側擺支架 × 2、髖部支架 × 2、大腿 × 2、大腿補強板 × 2、小腿 × 2、踝支架 × 2、脖子連桿 × 2、頭部俯仰支架、頭部轉向／側歪機構、頭部馬達座。
- 外殼與外觀：身體左右殼、頭殼上下、臉部面板、眼框、上下嘴喙和軟墊、腳掌和腳底軟墊 × 2。
- 電子（模擬模型版本）：Raspberry Pi Zero 2W 外形主板＋擴充板、NP-F970 外形電池、喇叭、M12 鏡頭＋鏡頭座。量產版改用 RK3566，模擬檔還沒更新。

## 重建

```bash
# 下載 robot_allcollisions.xml、scene.xml 和 assets/*.stl 到 SRC 之後：
python tools/export_viewer.py SRC && python3 tools/build_viewer.py
```

這裡的 `python` 是 `uvx --no-config --managed-python --python 3.13 --from cadgen==0.7.26 python`（需要 numpy）。
