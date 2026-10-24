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
