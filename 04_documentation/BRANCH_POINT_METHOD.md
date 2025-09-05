# 血管中心線提取 - 分支點方法 (Branch Point Method)

## 概述
這是第二個額外的中心線提取方法，專門設計用於從血管分支點開始提取中心線。

## 檔案位置
`01_pipeline_scripts/vessel_centerline_branch_point.py`

## 方法特點

### 1. 自動分支點檢測
- 使用3D骨架化算法提取血管骨架
- 自動識別分支點（鄰居數 ≥ 3 的骨架點）
- 檢查鄰居方向角度以確認真實分支點

### 2. 起點選擇策略
- 對所有分支點按Z軸高度排序
- 選擇**Z軸第二高的分支點**作為起點
- 如果只有一個分支點，則使用該分支點

### 3. 中心線提取
- 使用Dijkstra最短路徑算法
- 從選定的分支點找到最遠的點
- 生成平滑的中心線路徑

## 使用方法

```python
# 直接執行腳本
python vessel_centerline_branch_point.py

# 或在Python中調用
from vessel_centerline_branch_point import VesselCenterlineExtractor

extractor = VesselCenterlineExtractor(stl_file_path)
extractor.load_mesh()
centerline = extractor.extract_main_centerline()
fig = extractor.create_interactive_plot()
```

## 輸入輸出

### 輸入
- STL檔案路徑（血管3D模型）
- 預設處理目錄：`02_data_input/STL_smooth_solid/`

### 輸出
1. **CSV檔案** (`*_centerline.csv`)
   - 格式：x,y,z座標
   - 包含標題行
   - 每行一個中心線點

2. **HTML檔案** (`*_auto_centerline.html`)
   - 3D互動視覺化
   - 可旋轉、縮放、平移

## 視覺化說明

### 標記說明
- 🟢 **綠色圓形**：中心線起點
- 🟧 **橙色正方形**：中心線終點
- 💜 **紫色菱形**：所有檢測到的分支點
- 🟡 **黃色大菱形**：選中的第二高分支點（起點位置）
- 🔴 **紅色線**：主要中心線路徑
- ⚪ **灰色點**：血管表面點雲

## 參數設置

### 主要參數
- `resolution`: 體素化分辨率（預設 1.0 mm）
- `iterations`: 形態學操作迭代次數
- `smoothing_factor`: 中心線平滑參數

## 測試結果

### 3DModel822993
- 檢測到1個分支點
- 起點：(17.4, 53.4, 115.5)
- 生成127個中心線點

### 3DModel872531
- 檢測到11個分支點
- 選擇第二高分支點：Z=104.5
- 起點：(44.5, 78.4, 104.5)
- 生成140個中心線點

## 優勢
1. 自動找到血管分叉位置
2. 適合處理複雜的分支血管結構
3. 不需要手動指定起點
4. 生成的中心線從分支處開始，更符合臨床需求

## 限制
1. 需要血管有明顯的分支結構
2. 對於沒有分支的直管效果可能不佳
3. 計算時間較長（骨架化和分支檢測）

## 更新歷史
- 2024-08-29：初始版本，實現分支點檢測和中心線提取
- 2024-08-29：加入Z軸第二高分支點選擇邏輯
- 2024-08-29：更新視覺化標記（綠色起點、橙色終點）