# 血管中心線提取方法集合
## Vessel Centerline Extraction Methods Collection

這個資料夾包含了3種不同的血管中心線提取方法，每種方法都有其獨特的優勢和適用場景。

---

## 📂 資料夾內容

```
centerline_methods/
├── vessel_centerline_medial_axis.py      # 方法1：中軸線方法（核心）
├── method1_medial_axis_complete.py       # 方法1：完整執行腳本 ⭐
├── vessel_centerline_enhanced.py         # 方法2：增強版方法
├── vessel_centerline_branch_point.py     # 方法3：分支點方法
├── test_two_vessels_centerline.py        # 測試腳本
├── run_centerline_extraction.py          # 整合執行腳本
├── CENTERLINE_METHODS_COMPARISON.md      # 詳細比較文件
└── README.md                              # 本文件
```

---

## 🎯 快速開始

### 安裝依賴
```bash
pip install numpy trimesh scipy scikit-image plotly networkx
```

### 使用範例

#### 1. 中軸線方法（基礎快速）

**使用完整執行腳本（推薦）：**
```bash
# 測試模式（處理前2個檔案）
python method1_medial_axis_complete.py test

# 批次處理
python method1_medial_axis_complete.py batch -n 10

# 處理單個檔案
python method1_medial_axis_complete.py single vessel.stl
```

**或直接使用Python：**
```python
from vessel_centerline_medial_axis import VesselMedialAxisExtractor

extractor = VesselMedialAxisExtractor("path/to/vessel.stl")
extractor.load_mesh()
extractor.voxelize_mesh()
extractor.extract_skeleton()
extractor.order_skeleton_points()
extractor.save_results("output_dir")  # 產生完整輸出
```

#### 2. 增強版方法（高精度）
```python
from vessel_centerline_enhanced import EnhancedVesselExtractor

extractor = EnhancedVesselExtractor("path/to/vessel.stl")
# 執行完整的處理流程
centerline = extractor.process()
```

#### 3. 分支點方法（複雜血管）
```python
from vessel_centerline_branch_point import VesselCenterlineExtractor

# 初始化
extractor = VesselCenterlineExtractor("path/to/vessel.stl")

# 處理
extractor.load_mesh()
extractor.voxelize_mesh(resolution=1.0)
extractor.extract_skeleton()
extractor.find_endpoints()
extractor.find_branch_points()

# 提取中心線
centerline = extractor.extract_main_centerline()

# 生成視覺化
fig = extractor.create_interactive_plot()
fig.write_html("output.html")
```

---

## 🔧 測試腳本

使用提供的測試腳本快速測試兩個血管檔案：

```bash
python test_two_vessels_centerline.py
```

此腳本會：
1. 自動讀取STL檔案
2. 使用分支點方法處理
3. 輸出中心線CSV、NPY檔案
4. 生成HTML視覺化

---

## 📊 方法選擇指南

| 使用場景 | 推薦方法 | 原因 |
|---------|---------|------|
| 快速處理 | 中軸線方法 | 算法簡單，速度快 |
| 需要高精度 | 增強版方法 | 距離場優化確保中心性 |
| 複雜分支血管 | 分支點方法 | 能準確識別分叉點 |
| 批次處理 | 中軸線方法 | 執行速度最快 |
| 學術研究 | 分支點方法 | 提供最完整的資訊 |

---

## 📈 性能參數

### 中軸線方法（方法1）
- 基礎骨架化
- 處理時間：快速
- 記憶體使用：低
- 精確度：中等

### 增強版方法（方法2）
- 距離場計算：高精度
- 處理時間：較長
- 記憶體使用：較高
- 精確度：最高

### 分支點方法（方法3）
- 體素解析度：1.0 mm（可調整）
- 處理時間：中等
- 記憶體使用：中等
- 精確度：高

---

## 📁 輸入輸出格式

### 輸入
- STL格式的3D血管模型
- 支援ASCII和Binary STL

### 輸出
- **CSV檔案**：x,y,z座標點
- **NPY檔案**：NumPy陣列格式
- **HTML檔案**：3D互動視覺化
- **TXT檔案**：統計資訊

---

## 🔍 詳細比較

請參閱 `CENTERLINE_METHODS_COMPARISON.md` 獲取更詳細的方法比較和技術細節。

---

## ⚙️ 自定義參數

### 分支點方法
```python
extractor.voxelize_mesh(resolution=1.0)  # 調整體素大小
extractor.smooth_centerline(sigma=2)     # 調整平滑程度
```

### 增強版方法
```python
extractor.voxel_resolution = 1.0         # 體素解析度
extractor.smoothing_iterations = 5       # 平滑迭代次數
```

### 中軸線方法
```python
extractor.resolution = 1.0                # 體素分辨率
```

---

## 🐛 常見問題

1. **記憶體不足**
   - 降低體素解析度（增加resolution值）
   - 使用中軸線方法處理大檔案

2. **中心線不連續**
   - 檢查STL檔案完整性
   - 調整體素化參數
   - 使用分支點方法

3. **處理時間過長**
   - 降低解析度
   - 使用中軸線方法
   - 考慮並行處理

---

## 📝 更新記錄

- 2024-09-01：整理三種方法到專用資料夾
- 2024-08-29：新增分支點方法
- 2024-08-27：增強版方法優化
- 2024-08-27：基礎中軸線方法實作

---

## 📧 聯繫資訊

如有問題或建議，請參考專案主目錄的聯繫方式。