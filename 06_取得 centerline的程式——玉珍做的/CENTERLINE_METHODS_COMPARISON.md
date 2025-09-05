# 血管中心線提取方法比較與排序

## 📋 方法總覽

在 `/Users/julie/folder/vessel/organized` 專案中，共有 **3種主要的STL血管中心線提取方法**：

### 排序清單

| 優先順序 | 方法名稱 | 檔案位置 | 建議使用場景 |
|---------|---------|---------|------------|
| 1 | **中軸線方法 (Medial Axis Method)** | `01_pipeline_scripts/vessel_centerline_medial_axis.py` | 簡單血管結構、快速處理 |
| 2 | **增強版方法 (Enhanced Method)** | `01_pipeline_scripts/vessel_centerline_enhanced.py` | 需要高精度中心線、強調中心性 |
| 3 | **分支點方法 (Branch Point Method)** | `01_pipeline_scripts/vessel_centerline_branch_point.py` | 複雜分支血管、需要準確識別分叉點 |

---

## 🔍 詳細方法比較

### 1. 中軸線方法 (Medial Axis Method) ⭐⭐⭐
**檔案：** `vessel_centerline_medial_axis.py`
**類別：** `VesselMedialAxisExtractor`

#### 特點：
- ✅ **基礎中軸線提取**
- ✅ 使用體素化和細化算法
- ✅ 相對簡單快速
- ✅ 適合規則血管結構

#### 技術細節：
```python
- 標準skeletonize_3d算法
- 體素分辨率：1.0 mm
- KDTree鄰近搜索
- 基礎平滑處理
```

#### 輸出格式：
- CSV檔案
- 基礎視覺化
- 點雲數據

#### 適用場景：
- 簡單血管結構
- 快速原型開發
- 初步分析

---

### 2. 增強版方法 (Enhanced Method) ⭐⭐⭐⭐
**檔案：** `vessel_centerline_enhanced.py`
**類別：** `EnhancedVesselExtractor`

#### 特點：
- ✅ **端點檢測優化**：使用連接度分析
- ✅ **路徑優化**：Dijkstra最短路徑算法
- ✅ **中心性改進**：距離場方法確保通過血管腔中央
- ✅ 整合多種改進算法

#### 技術細節：
```python
- 使用距離變換（distance_transform_edt）
- 連接度分析找出真正端點
- 高斯濾波平滑處理
- Plotly 3D視覺化
```

#### 輸出格式：
- CSV檔案
- 3D視覺化圖表
- 統計報告

#### 適用場景：
- 需要高精度中心線
- 強調中心線的中心性
- 血管狹窄或彎曲處理

---

### 3. 分支點方法 (Branch Point Method) ⭐⭐⭐⭐⭐
**檔案：** `vessel_centerline_branch_point.py`
**類別：** `VesselCenterlineExtractor`

#### 特點：
- ✅ **自動檢測血管分支點**（分叉位置）
- ✅ 選擇Z軸第二高的分支點作為起點
- ✅ 使用Dijkstra最短路徑算法
- ✅ 生成3D互動視覺化（HTML格式）
- ✅ 包含完整的統計資訊輸出

#### 技術細節：
```python
- 體素化解析度：1.0 mm
- 骨架化算法：skeletonize_3d
- 路徑搜索：NetworkX圖論算法
- 平滑處理：Gaussian filter
```

#### 輸出格式：
- CSV檔案（x,y,z座標）
- HTML互動視覺化
- NPY二進制陣列

#### 適用場景：
- 複雜的血管分支結構
- 需要準確識別分叉點
- 學術研究和臨床分析

---

## 📊 性能比較

| 方法 | 處理速度 | 精確度 | 複雜度處理 | 視覺化品質 |
|-----|---------|--------|-----------|------------|
| 中軸線方法 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 增強版方法 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 分支點方法 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 🚀 使用建議

### 推薦工作流程：

1. **初步分析**：使用中軸線方法快速瀏覽
2. **詳細處理**：使用分支點方法處理複雜血管
3. **精確優化**：對關鍵區域使用增強版方法

### 批次處理腳本：
- `test_two_vessels_centerline.py` - 測試處理兩個血管
- `process_all_vessels_branch_point.py` - 批次處理所有血管

### 輔助工具：
- `generate_centerline_stl.py` - 生成中心線的STL管道模型
- `04_centerline_extract.py` - 整合的中心線提取工具

---

## 📁 專案結構

```
/Users/julie/folder/vessel/organized/
├── 01_pipeline_scripts/           # 核心算法腳本
│   ├── vessel_centerline_medial_axis.py     # 方法1
│   ├── vessel_centerline_enhanced.py        # 方法2
│   └── vessel_centerline_branch_point.py    # 方法3
├── 02_data_input/                 # 輸入資料
│   └── STL_smooth_solid/          # STL檔案
├── 03_data_output/                # 輸出結果
│   └── vessel_centerlines/        # 中心線結果
└── 04_documentation/              # 文件
    └── BRANCH_POINT_METHOD.md     # 方法說明
```

---

## 🔧 技術依賴

所有方法共同依賴：
- `numpy` - 數值計算
- `trimesh` - 3D網格處理
- `scipy` - 科學計算
- `scikit-image` - 影像處理
- `plotly` - 3D視覺化
- `networkx` - 圖論算法（分支點方法）

---

## 📝 備註

- 所有方法都已在實際血管數據上測試
- 建議根據具體血管結構選擇合適方法
- 可以組合使用多種方法以獲得最佳結果
- 輸出格式統一，便於後續分析

更新日期：2024-09-01