# 血管分析圖表組織結構

## 資料夾結構
```
batch_10_vessels_organized/
├── figure1_original/          # 所有血管的原始3D視圖
├── figure2_with_centerline/   # 所有血管的中心線視圖
├── figure3_angle_analysis/    # 所有血管的角度分析
├── figure4_xy_projection/     # 所有血管的XY平面投影
├── figure5_xz_projection/     # 所有血管的XZ平面投影
├── STL_cropped/               # 所有裁剪後的STL檔案
├── centerlines/               # 所有中心線數據檔案
└── SUMMARY.md                 # 處理總結報告
```

## 檔案命名規則
- 圖表: `{血管名稱}_{圖表類型}.png`
- STL: `{血管名稱}_cropped.stl`
- 中心線: `{血管名稱}_centerline.npy`

## 統計
- 血管總數: 10
- 圖表類型: 5種
- 總圖表數: 50個

## 使用方式

### 查看特定類型的所有血管
例如查看所有血管的中心線視圖：
```
打開 figure2_with_centerline/ 資料夾
```

### 比較不同血管的同一視角
將同一資料夾內的圖片並排查看即可比較

### 批次處理
每個資料夾內的檔案格式統一，便於批次處理和分析

## 圖表說明
1. **figure1_original**: 原始血管3D模型視圖
2. **figure2_with_centerline**: 血管與提取的中心線
3. **figure3_angle_analysis**: 角度變化分析（顏色編碼）
4. **figure4_xy_projection**: XY平面（俯視圖）投影
5. **figure5_xz_projection**: XZ平面（側視圖）投影

生成日期: 2024-08-23
