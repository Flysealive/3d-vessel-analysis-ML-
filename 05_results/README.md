# 分析結果資料夾

## 📅 更新日期: 2024-09-05

本資料夾包含導管更換預測分析的最終結果（已修正資料洩漏問題）。

## 📁 檔案清單

### 分析程式（修正版）
- `analyze_parameter_importance_fixed.py` - 14個參數重要性分析（使用Pipeline防止資料洩漏）
- `analyze_combined_angles_fixed.py` - 角度參數組合分析（使用Pipeline防止資料洩漏）

### 資料檔案
- `vessel_morphology_14_parameters.csv` - 完整14個參數的原始資料（94個血管）
- `20250905_parameter_importance_final(修正資料洩露).csv` - 參數重要性最終結果
- `20250908_最後分析結果(已經修正資料洩露).csv` - 角度組合分析結果

### 視覺化圖表
- `parameter_importance_analysis_fixed.png` - 參數重要性視覺化
- `combined_angles_analysis_fixed.png` - 角度組合效果視覺化

## 🔑 重要發現

### 最顯著的預測參數（P < 0.05）
1. **angle_degrees** (P=0.0198) - 左鎖骨下動脈-主動脈弓角度
2. **arch_angle** (P=0.0433) - 主動脈弓角度
3. **segment_angle_max** (P=0.0433) - 最大分段角度

### 模型表現（5折交叉驗證）
- 隨機森林：準確率 83.0% ± 2.0%
- 邏輯迴歸：準確率 62.9% ± 8.5%
- 最佳單一參數（segment_angle_max）：準確率 63.9% ± 4.9%

### 臨床建議
- **高風險指標**：angle_degrees < 70° 且 segment_angle_max > 60°
- 導管更換率：高風險組約27.3%，低風險組約5.6%

## 🔧 技術說明

### 資料洩漏修正
所有分析都使用sklearn Pipeline確保：
- 標準化只在訓練集上進行
- 每個CV fold獨立預處理
- 測試集從未參與統計計算

### 14個分析參數
**原有參數（3個）**：
1. left_subclavian_diameter_cm
2. aortic_arch_diameter_cm
3. angle_degrees

**形態參數（11個）**：
4. distance_factor
5. arc_chord_ratio_mean
6. arc_chord_ratio_max
7. curvature_mean
8. curvature_max
9. torsion_mean
10. torsion_max
11. branch_angle
12. arch_angle
13. segment_angle_mean
14. segment_angle_max

## 📊 資料集資訊
- 總樣本數：94個血管
- 標籤分布：不需更換78個(83%)，需要更換16個(17%)
- 嚴重的類別不平衡需要使用balanced class weights

## 📝 使用說明

執行分析：
```bash
python analyze_parameter_importance_fixed.py
python analyze_combined_angles_fixed.py
```

所需套件：
- numpy, pandas
- scikit-learn >= 0.24
- matplotlib, seaborn
- scipy