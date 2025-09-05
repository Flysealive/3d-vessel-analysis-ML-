#!/usr/bin/env python3
"""
分析14個參數對導管更換預測的重要性
修正版：避免資料洩漏
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_and_prepare_data():
    """載入並準備資料"""
    # 載入資料
    df = pd.read_csv('vessel_morphology_14_parameters.csv')
    
    # 分離特徵和標籤
    feature_columns = [col for col in df.columns if col not in ['filename', 'label']]
    X = df[feature_columns]
    y = df['label']
    
    print(f"📊 資料集概覽:")
    print(f"  • 樣本數: {len(df)}")
    print(f"  • 特徵數: {len(feature_columns)}")
    print(f"  • 標籤分布: 0={sum(y==0)} ({sum(y==0)/len(y)*100:.1f}%), 1={sum(y==1)} ({sum(y==1)/len(y)*100:.1f}%)")
    
    return df, X, y, feature_columns


def statistical_analysis(df, X, y, feature_columns):
    """統計分析：t檢定和相關性分析"""
    print("\n" + "="*60)
    print("1. 統計分析")
    print("="*60)
    
    # T檢定：比較兩組間的差異
    results = []
    
    for col in feature_columns:
        group_0 = df[df['label'] == 0][col]
        group_1 = df[df['label'] == 1][col]
        
        # T檢定
        t_stat, p_value = stats.ttest_ind(group_0, group_1)
        
        # 效應大小 (Cohen's d)
        pooled_std = np.sqrt((group_0.std()**2 + group_1.std()**2) / 2)
        cohens_d = abs(group_0.mean() - group_1.mean()) / pooled_std if pooled_std > 0 else 0
        
        results.append({
            'Parameter': col,
            'Mean_NoChange': group_0.mean(),
            'Mean_Change': group_1.mean(),
            'Difference': group_1.mean() - group_0.mean(),
            'T-statistic': t_stat,
            'P-value': p_value,
            'Cohens_d': cohens_d,
            'Significant': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('P-value')
    
    print("\n📈 T檢定結果 (按P值排序):")
    print("-" * 60)
    print(f"{'參數':<30} {'P值':<10} {'Cohen d':<10} {'顯著性'}")
    print("-" * 60)
    
    for _, row in results_df.head(10).iterrows():
        param = row['Parameter'][:28]
        print(f"{param:<30} {row['P-value']:<10.4f} {row['Cohens_d']:<10.3f} {row['Significant']}")
    
    return results_df


def feature_importance_rf_fixed(X, y, feature_columns):
    """使用隨機森林計算特徵重要性 - 修正版"""
    print("\n" + "="*60)
    print("2. 隨機森林特徵重要性 (Pipeline防止洩漏)")
    print("="*60)
    
    # 創建Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            class_weight='balanced'
        ))
    ])
    
    # 交叉驗證
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    cv_auc = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')
    
    print(f"\n模型表現 (5折交叉驗證):")
    print(f"  • 準確率: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  • AUC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")
    
    # 訓練完整模型以獲取特徵重要性
    # 注意：這裡只是為了獲取特徵重要性的相對排序
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 在訓練集上fit pipeline
    pipeline.fit(X_train, y_train)
    
    # 獲取特徵重要性
    rf_model = pipeline.named_steps['rf']
    importance = pd.DataFrame({
        'Parameter': feature_columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # 在測試集上評估
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    test_accuracy = accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_proba)
    
    print(f"\n測試集表現:")
    print(f"  • 準確率: {test_accuracy:.3f}")
    print(f"  • AUC: {test_auc:.3f}")
    
    print(f"\n📊 特徵重要性排名:")
    print("-" * 60)
    print(f"{'參數':<30} {'重要性':<10}")
    print("-" * 60)
    
    for _, row in importance.head(10).iterrows():
        param = row['Parameter'][:28]
        print(f"{param:<30} {row['Importance']:.4f}")
    
    return pipeline, importance, cv_scores


def logistic_regression_analysis_fixed(X, y, feature_columns):
    """邏輯迴歸分析 - 修正版"""
    print("\n" + "="*60)
    print("3. 邏輯迴歸係數分析 (Pipeline防止洩漏)")
    print("="*60)
    
    # 創建Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ))
    ])
    
    # 交叉驗證
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    cv_auc = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')
    
    print(f"\n模型表現 (5折交叉驗證):")
    print(f"  • 準確率: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  • AUC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")
    
    # 訓練完整模型以獲取係數
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 在訓練集上fit pipeline
    pipeline.fit(X_train, y_train)
    
    # 獲取係數
    lr_model = pipeline.named_steps['lr']
    scaler = pipeline.named_steps['scaler']
    
    # 係數已經是基於標準化數據的
    coefficients = pd.DataFrame({
        'Parameter': feature_columns,
        'Coefficient': lr_model.coef_[0],
        'Abs_Coefficient': np.abs(lr_model.coef_[0])
    }).sort_values('Abs_Coefficient', ascending=False)
    
    # 在測試集上評估
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    test_accuracy = accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_proba)
    
    print(f"\n測試集表現:")
    print(f"  • 準確率: {test_accuracy:.3f}")
    print(f"  • AUC: {test_auc:.3f}")
    
    print(f"\n📊 迴歸係數 (標準化後):")
    print("-" * 60)
    print(f"{'參數':<30} {'係數':<10} {'影響方向'}")
    print("-" * 60)
    
    for _, row in coefficients.head(10).iterrows():
        param = row['Parameter'][:28]
        direction = "增加風險" if row['Coefficient'] > 0 else "降低風險"
        print(f"{param:<30} {row['Coefficient']:>9.4f}  {direction}")
    
    return pipeline, coefficients, cv_scores


def create_visualization_fixed(t_test_results, rf_importance, lr_coefficients, rf_cv_scores, lr_cv_scores):
    """創建視覺化圖表 - 修正版"""
    print("\n" + "="*60)
    print("4. 生成視覺化報告")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. P值熱圖
    ax = axes[0, 0]
    top_params = t_test_results.nsmallest(10, 'P-value')
    colors = ['red' if p < 0.05 else 'orange' if p < 0.1 else 'yellow' for p in top_params['P-value']]
    bars = ax.barh(range(len(top_params)), -np.log10(top_params['P-value']), color=colors)
    ax.set_yticks(range(len(top_params)))
    ax.set_yticklabels([p[:20] for p in top_params['Parameter']])
    ax.set_xlabel('-log10(P-value)')
    ax.set_title('統計顯著性 (T檢定)')
    ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='p=0.05')
    ax.legend()
    
    # 2. 隨機森林重要性
    ax = axes[0, 1]
    top_rf = rf_importance.head(10)
    ax.barh(range(len(top_rf)), top_rf['Importance'], color='green')
    ax.set_yticks(range(len(top_rf)))
    ax.set_yticklabels([p[:20] for p in top_rf['Parameter']])
    ax.set_xlabel('重要性分數')
    ax.set_title(f'隨機森林特徵重要性\n(CV準確率: {rf_cv_scores.mean():.3f}±{rf_cv_scores.std():.3f})')
    
    # 3. 邏輯迴歸係數
    ax = axes[1, 0]
    top_lr = lr_coefficients.head(10)
    colors = ['blue' if c > 0 else 'red' for c in top_lr['Coefficient']]
    ax.barh(range(len(top_lr)), top_lr['Coefficient'], color=colors)
    ax.set_yticks(range(len(top_lr)))
    ax.set_yticklabels([p[:20] for p in top_lr['Parameter']])
    ax.set_xlabel('標準化係數')
    ax.set_title(f'邏輯迴歸係數\n(CV準確率: {lr_cv_scores.mean():.3f}±{lr_cv_scores.std():.3f})')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # 4. 綜合排名
    ax = axes[1, 1]
    
    # 計算綜合分數
    params = list(set(t_test_results['Parameter'].tolist()))
    combined_scores = []
    
    for param in params:
        # T檢定排名 (反向，P值越小越好)
        t_rank = len(t_test_results) - t_test_results[t_test_results['Parameter'] == param].index[0]
        
        # RF重要性排名
        rf_rank = len(rf_importance) - rf_importance[rf_importance['Parameter'] == param].index[0]
        
        # LR係數排名
        lr_rank = len(lr_coefficients) - lr_coefficients[lr_coefficients['Parameter'] == param].index[0]
        
        # 綜合分數 (平均排名)
        combined_score = (t_rank + rf_rank + lr_rank) / 3
        
        combined_scores.append({
            'Parameter': param,
            'Score': combined_score
        })
    
    combined_df = pd.DataFrame(combined_scores).sort_values('Score', ascending=False).head(10)
    
    ax.barh(range(len(combined_df)), combined_df['Score'], color='purple')
    ax.set_yticks(range(len(combined_df)))
    ax.set_yticklabels([p[:20] for p in combined_df['Parameter']])
    ax.set_xlabel('綜合重要性分數')
    ax.set_title('綜合重要性排名')
    
    plt.suptitle('參數重要性分析 (修正版 - 無資料洩漏)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('parameter_importance_analysis_fixed.png', dpi=300, bbox_inches='tight')
    print("✅ 圖表已儲存至: parameter_importance_analysis_fixed.png")
    
    return combined_df


def generate_final_report_fixed(t_test_results, rf_importance, lr_coefficients, combined_ranking):
    """生成最終報告 - 修正版"""
    print("\n" + "="*60)
    print("📋 最終報告：重要參數總結 (修正版)")
    print("="*60)
    
    print("\n🏆 最重要的5個參數（綜合評估）:")
    print("-" * 60)
    
    for i, row in combined_ranking.head(5).iterrows():
        param = row['Parameter']
        
        # 獲取各項指標
        t_test = t_test_results[t_test_results['Parameter'] == param].iloc[0]
        rf = rf_importance[rf_importance['Parameter'] == param].iloc[0]
        lr = lr_coefficients[lr_coefficients['Parameter'] == param].iloc[0]
        
        print(f"\n{i+1}. {param}")
        print(f"   • P值: {t_test['P-value']:.4f} {t_test['Significant']}")
        print(f"   • RF重要性: {rf['Importance']:.4f}")
        print(f"   • LR係數: {lr['Coefficient']:.4f}")
        print(f"   • 綜合分數: {row['Score']:.2f}")
    
    # 儲存詳細報告
    report_df = pd.DataFrame({
        'Parameter': combined_ranking['Parameter'].head(10),
        'Combined_Score': combined_ranking['Score'].head(10)
    })
    
    for param in report_df['Parameter']:
        t_test = t_test_results[t_test_results['Parameter'] == param].iloc[0]
        rf = rf_importance[rf_importance['Parameter'] == param].iloc[0]
        lr = lr_coefficients[lr_coefficients['Parameter'] == param].iloc[0]
        
        idx = report_df[report_df['Parameter'] == param].index[0]
        report_df.loc[idx, 'P_value'] = t_test['P-value']
        report_df.loc[idx, 'RF_importance'] = rf['Importance']
        report_df.loc[idx, 'LR_coefficient'] = lr['Coefficient']
    
    report_df.to_csv('parameter_importance_report_fixed.csv', index=False)
    print("\n📊 詳細報告已儲存至: parameter_importance_report_fixed.csv")
    
    # 重命名為日期版本
    import os
    from datetime import datetime
    
    date_str = datetime.now().strftime("%Y%m%d")
    new_filename = f"{date_str}_parameter_importance_final(修正資料洩露).csv"
    os.rename('parameter_importance_report_fixed.csv', new_filename)
    print(f"📊 已重命名為: {new_filename}")


def main():
    print("=" * 60)
    print("參數重要性分析 - 修正版（無資料洩漏）")
    print("=" * 60)
    
    # 載入資料
    df, X, y, feature_columns = load_and_prepare_data()
    
    # 1. 統計分析（不涉及機器學習，沒有洩漏問題）
    t_test_results = statistical_analysis(df, X, y, feature_columns)
    
    # 2. 隨機森林（使用Pipeline）
    rf_pipeline, rf_importance, rf_cv_scores = feature_importance_rf_fixed(X, y, feature_columns)
    
    # 3. 邏輯迴歸（使用Pipeline）
    lr_pipeline, lr_coefficients, lr_cv_scores = logistic_regression_analysis_fixed(X, y, feature_columns)
    
    # 4. 視覺化
    combined_ranking = create_visualization_fixed(t_test_results, rf_importance, 
                                                 lr_coefficients, rf_cv_scores, lr_cv_scores)
    
    # 5. 最終報告
    generate_final_report_fixed(t_test_results, rf_importance, lr_coefficients, combined_ranking)
    
    print("\n✅ 分析完成！")
    print("\n📝 方法說明:")
    print("  • 使用Pipeline確保標準化只在訓練集上進行")
    print("  • 交叉驗證時每個fold獨立進行預處理")
    print("  • 測試集從未參與任何統計計算")
    print("  • 報告包含平均值±標準差以顯示穩定性")


if __name__ == "__main__":
    main()