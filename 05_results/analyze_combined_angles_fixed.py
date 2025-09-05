#!/usr/bin/env python3
"""
分析結合angle_degrees和segment_angle_max對預測導管更換的效果
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
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report, roc_curve
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def compare_models_fixed():
    """比較不同參數組合的預測效果 - 修正版"""
    
    # 載入資料
    df = pd.read_csv('vessel_morphology_14_parameters.csv')
    
    print("=" * 70)
    print("角度參數組合預測效果分析 (修正版 - 無資料洩漏)")
    print("=" * 70)
    
    # 準備不同的特徵組合
    feature_sets = {
        '1. 僅 angle_degrees': ['angle_degrees'],
        '2. 僅 segment_angle_max': ['segment_angle_max'],
        '3. 兩個角度結合': ['angle_degrees', 'segment_angle_max'],
        '4. 所有角度參數': ['angle_degrees', 'segment_angle_max', 'arch_angle', 
                         'branch_angle', 'segment_angle_mean'],
        '5. 所有14個參數': [col for col in df.columns if col not in ['filename', 'label']]
    }
    
    # 標籤
    y = df['label'].values
    
    # 交叉驗證設定
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 儲存結果
    results = []
    detailed_predictions = {}
    
    print("\n📊 模型比較（5折交叉驗證 - Pipeline防止洩漏）:")
    print("-" * 70)
    print(f"{'特徵組合':<25} {'準確率':<12} {'AUC':<12} {'敏感度':<12} {'特異度'}")
    print("-" * 70)
    
    # 測試每個特徵組合
    for name, features in feature_sets.items():
        X = df[features].values
        
        # 創建Pipeline來防止資料洩漏
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
        ])
        
        # 交叉驗證 - 使用Pipeline確保每個fold獨立進行標準化
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
        cv_auc = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')
        
        # 為了獲取敏感度和特異度，需要手動進行交叉驗證
        sensitivities = []
        specificities = []
        all_y_true = []
        all_y_pred = []
        
        for train_idx, test_idx in cv.split(X, y):
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]
            
            # 在每個fold內部獨立fit pipeline
            pipeline_fold = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
            ])
            
            pipeline_fold.fit(X_train_fold, y_train_fold)
            y_pred_fold = pipeline_fold.predict(X_test_fold)
            
            # 計算混淆矩陣
            tn, fp, fn, tp = confusion_matrix(y_test_fold, y_pred_fold).ravel()
            
            if (tp + fn) > 0:
                sensitivities.append(tp / (tp + fn))
            if (tn + fp) > 0:
                specificities.append(tn / (tn + fp))
            
            all_y_true.extend(y_test_fold)
            all_y_pred.extend(y_pred_fold)
        
        # 平均敏感度和特異度
        avg_sensitivity = np.mean(sensitivities) if sensitivities else 0
        avg_specificity = np.mean(specificities) if specificities else 0
        
        results.append({
            'name': name,
            'features': features,
            'accuracy': cv_scores.mean(),
            'accuracy_std': cv_scores.std(),
            'auc': cv_auc.mean(),
            'auc_std': cv_auc.std(),
            'sensitivity': avg_sensitivity,
            'specificity': avg_specificity,
            'y_true': all_y_true,
            'y_pred': all_y_pred
        })
        
        detailed_predictions[name] = {
            'y_true': all_y_true,
            'y_pred': all_y_pred
        }
        
        print(f"{name:<25} {cv_scores.mean():<12.3f} {cv_auc.mean():<12.3f} "
              f"{avg_sensitivity:<12.3f} {avg_specificity:.3f}")
    
    # 顯示標準差
    print("\n📊 準確率標準差（穩定性指標）:")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<25} 準確率: {r['accuracy']:.3f} ± {r['accuracy_std']:.3f}")
    
    # 創建視覺化
    create_fixed_visualizations(df, results)
    
    # 詳細分析
    print_fixed_detailed_analysis(df, results)
    
    return results


def create_fixed_visualizations(df, results):
    """創建修正版的視覺化"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. 準確率比較（含誤差條）
    ax1 = plt.subplot(2, 3, 1)
    names = [r['name'].replace('. ', '.\n') for r in results]
    accuracies = [r['accuracy'] for r in results]
    errors = [r['accuracy_std'] for r in results]
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    
    bars = ax1.bar(range(len(names)), accuracies, yerr=errors, 
                   color=colors, alpha=0.7, capsize=5)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('準確率')
    ax1.set_title('不同特徵組合的準確率 (含標準差)')
    ax1.set_ylim([0, 1])
    
    # 添加數值標籤
    for bar, acc, err in zip(bars, accuracies, errors):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 0.01,
                f'{acc:.3f}', ha='center', fontsize=9)
    
    # 2. AUC比較（含誤差條）
    ax2 = plt.subplot(2, 3, 2)
    aucs = [r['auc'] for r in results]
    auc_errors = [r['auc_std'] for r in results]
    
    bars = ax2.bar(range(len(names)), aucs, yerr=auc_errors,
                   color=colors, alpha=0.7, capsize=5)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('AUC')
    ax2.set_title('不同特徵組合的AUC (含標準差)')
    ax2.set_ylim([0, 1])
    
    # 3. 散點圖：兩個角度的原始分布
    ax3 = plt.subplot(2, 3, 3)
    for label in [0, 1]:
        mask = df['label'] == label
        label_text = "不需更換" if label == 0 else "需要更換"
        ax3.scatter(df[mask]['angle_degrees'], 
                   df[mask]['segment_angle_max'],
                   alpha=0.6, s=50, label=label_text)
    
    ax3.set_xlabel('angle_degrees')
    ax3.set_ylabel('segment_angle_max')
    ax3.set_title('兩個角度參數的分布')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 敏感度和特異度
    ax4 = plt.subplot(2, 3, 4)
    x = np.arange(len(names))
    width = 0.35
    
    sensitivities = [r['sensitivity'] for r in results]
    specificities = [r['specificity'] for r in results]
    
    ax4.bar(x - width/2, sensitivities, width, label='敏感度', alpha=0.7, color='blue')
    ax4.bar(x + width/2, specificities, width, label='特異度', alpha=0.7, color='orange')
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel('比率')
    ax4.set_title('敏感度 vs 特異度')
    ax4.legend()
    ax4.set_ylim([0, 1])
    
    # 5. 混淆矩陣熱圖（最佳模型）
    ax5 = plt.subplot(2, 3, 5)
    best_idx = np.argmax([r['accuracy'] for r in results])
    best_result = results[best_idx]
    
    cm = confusion_matrix(best_result['y_true'], best_result['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
                xticklabels=['不需更換', '需更換'],
                yticklabels=['不需更換', '需更換'])
    ax5.set_title(f'最佳模型混淆矩陣\n({best_result["name"]})')
    ax5.set_ylabel('真實值')
    ax5.set_xlabel('預測值')
    
    # 6. 改善幅度比較
    ax6 = plt.subplot(2, 3, 6)
    base_accuracy = results[0]['accuracy']  # 僅angle_degrees
    improvements = [(r['accuracy'] - base_accuracy) * 100 for r in results[1:]]
    names_short = [r['name'].split('. ')[1] for r in results[1:]]
    
    colors_imp = ['red' if imp < 0 else 'green' for imp in improvements]
    bars = ax6.barh(range(len(improvements)), improvements, color=colors_imp, alpha=0.7)
    ax6.set_yticks(range(len(improvements)))
    ax6.set_yticklabels(names_short, fontsize=9)
    ax6.set_xlabel('準確率改善 (%)')
    ax6.set_title(f'相對於僅用angle_degrees的改善')
    ax6.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    plt.suptitle('角度參數組合效果分析 (修正版 - 無資料洩漏)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('combined_angles_analysis_fixed.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ 視覺化已儲存至: combined_angles_analysis_fixed.png")


def print_fixed_detailed_analysis(df, results):
    """輸出修正版的詳細分析報告"""
    
    print("\n" + "=" * 70)
    print("📋 詳細分析報告 (修正版)")
    print("=" * 70)
    
    # 比較單獨使用vs結合使用
    acc_angle_only = results[0]['accuracy']
    acc_segment_only = results[1]['accuracy']
    acc_combined = results[2]['accuracy']
    
    print("\n🔍 關鍵發現:")
    print("-" * 40)
    
    print(f"\n1. 單獨使用 vs 結合使用:")
    print(f"   • 僅 angle_degrees: {acc_angle_only:.3f} ± {results[0]['accuracy_std']:.3f}")
    print(f"   • 僅 segment_angle_max: {acc_segment_only:.3f} ± {results[1]['accuracy_std']:.3f}")
    print(f"   • 兩者結合: {acc_combined:.3f} ± {results[2]['accuracy_std']:.3f}")
    
    improvement = (acc_combined - max(acc_angle_only, acc_segment_only)) * 100
    if improvement > 0:
        print(f"   ✅ 結合使用提升了 {improvement:.1f}% 的準確率")
    else:
        print(f"   ⚠️ 結合使用沒有顯著提升 ({improvement:+.1f}%)")
    
    print(f"\n2. 最佳特徵組合:")
    best_result = max(results, key=lambda x: x['accuracy'])
    print(f"   • 名稱: {best_result['name']}")
    print(f"   • 準確率: {best_result['accuracy']:.3f} ± {best_result['accuracy_std']:.3f}")
    print(f"   • AUC: {best_result['auc']:.3f} ± {best_result['auc_std']:.3f}")
    print(f"   • 特徵數: {len(best_result['features'])}")
    
    # 保存結果
    results_df = pd.DataFrame([{
        'Model': r['name'],
        'Accuracy': f"{r['accuracy']:.3f} ± {r['accuracy_std']:.3f}",
        'AUC': f"{r['auc']:.3f} ± {r['auc_std']:.3f}",
        'Sensitivity': f"{r['sensitivity']:.3f}",
        'Specificity': f"{r['specificity']:.3f}"
    } for r in results])
    
    results_df.to_csv('angle_combination_results_fixed.csv', index=False)
    print(f"\n📊 結果已儲存至: angle_combination_results_fixed.csv")
    
    print("\n📝 方法說明:")
    print("-" * 40)
    print("• 使用Pipeline確保每個CV fold內部獨立進行標準化")
    print("• 避免在整個資料集上先進行標準化造成資料洩漏")
    print("• 報告準確率和AUC的平均值±標準差")
    print("• 確保測試集從未參與任何預處理步驟")


if __name__ == "__main__":
    results = compare_models_fixed()