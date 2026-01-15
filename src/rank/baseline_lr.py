# baseline_lr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, precision_recall_curve, classification_report
import joblib
import json
from datetime import datetime
import os

class LRModelTrainer:
    def __init__(self, data_path):
        """
        初始化LR模型训练器
        Args:
            data_path: train_samples_enriched.csv 文件路径
        """
        self.data_path = data_path
        self.df = None  # 存储加载的数据集
        self.X = None   # 特征矩阵（用户/新闻/交叉特征）
        self.y = None   # 标签（1=点击，0=未点击）
        self.feature_names = None
        self.scaler = StandardScaler()  # 特征标准化工具
        self.model = None  # 存储训练好的LR模型
        self.results = {}  # 存储评估结果
        
    def load_and_prepare_data(self, sample_fraction=0.1):
        """
        加载数据并准备训练集
        Args:
            sample_fraction: 采样比例（为避免内存问题，可以先采样）
        """
        print("步骤1: 加载数据...")
        self.df = pd.read_csv(self.data_path)
        print(f"  数据集维度: {self.df.shape}")
        print(f"  特征数量: {len(self.df.columns) - 3}")  # 减去 user_id, news_id, label
        
        # 采样（如果数据太大）
        if sample_fraction < 1.0:
            original_size = len(self.df)
            self.df = self.df.sample(frac=sample_fraction, random_state=42)
            print(f"  采样后: {len(self.df)} 行 ({sample_fraction*100:.0f}% 原始数据)")
        
        # 分离特征和标签
        self.feature_names = [col for col in self.df.columns 
                             if col not in ['user_id', 'news_id', 'label']]
        
        self.X = self.df[self.feature_names].values
        self.y = self.df['label'].values
        
        print(f"  特征矩阵: {self.X.shape}")
        print(f"  正样本比例: {self.y.mean():.4f} ({self.y.sum()}/{len(self.y)})")
        
    def preprocess_features(self):
        """特征预处理：标准化"""
        print("步骤2: 特征预处理...")
        
        # 标准化数值特征
        self.X = self.scaler.fit_transform(self.X)
        print(f"  特征标准化完成")
        
        # 检查是否有异常值
        nan_count = np.isnan(self.X).sum()
        inf_count = np.isinf(self.X).sum()
        print(f"  NaN值: {nan_count}, Inf值: {inf_count}")
        
        # 处理异常值（如果有）
        if nan_count > 0 or inf_count > 0:
            self.X = np.nan_to_num(self.X, nan=0.0, posinf=100, neginf=-100)
            print("  已处理异常值")
    
    def split_data(self, test_size=0.2):
        """划分训练集和测试集"""
        print("步骤3: 划分训练集和测试集...")
        
        # 分层抽样，保持正负样本比例
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=42,
            stratify=self.y  # 保持类别比例
        )
        
        print(f"  训练集: {X_train.shape}, 正样本: {y_train.mean():.4f}")
        print(f"  测试集: {X_test.shape}, 正样本: {y_test.mean():.4f}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """训练逻辑回归模型"""
        print("步骤4: 训练逻辑回归模型...")
        
        # 使用类别权重处理不平衡
        class_weight = 'balanced'  # 自动平衡类别权重
        
        self.model = LogisticRegression(
            penalty='l2',           # L2正则化
            C=1.0,                  # 正则化强度
            class_weight=class_weight,
            solver='lbfgs',         # 适合中小型数据集
            max_iter=1000,          # 最大迭代次数
            random_state=42,
            verbose=1               # 显示训练过程
        )
        
        self.model.fit(X_train, y_train)
        print(f"  训练完成！迭代次数: {self.model.n_iter_[0]}")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """评估模型性能"""
        print("步骤5: 评估模型...")
        
        # 预测概率
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # 计算评估指标
        auc = roc_auc_score(y_test, y_pred_proba)
        loss = log_loss(y_test, y_pred_proba)
        
        print(f"  AUC: {auc:.4f}")
        print(f"  Log Loss: {loss:.4f}")
        
        # 不同阈值下的性能
        print("\n  不同阈值下的性能:")
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        threshold_metrics = []
        
        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            
            # 计算混淆矩阵
            tp = ((y_pred == 1) & (y_test == 1)).sum()
            fp = ((y_pred == 1) & (y_test == 0)).sum()
            fn = ((y_pred == 0) & (y_test == 1)).sum()
            tn = ((y_pred == 0) & (y_test == 0)).sum()
            
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            
            threshold_metrics.append({
                'threshold': thresh,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'tn': int(tn)
            })
            
            print(f"    阈值 {thresh}: Precision={precision:.4f}, "
                  f"Recall={recall:.4f}, F1={f1:.4f}")
        
        self.results = {
            'auc': auc,
            'log_loss': loss,
            'threshold_metrics': threshold_metrics,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test
        }
        
        return self.results
    
    def analyze_features(self):
        """分析特征重要性"""
        print("步骤6: 分析特征重要性...")
        
        if self.model is None:
            print("  请先训练模型")
            return
        
        # 获取特征系数
        coefficients = self.model.coef_[0]
        
        # 创建特征重要性DataFrame
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'abs_coefficient': abs(coefficients),
            'importance_rank': range(len(self.feature_names))
        })
        
        # 按重要性排序
        feature_importance = feature_importance.sort_values(
            'abs_coefficient', ascending=False
        ).reset_index(drop=True)
        
        print("\n  特征重要性排名 (前15):")
        for i, row in feature_importance.head(15).iterrows():
            sign = '+' if row['coefficient'] > 0 else '-'
            print(f"    {i+1:2d}. {row['feature']:30s} {sign} {abs(row['coefficient']):.4f}")
        
        return feature_importance
    
    def visualize_results(self, X_test, y_test, feature_importance):
        """可视化结果"""
        print("步骤7: 生成可视化图表...")
        
        os.makedirs('../results', exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. ROC曲线
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, self.results['y_pred_proba'])
        
        axes[0, 0].plot(fpr, tpr, 'b-', linewidth=2, 
                       label=f'AUC = {self.results["auc"]:.4f}')
        axes[0, 0].plot([0, 1], [0, 1], 'r--', linewidth=1)
        axes[0, 0].set_xlabel('False Positive Rate', fontsize=12)
        axes[0, 0].set_ylabel('True Positive Rate', fontsize=12)
        axes[0, 0].set_title('ROC Curve', fontsize=14)
        axes[0, 0].legend(loc='lower right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 特征重要性
        top_features = feature_importance.head(15)
        colors = ['red' if c < 0 else 'blue' 
                 for c in top_features['coefficient']]
        
        axes[0, 1].barh(range(len(top_features)), 
                       top_features['abs_coefficient'], 
                       color=colors)
        axes[0, 1].set_yticks(range(len(top_features)))
        axes[0, 1].set_yticklabels(top_features['feature'])
        axes[0, 1].set_xlabel('Feature Importance (|coefficient|)', fontsize=12)
        axes[0, 1].set_title('Top 15 Feature Importance', fontsize=14)
        axes[0, 1].invert_yaxis()
        
        # 3. 预测概率分布
        axes[1, 0].hist(self.results['y_pred_proba'][y_test == 0], 
                       bins=50, alpha=0.5, label='Negative samples (not clicked)', density=True)
        axes[1, 0].hist(self.results['y_pred_proba'][y_test == 1], 
                       bins=50, alpha=0.5, label='Positive samples (click)', density=True)
        axes[1, 0].set_xlabel('Predicted probability', fontsize=12)
        axes[1, 0].set_ylabel('density', fontsize=12)
        axes[1, 0].set_title('Predicted probability distribution', fontsize=14)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 精确率-召回率曲线
        precision, recall, _ = precision_recall_curve(
            y_test, self.results['y_pred_proba']
        )
        axes[1, 1].plot(recall, precision, 'g-', linewidth=2)
        axes[1, 1].set_xlabel('Recall', fontsize=12)
        axes[1, 1].set_ylabel('Precision', fontsize=12)
        axes[1, 1].set_title('Precision-Recall Curve', fontsize=14)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../results/lr_model_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("  图表已保存到 ../results/lr_model_results.png")
    
    def save_model(self, model_dir='../models'):
        """保存模型和结果"""
        print("步骤8: 保存模型和结果...")
        
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存模型
        model_path = f'{model_dir}/lr_model_{timestamp}.pkl'
        joblib.dump(self.model, model_path)
        
        # 保存特征处理器
        scaler_path = f'{model_dir}/scaler_{timestamp}.pkl'
        joblib.dump(self.scaler, scaler_path)
        
        # 保存特征列表
        features_path = f'{model_dir}/features_{timestamp}.json'
        with open(features_path, 'w') as f:
            json.dump({
                'feature_names': self.feature_names,
                'feature_count': len(self.feature_names)
            }, f, indent=2)
        
        # 保存实验结果
        results_path = f'{model_dir}/results_{timestamp}.json'
        results_to_save = {
            'timestamp': timestamp,
            'model': 'LogisticRegression',
            'metrics': {
                'auc': float(self.results['auc']),
                'log_loss': float(self.results['log_loss'])
            },
            'threshold_metrics': self.results['threshold_metrics'],
            'feature_count': len(self.feature_names)
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"  模型保存到: {model_path}")
        print(f"  Scaler保存到: {scaler_path}")
        print(f"  特征列表保存到: {features_path}")
        print(f"  结果保存到: {results_path}")
        
        return model_path

def main():
    """主函数"""
    print("="*60)
    print("逻辑回归模型训练")
    print("="*60)
    
    # 1. 初始化训练器
    data_path = '../../data/processed/train_samples.csv'
    trainer = LRModelTrainer(data_path)
    
    # 2. 加载数据（先采样10%避免内存问题）
    trainer.load_and_prepare_data(sample_fraction=0.1)
    
    # 3. 预处理
    trainer.preprocess_features()
    
    # 4. 划分数据集
    X_train, X_test, y_train, y_test = trainer.split_data(test_size=0.2)
    
    # 5. 训练模型
    trainer.train_model(X_train, y_train)
    
    # 6. 评估模型
    results = trainer.evaluate_model(X_test, y_test)
    
    # 7. 分析特征
    feature_importance = trainer.analyze_features()
    
    # 8. 可视化
    trainer.visualize_results(X_test, y_test, feature_importance)
    
    # 9. 保存模型
    trainer.save_model()
    
    print("\n" + "="*60)
    print("训练完成！")
    print(f"模型AUC: {results['auc']:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()