# deepfm_baseline_modular.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, roc_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import joblib
import json
from datetime import datetime
import os
import argparse

print("="*60)
print("DeepFM CTR预估模型训练")
print("="*60)

# ==================== 配置模块 ====================
def get_config(args):  # 添加args参数
    """返回配置参数"""
    config = {
        'data': {
            'path': '../../data/processed/train_deepfm_samples.csv',
            'sparse_features': [
                'active_hour',          # 0-23，24个类别
                'is_weekend',          # 0/1，2个类别  
                'has_abstract',        # 0/1，2个类别
                'subcategory_encoded', # 子类别编码
                'category_encoded'     # 主类别编码
            ],
            'dense_features': [
                'hist_click_count',    # 用户历史点击
                'hist_ctr',           # 用户历史CTR
                'title_length',       # 标题长度
                'impression_count',   # 新闻曝光
                'click_count',        # 新闻点击
                'ctr'                # 新闻CTR
            ],
            'sample_fraction': args.sample_fraction if args else 1.0  #支持命令行参数
        },
        'model': {
            'embedding_dim': args.embedding_dim if args else 10,
            'hidden_dims': [128, 64, 32],
            'dropout': 0.2,
            'learning_rate': 0.001,
            'weight_decay': 0.0001
        },
        'training': {
            'batch_size': 1024,
            'epochs': args.epochs if args else 30,
            'test_size': 0.2,
            'early_stop_patience': 5
        },
        'paths': {
            'results_dir': '../results',
            'models_dir': '../models'
        }
    }
    return config

# ==================== 数据加载模块 ====================
def load_and_prepare_data(config):
    """加载和准备数据"""
    print("\n[1] 数据加载与准备")
    print("-" * 40)
    
    # 加载数据
    df = pd.read_csv(config['data']['path'])
    if config['data']['sample_fraction'] < 1.0:
      df = df.sample(frac=config['data']['sample_fraction'], random_state=42)  # ← 只用10%数据
    print(f"  数据集: {df.shape[0]} 样本, {df.shape[1]} 特征")
    print(f"  正样本比例: {df['label'].mean():.4f}")
    
    # 提取特征
    sparse_features = config['data']['sparse_features']
    dense_features = config['data']['dense_features']
    
    sparse_data = df[sparse_features].values.astype(np.int32)
    dense_data = df[dense_features].values.astype(np.float32)
    labels = df['label'].values.astype(np.float32)
    
    print(f"  稀疏特征: {len(sparse_features)} 个, 形状: {sparse_data.shape}")
    print(f"  稠密特征: {len(dense_features)} 个, 形状: {dense_data.shape}")
    
    return sparse_data, dense_data, labels, sparse_features, dense_features
# ==================== 数据处理模块 ====================
def preprocess_data(sparse_data, dense_data, labels, config):
    """数据预处理：标准化和划分数据集"""
    print("\n[2] 数据预处理")
    print("-" * 40)
    
    # 标准化稠密特征
    scaler = StandardScaler()
    dense_data_scaled = scaler.fit_transform(dense_data)
    print(f"  稠密特征标准化完成")
    
    # 划分数据集
    test_size = config['training']['test_size']
    (sparse_train, sparse_test,
     dense_train, dense_test,
     labels_train, labels_test) = train_test_split(
        sparse_data, dense_data_scaled, labels,
        test_size=test_size,
        random_state=42,
        stratify=labels
    )
    
    print(f"  训练集: {len(labels_train)} 样本 ({1-test_size:.0%})")
    print(f"  测试集: {len(labels_test)} 样本 ({test_size:.0%})")
    print(f"  正样本比例 - 训练: {labels_train.mean():.4f}, 测试: {labels_test.mean():.4f}")
    
    return (sparse_train, dense_train, labels_train,
            sparse_test, dense_test, labels_test, scaler)
# ==================== 数据转换模块 ====================
def create_dataloaders(sparse_train, dense_train, labels_train,
                       sparse_test, dense_test, labels_test, config):
    """创建PyTorch DataLoader"""
    print("\n[3] 创建数据加载器")
    print("-" * 40)
    
    # 转换为张量
    sparse_train_tensor = torch.LongTensor(sparse_train)
    dense_train_tensor = torch.FloatTensor(dense_train)
    labels_train_tensor = torch.FloatTensor(labels_train).view(-1, 1)
    
    sparse_test_tensor = torch.LongTensor(sparse_test)
    dense_test_tensor = torch.FloatTensor(dense_test)
    labels_test_tensor = torch.FloatTensor(labels_test).view(-1, 1)
    
    # 创建Dataset
    train_dataset = TensorDataset(sparse_train_tensor, dense_train_tensor, labels_train_tensor)
    test_dataset = TensorDataset(sparse_test_tensor, dense_test_tensor, labels_test_tensor)
    
    # 创建DataLoader
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  批次大小: {batch_size}")
    print(f"  训练批次数: {len(train_loader)}")
    print(f"  测试批次数: {len(test_loader)}")
    
    return train_loader, test_loader
# ==================== 模型定义模块 ====================
class DeepFM(nn.Module):
    """DeepFM模型定义"""
    
    def __init__(self, sparse_feature_dims, dense_feature_dim, config):
        super(DeepFM, self).__init__()
        
        self.embedding_dim = config['model']['embedding_dim']
        hidden_dims = config['model']['hidden_dims']
        dropout = config['model']['dropout']
        
        print(f"  稀疏特征维度: {sparse_feature_dims}")
        print(f"  稠密特征维度: {dense_feature_dim}")
        print(f"  Embedding维度: {self.embedding_dim}")
        print(f"  DNN结构: {hidden_dims}")
        
        # 1. FM一阶部分
        self.fm_first_order = nn.ModuleList()
        for dim in sparse_feature_dims:
          self.fm_first_order.append(nn.Embedding(dim, 1)) # nn.Embedding(10000,1)：造一本有 10000 个单词的字典，每个单词对应 1 个数字。【创建了多个 Embedding 层（多本字典），每个层对应一个稀疏特征维度（用户 ID / 类别 / 子类别）】
        
        self.fm_first_order_dense = nn.Linear(dense_feature_dim, 1)
        
        # 2. FM二阶部分
        self.sparse_embeddings = nn.ModuleList()
        for dim in sparse_feature_dims:
            self.sparse_embeddings.append(nn.Embedding(dim, self.embedding_dim))
        
        # 3. DNN部分
        dnn_input_dim = len(sparse_feature_dims) * self.embedding_dim + dense_feature_dim
        
        layers = []
        prev_dim = dnn_input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.dnn = nn.Sequential(*layers)
        
        # 4. 偏置
        self.bias = nn.Parameter(torch.zeros(1))
        
        # 初始化权重
        self._init_weights()
        
        print(f"  总参数量: {self.count_parameters():,}")
    
    def _init_weights(self):
        """初始化权重"""
        for embed in self.fm_first_order:
            nn.init.normal_(embed.weight, std=0.01)
        
        for embed in self.sparse_embeddings:
            nn.init.normal_(embed.weight, std=0.01)
        
        for layer in self.dnn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
        nn.init.normal_(self.fm_first_order_dense.weight, std=0.01)
        nn.init.constant_(self.fm_first_order_dense.bias, 0)
    
    def count_parameters(self):
        """计算参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, sparse_features, dense_features):
        batch_size = sparse_features.size(0)
        
        # 1. FM一阶
        fm_first_order = 0
        for i in range(len(self.fm_first_order)):
            fm_first_order += self.fm_first_order[i](sparse_features[:, i]) # 用第 i 个稀疏特征的 Embedding 层，查所有样本的第 i 个稀疏特征值对应的一阶权重”
        fm_first_order = fm_first_order.squeeze()
        fm_first_order += self.fm_first_order_dense(dense_features).squeeze()
        
        # 2. FM二阶
        embeddings = []
        for i in range(len(self.sparse_embeddings)):
            emb = self.sparse_embeddings[i](sparse_features[:, i])
            embeddings.append(emb.unsqueeze(1))
        
        all_embeddings = torch.cat(embeddings, dim=1)
        
        sum_emb = torch.sum(all_embeddings, dim=1)
        sum_emb_square = torch.pow(sum_emb, 2)
        square_emb = torch.pow(all_embeddings, 2)
        square_sum_emb = torch.sum(square_emb, dim=1)
        
        fm_second_order = 0.5 * torch.sum(sum_emb_square - square_sum_emb, dim=1)
        
        # 3. DNN
        dnn_input = torch.cat([
            all_embeddings.view(batch_size, -1),
            dense_features
        ], dim=1)
        
        dnn_output = self.dnn(dnn_input).squeeze()
        
        # 4. 最终输出
        output = fm_first_order + fm_second_order + dnn_output + self.bias
        output = torch.sigmoid(output)
        
        return output
# ==================== 训练模块 ====================
def train_model(model, train_loader, test_loader, config, device):
    """训练模型"""
    print("\n[4] 模型训练")
    print("-" * 40)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['model']['learning_rate'],
        weight_decay=config['model']['weight_decay']
    )
    
    epochs = config['training']['epochs']
    patience = config['training']['early_stop_patience']
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_auc': [], 'val_auc': []
    }
    
    best_auc = 0
    patience_counter = 0
    best_model_state = None
    
    print(f"  设备: {device}")
    print(f"  训练周期: {epochs}")
    print(f"  早停耐心值: {patience}")
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss, train_predictions, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        train_auc = roc_auc_score(train_labels, train_predictions)
        
        # 验证阶段
        model.eval()
        val_predictions, val_labels = validate_model(model, test_loader, device)
        val_auc = roc_auc_score(val_labels, val_predictions)
        val_loss = log_loss(val_labels, val_predictions)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        
        # 打印进度
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}, AUC: {train_auc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")
        
        # 保存最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  早停触发，最佳验证AUC: {best_auc:.4f}")
                break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"  训练完成，最佳验证AUC: {best_auc:.4f}")
    
    return model, history, best_auc

def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个周期"""
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    for sparse_batch, dense_batch, label_batch in train_loader:
        sparse_batch = sparse_batch.to(device)
        dense_batch = dense_batch.to(device)
        label_batch = label_batch.to(device).squeeze()
        
        # 前向传播
        predictions = model(sparse_batch, dense_batch)
        loss = criterion(predictions, label_batch)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        all_predictions.append(predictions.detach().cpu().numpy())
        all_labels.append(label_batch.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    
    return avg_loss, all_predictions, all_labels

def validate_model(model, test_loader, device):
    """验证模型"""
    all_predictions = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for sparse_batch, dense_batch, label_batch in test_loader:
            sparse_batch = sparse_batch.to(device)
            dense_batch = dense_batch.to(device)
            label_batch = label_batch.to(device).squeeze()
            
            predictions = model(sparse_batch, dense_batch)
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(label_batch.cpu().numpy())
    
    return np.concatenate(all_predictions), np.concatenate(all_labels)
# ==================== 评估模块 ====================
def evaluate_model(model, test_loader, device, history):
    """评估模型性能"""
    print("\n[5] 模型评估")
    print("-" * 40)
    
    # 获取预测
    predictions, labels = validate_model(model, test_loader, device)
    
    # 计算指标
    auc = roc_auc_score(labels, predictions)
    loss = log_loss(labels, predictions)
    
    print(f"  测试集 AUC: {auc:.4f}")
    print(f"  测试集 Log Loss: {loss:.4f}")
    
    # 不同阈值下的性能
    print(f"\n  不同阈值下的性能:")
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    metrics_by_threshold = []
    
    for thresh in thresholds:
        pred_labels = (predictions >= thresh).astype(int)
        
        tp = np.sum((pred_labels == 1) & (labels == 1))
        fp = np.sum((pred_labels == 1) & (labels == 0))
        fn = np.sum((pred_labels == 0) & (labels == 1))
        tn = np.sum((pred_labels == 0) & (labels == 0))
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        accuracy = (tp + tn) / len(labels)
        
        metrics_by_threshold.append({
            'threshold': thresh,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        })
        
        print(f"    阈值 {thresh:.1f}: "
              f"Precision={precision:.4f}, "
              f"Recall={recall:.4f}, "
              f"F1={f1:.4f}, "
              f"Accuracy={accuracy:.4f}")
    
    return {
        'auc': auc,
        'log_loss': loss,
        'predictions': predictions,
        'labels': labels,
        'metrics_by_threshold': metrics_by_threshold,
        'history': history
    }

# ==================== 可视化模块 ====================
def visualize_results(results, config):
    """可视化结果"""
    print("\n[6] 生成可视化图表")
    print("-" * 40)
    
    os.makedirs(config['paths']['results_dir'], exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    history = results['history']
    
    # 1. 损失曲线
    axes[0, 0].plot(history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    axes[0, 0].plot(history['val_loss'], 'r-', linewidth=2, label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('训练和验证损失曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. AUC曲线
    axes[0, 1].plot(history['train_auc'], 'b-', linewidth=2, label='Train AUC')
    axes[0, 1].plot(history['val_auc'], 'r-', linewidth=2, label='Val AUC')
    axes[0, 1].axhline(y=0.72, color='g', linestyle='--', alpha=0.7, label='目标AUC=0.72')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].set_title('训练和验证AUC曲线')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 预测概率分布
    axes[1, 0].hist(results['predictions'][results['labels'] == 0], 
                   bins=50, alpha=0.5, label='负样本 (未点击)', density=True)
    axes[1, 0].hist(results['predictions'][results['labels'] == 1], 
                   bins=50, alpha=0.5, label='正样本 (点击)', density=True)
    axes[1, 0].set_xlabel('预测概率')
    axes[1, 0].set_ylabel('密度')
    axes[1, 0].set_title('预测概率分布')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. ROC曲线
    fpr, tpr, _ = roc_curve(results['labels'], results['predictions'])
    axes[1, 1].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {results["auc"]:.4f}')
    axes[1, 1].plot([0, 1], [0, 1], 'r--', linewidth=1, label='随机猜测')
    axes[1, 1].set_xlabel('假正率 (False Positive Rate)')
    axes[1, 1].set_ylabel('真正率 (True Positive Rate)')
    axes[1, 1].set_title('ROC曲线')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f"{config['paths']['results_dir']}/deepfm_results.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"  图表已保存到: {save_path}")

# ==================== 保存模块 ====================
def save_results(model, scaler, results, sparse_features, dense_features, 
                 sparse_feature_dims, config):
    """保存模型和结果"""
    print("\n[7] 保存模型和结果")
    print("-" * 40)
    
    os.makedirs(config['paths']['models_dir'], exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存模型
    model_path = f"{config['paths']['models_dir']}/deepfm_model_{timestamp}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'sparse_feature_dims': sparse_feature_dims,
        'dense_feature_dim': len(dense_features),
        'sparse_features': sparse_features,
        'dense_features': dense_features,
        'config': config
    }, model_path)
    
    # 保存特征处理器
    scaler_path = f"{config['paths']['models_dir']}/deepfm_scaler_{timestamp}.pkl"
    joblib.dump(scaler, scaler_path)
    
    # 保存实验结果
    results_path = f"{config['paths']['models_dir']}/deepfm_results_{timestamp}.json"
    results_to_save = {
        'timestamp': timestamp,
        'model': 'DeepFM',
        'metrics': {
            'auc': float(results['auc']),
            'log_loss': float(results['log_loss'])
        },
        'training_history': {
            'train_losses': [float(x) for x in results['history']['train_loss']],
            'train_aucs': [float(x) for x in results['history']['train_auc']],
            'val_losses': [float(x) for x in results['history']['val_loss']],
            'val_aucs': [float(x) for x in results['history']['val_auc']]
        },
        'feature_info': {
            'sparse_features': sparse_features,
            'dense_features': dense_features,
            'sparse_feature_dims': sparse_feature_dims
        }
    }
    
    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"  模型保存到: {model_path}")
    print(f"  特征处理器保存到: {scaler_path}")
    print(f"  实验结果保存到: {results_path}")

# ==================== 对比分析模块 ====================
def compare_with_baseline(results):
    """与LR基线模型对比"""
    print("\n" + "="*60)
    print("对比分析")
    print("="*60)
    
    deepfm_auc = results['auc']
    lr_auc = 0.7341  # 你的LR基线结果
    
    print(f"\n模型对比:")
    print(f"  LR模型 (有人工交叉特征): AUC ≈ {lr_auc:.4f}")
    print(f"  DeepFM模型 (无人工交叉特征): AUC = {deepfm_auc:.4f}")
    
    improvement = (deepfm_auc - lr_auc) / lr_auc * 100
    
    if deepfm_auc > lr_auc + 0.03:
        print(f"  ✅ DeepFM提升了 {improvement:.1f}%，证明自动特征交叉有效！")
        print(f"  结论: DeepFM的自动特征学习能力可以替代人工特征工程")
    elif deepfm_auc > lr_auc:
        print(f"  ⚠️  DeepFM有轻微提升 ({improvement:.1f}%)")
        print(f"  结论: DeepFM能学到一些交叉，但可能还需要优化")
    else:
        print(f"  ❌ DeepFM效果不如LR (下降 {abs(improvement):.1f}%)")
        print(f"  可能原因:")
        print(f"  1. 训练时间或数据量不足")
        print(f"  2. 模型参数需要调整")
        print(f"  3. 特征处理需要优化")
    
    print(f"\n项目目标验收:")
    print(f"  目标AUC: > 0.72")
    print(f"  当前AUC: {deepfm_auc:.4f}")
    print(f"  达成情况: {'✅ 完成目标' if deepfm_auc > 0.72 else '❌ 未达目标'}")
    
    return improvement

# ==================== 主函数 ====================
def main(args=None):
    """主函数 - 协调所有模块"""
    
    # 1. 获取配置
    config = get_config(args)
    
    # 2. 加载和准备数据
    sparse_data, dense_data, labels, sparse_features, dense_features = load_and_prepare_data(config)
    
    # 3. 预处理数据
    (sparse_train, dense_train, labels_train,
     sparse_test, dense_test, labels_test, scaler) = preprocess_data(
        sparse_data, dense_data, labels, config
    )
    
    # 4. 创建数据加载器
    train_loader, test_loader = create_dataloaders(
        sparse_train, dense_train, labels_train,
        sparse_test, dense_test, labels_test, config
    )
    
    # 5. 计算稀疏特征维度
    sparse_feature_dims = []
    for col in sparse_features:
        max_val = sparse_data[:, sparse_features.index(col)].max()
        sparse_feature_dims.append(int(max_val) + 1)
    
    # 6. 创建模型
    print("\n[创建DeepFM模型]")
    print("-" * 40)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DeepFM(
        sparse_feature_dims=sparse_feature_dims,
        dense_feature_dim=len(dense_features),
        config=config
    ).to(device)
    
    # 7. 训练模型
    model, history, best_auc = train_model(model, train_loader, test_loader, config, device)
    
    # 8. 评估模型
    results = evaluate_model(model, test_loader, device, history)
    
    # 9. 可视化结果
    visualize_results(results, config)
    
    # 10. 保存结果
    save_results(model, scaler, results, sparse_features, dense_features, 
                 sparse_feature_dims, config)
    
    # 11. 对比分析
    improvement = compare_with_baseline(results)
    
    print("\n" + "="*60)
    print("DeepFM训练流程完成！")
    print("="*60)
    
    return model, results, improvement

if __name__ == "__main__":
  # 新增：创建命令行参数解析器
  parser = argparse.ArgumentParser(description='DeepFM模型训练')
  parser.add_argument('--embedding_dim', type=int, default=10, help='Embedding维度大小')
  parser.add_argument('--sample_fraction', type=float, default=1.0, help='数据采样比例')
  parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
  
  # 解析命令行参数
  args = parser.parse_args()
  
  # 打印命令行参数信息
  print(f"\n命令行参数配置:")
  print(f"  embedding_dim: {args.embedding_dim}")
  print(f"  sample_fraction: {args.sample_fraction}")
  print(f"  epochs: {args.epochs}")
  print("-" * 60)
  model, results, improvement = main(args)
