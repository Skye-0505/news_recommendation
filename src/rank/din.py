import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, log_loss, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
import os
import argparse
import ast

# ==================== 路径配置 ====================
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, '../../data/processed/train_din_samples.csv')
mapping_path = os.path.join(current_dir, '../../data/processed/mapping_info.pkl')
models_dir = os.path.join(current_dir, '../models')
results_dir = os.path.join(current_dir, '../results')

# 创建目录
os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# ==================== 模型超参数 ====================
USER_EMB_DIM = 64
NEWS_EMB_DIM = 64
CAT_EMB_DIM = 32
HIDDEN_DIMS = [512, 256, 128]
DROPOUT = 0.2
BATCH_SIZE = 512
EPOCHS = 30
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
EARLY_STOP_PATIENCE = 5
VAL_RATIO = 0.2
MAX_SEQ_LEN = 20

# ==================== Activation Unit ====================
class ActivationUnit(nn.Module):
    """DIN核心：计算历史行为和候选新闻的匹配得分"""
    def __init__(self, embedding_dim):
        super().__init__()
        input_dim = embedding_dim * 3
        self.dnn = nn.Sequential(
            nn.Linear(input_dim, 80),
            nn.ReLU(),
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, 1)
        )
        
    def forward(self, history_emb, candidate_emb):
        candidate_emb = candidate_emb.unsqueeze(1).expand_as(history_emb)
        product = history_emb * candidate_emb
        input = torch.cat([history_emb, candidate_emb, product], dim=-1)
        score = self.dnn(input)
        return score.squeeze(-1)

# ==================== DIN模型 ====================
class DIN(nn.Module):
    def __init__(self, num_users, num_news, num_categories, num_dense):
        super().__init__()
        
        # Embedding层
        self.user_embedding = nn.Embedding(num_users, USER_EMB_DIM, padding_idx=0)
        self.news_embedding = nn.Embedding(num_news, NEWS_EMB_DIM, padding_idx=0)
        self.cat_embedding = nn.Embedding(num_categories, CAT_EMB_DIM, padding_idx=0)
        
        # 激活单元
        self.activation_unit = ActivationUnit(NEWS_EMB_DIM)
        
        # DNN层
        input_dim = USER_EMB_DIM + NEWS_EMB_DIM + NEWS_EMB_DIM + num_dense
        layers = []
        prev_dim = input_dim
        for hidden_dim in HIDDEN_DIMS:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(DROPOUT))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.dnn = nn.Sequential(*layers)
        
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                
    def forward(self, user_idx, candidate_idx, history_idx, mask, dense):
        # Embedding
        user_emb = self.user_embedding(user_idx)
        cand_emb = self.news_embedding(candidate_idx)
        history_emb = self.news_embedding(history_idx)
        
        # 注意力权重
        attn_weights = self.activation_unit(history_emb, cand_emb)
        attn_weights = attn_weights.masked_fill(~mask, -1e9)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # 用户兴趣表示
        interest_emb = torch.sum(attn_weights.unsqueeze(-1) * history_emb, dim=1)
        
        # 拼接
        combined = torch.cat([user_emb, cand_emb, interest_emb, dense], dim=1)
        
        # 返回logits，不要sigmoid
        logits = self.dnn(combined).squeeze()
        return logits

# ==================== Dataset ====================
class DINDataset(Dataset):
    def __init__(self, df, dense_cols, max_seq_len=20):
        self.df = df.reset_index(drop=True)
        self.dense_cols = dense_cols
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 用户和候选索引
        user_idx = int(row['user_idx'])
        candidate_idx = int(row['candidate_news_idx'])
        
        # 历史序列
        hist_str = row['history_seq_idx']
        if isinstance(hist_str, str):
            if hist_str.startswith('['):
                history_idx = ast.literal_eval(hist_str)
            else:
                history_idx = [int(x) for x in hist_str.split(',') if x.strip()]
        else:
            history_idx = []
        
        # Padding到固定长度
        if len(history_idx) > self.max_seq_len:
            history_idx = history_idx[-self.max_seq_len:]
        pad_len = self.max_seq_len - len(history_idx)
        history_idx = [0] * pad_len + history_idx
        mask = [0] * pad_len + [1] * (self.max_seq_len - pad_len)
        
        # 数值特征
        dense = torch.FloatTensor([row[col] for col in self.dense_cols])
        
        # 标签
        label = torch.FloatTensor([row['label']])
        
        return {
            'user_idx': torch.LongTensor([user_idx]),
            'candidate_idx': torch.LongTensor([candidate_idx]),
            'history_idx': torch.LongTensor(history_idx),
            'mask': torch.BoolTensor(mask),
            'dense': dense,
            'label': label
        }

# ==================== 训练器 ====================
class Trainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 动态计算正负样本权重（从训练数据中统计）
        pos_count = sum(batch['label'].sum().item() for batch in train_loader)
        neg_count = len(train_loader)*BATCH_SIZE - pos_count
        pos_weight = torch.tensor([neg_count / pos_count]).to(device) if pos_count > 0 else torch.tensor([1.0]).to(device)
        
        # 使用带权重的损失
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        
        # 训练历史记录
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_auc': [], 'val_auc': []
        }
    
    def train_epoch(self):
        """训练单个epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in self.train_loader:
            user_idx = batch['user_idx'].to(self.device).squeeze()
            candidate_idx = batch['candidate_idx'].to(self.device).squeeze()
            history_idx = batch['history_idx'].to(self.device)
            mask = batch['mask'].to(self.device)
            dense = batch['dense'].to(self.device)
            label = batch['label'].to(self.device).squeeze()
            
            # 前向传播（模型返回logits）
            logits = self.model(user_idx, candidate_idx, history_idx, mask, dense)
            
            # 计算损失
            loss = self.criterion(logits, label)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 保存预测概率（用于计算AUC）
            pred = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.extend(pred)
            all_labels.extend(label.cpu().numpy())
        
        avg_loss = total_loss / len(self.train_loader)
        auc = roc_auc_score(all_labels, all_preds)
        
        return avg_loss, auc
    
    def validate(self):
        """验证单个epoch"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                user_idx = batch['user_idx'].to(self.device).squeeze()
                candidate_idx = batch['candidate_idx'].to(self.device).squeeze()
                history_idx = batch['history_idx'].to(self.device)
                mask = batch['mask'].to(self.device)
                dense = batch['dense'].to(self.device)
                label = batch['label'].to(self.device).squeeze()
                
                logits = self.model(user_idx, candidate_idx, history_idx, mask, dense)
                loss = self.criterion(logits, label)
                
                total_loss += loss.item()
                
                pred = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(pred)
                all_labels.extend(label.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        auc = roc_auc_score(all_labels, all_preds)
        
        return avg_loss, auc
    
    def train(self, epochs, early_stop_patience=EARLY_STOP_PATIENCE):
        """完整的多轮训练逻辑，带早停"""
        print(f"\n开始训练，设备: {self.device}")
        print(f"训练集批次: {len(self.train_loader)}, 验证集批次: {len(self.val_loader)}")
        print(f"正负样本权重: {self.criterion.pos_weight.item():.2f}")
        
        best_auc = 0.0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # 训练
            train_loss, train_auc = self.train_epoch()
            
            # 验证
            val_loss, val_auc = self.validate()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_auc'].append(train_auc)
            self.history['val_loss'].append(val_loss)
            self.history['val_auc'].append(val_auc)
            
            # 打印每个epoch的结果（每轮都打印，也可以改成每5轮）
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}, AUC: {train_auc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")
            
            # 早停逻辑
            if val_auc > best_auc:
                best_auc = val_auc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f"  → 验证AUC提升至 {best_auc:.4f}，保存最佳模型")
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"\n早停触发！最佳验证AUC: {best_auc:.4f} (Epoch {epoch+1-early_stop_patience})")
                    break
        
        # 加载最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        print(f"\n训练完成！最佳验证AUC: {best_auc:.4f}")
        return best_auc

# ==================== 评估和可视化 ====================
def evaluate_and_visualize(model, test_loader, device, history, timestamp):
    """评估模型并保存可视化结果"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            user_idx = batch['user_idx'].to(device).squeeze()
            candidate_idx = batch['candidate_idx'].to(device).squeeze()
            history_idx = batch['history_idx'].to(device)
            mask = batch['mask'].to(device)
            dense = batch['dense'].to(device)
            label = batch['label'].to(device).squeeze()
            
            logits = model(user_idx, candidate_idx, history_idx, mask, dense)
            pred = torch.sigmoid(logits)

            all_preds.append(pred.cpu().numpy())
            all_labels.append(label.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # 计算指标
    auc = roc_auc_score(all_labels, all_preds)
    logloss = log_loss(all_labels, all_preds)
    
    print(f"\n测试集结果:")
    print(f"  AUC: {auc:.4f}")
    print(f"  LogLoss: {logloss:.4f}")
    
    # ========== 保存可视化图片到results目录 ==========
    try:
        # 1. 损失曲线和AUC曲线
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 损失曲线
        axes[0, 0].plot(history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('训练和验证损失曲线')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # AUC曲线
        axes[0, 1].plot(history['train_auc'], 'b-', label='Train AUC')
        axes[0, 1].plot(history['val_auc'], 'r-', label='Val AUC')
        axes[0, 1].axhline(y=0.72, color='g', linestyle='--', alpha=0.7, label='目标AUC=0.72')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].set_title('训练和验证AUC曲线')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 预测概率分布
        axes[1, 0].hist(all_preds[all_labels == 0], bins=50, alpha=0.5, label='负样本', density=True)
        axes[1, 0].hist(all_preds[all_labels == 1], bins=50, alpha=0.5, label='正样本', density=True)
        axes[1, 0].set_xlabel('预测概率')
        axes[1, 0].set_ylabel('密度')
        axes[1, 0].set_title('预测概率分布')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # ROC曲线
        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        axes[1, 1].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc:.4f}')
        axes[1, 1].plot([0, 1], [0, 1], 'r--', label='随机猜测')
        axes[1, 1].set_xlabel('假正率')
        axes[1, 1].set_ylabel('真正率')
        axes[1, 1].set_title('ROC曲线')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = os.path.join(results_dir, f'din_results_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  结果图保存到: {plot_path}")
        
        # 2. 注意力权重可视化（采样几个样本）
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        model.eval()
        sample_count = 0
        with torch.no_grad():
            for batch in test_loader:
                if sample_count >= 6:
                    break
                    
                user_idx = batch['user_idx'].to(device).squeeze()
                candidate_idx = batch['candidate_idx'].to(device).squeeze()
                history_idx = batch['history_idx'].to(device)
                mask = batch['mask'].to(device)
                
                # 计算注意力权重
                cand_emb = model.news_embedding(candidate_idx)
                history_emb = model.news_embedding(history_idx)
                attn_weights = model.activation_unit(history_emb, cand_emb)
                attn_weights = attn_weights.masked_fill(~mask, -1e9)
                attn_weights = F.softmax(attn_weights, dim=1)
                
                attn_weights = attn_weights.squeeze().cpu().numpy()
                mask_np = mask.squeeze().cpu().numpy()
                
                # 只显示有效历史
                valid_indices = np.where(mask_np)[0]
                valid_weights = attn_weights[valid_indices]
                
                if len(valid_weights) > 0:
                    axes[sample_count].bar(range(len(valid_weights)), valid_weights)
                    axes[sample_count].set_title(f'样本 {sample_count+1}: 候选新闻索引 {candidate_idx.item()}')
                    axes[sample_count].set_xlabel('历史新闻位置')
                    axes[sample_count].set_ylabel('注意力权重')
                    sample_count += 1
        
        # 填充空的子图
        for i in range(sample_count, 6):
            axes[i].axis('off')
            axes[i].set_title('无有效历史序列')
        
        plt.tight_layout()
        
        # 保存注意力可视化
        attn_path = os.path.join(results_dir, f'din_attention_{timestamp}.png')
        plt.savefig(attn_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  注意力可视化保存到: {attn_path}")
    except Exception as e:
        print(f"⚠️ 可视化出错: {e}")
    
    return {
        'auc': auc,
        'log_loss': logloss,
        'predictions': all_preds,
        'labels': all_labels
    }

# ==================== 主函数 ====================
def main():
    print("="*60)
    print("DIN CTR预估模型训练")
    print("="*60)
    
    # 1. 加载数据
    print(f"\n[1] 加载数据")
    print("-"*40)
    print(f"数据文件: {csv_path}")
    print(f"映射文件: {mapping_path}")
    
    df = pd.read_csv(csv_path)
    print(f"  数据集: {df.shape[0]} 样本, {df.shape[1]} 特征")
    print(f"  正样本比例: {df['label'].mean():.4f}")
    
    # 2. 加载映射信息
    with open(mapping_path, 'rb') as f:
        mapping_info = pickle.load(f)
    
    print(f"  用户数: {mapping_info['num_users']}")
    print(f"  新闻数: {mapping_info['num_news']}")
    
    # 3. 定义特征列
    dense_cols = [
        'hist_click_count', 'hist_ctr', 'title_length',
        'impression_count', 'click_count', 'ctr'
    ]
    
    # 4. 划分数据集
    train_df, val_df = train_test_split(
        df, test_size=VAL_RATIO, random_state=42, stratify=df['label']
    )
    
    print(f"\n[2] 数据划分")
    print("-"*40)
    print(f"  训练集: {len(train_df)} 样本 ({1-VAL_RATIO:.0%})")
    print(f"  验证集: {len(val_df)} 样本 ({VAL_RATIO:.0%})")
    
    # 5. 创建DataLoader
    train_dataset = DINDataset(train_df, dense_cols, MAX_SEQ_LEN)
    val_dataset = DINDataset(val_df, dense_cols, MAX_SEQ_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\n[3] 创建DataLoader")
    print("-"*40)
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  训练批次数: {len(train_loader)}")
    print(f"  验证批次数: {len(val_loader)}")
    
    # 6. 创建模型
    print(f"\n[4] 创建DIN模型")
    print("-"*40)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  设备: {device}")
    
    # 计算类别数
    num_categories = int(df['category_encoded'].max()) + 1
    
    model = DIN(
        num_users=mapping_info['num_users'],
        num_news=mapping_info['num_news'],
        num_categories=num_categories,
        num_dense=len(dense_cols)
    ).to(device)
    
    print(f"  用户Embedding维度: {USER_EMB_DIM}")
    print(f"  新闻Embedding维度: {NEWS_EMB_DIM}")
    print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 7. 训练（核心修复：调用完整的train方法）
    print(f"\n[5] 开始训练")
    print("-"*40)
    
    trainer = Trainer(model, train_loader, val_loader, device)
    best_auc = trainer.train(epochs=EPOCHS)  # 使用全局EPOCHS参数
    
    # 8. 评估和可视化
    print(f"\n[6] 评估和可视化")
    print("-"*40)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = evaluate_and_visualize(model, val_loader, device, trainer.history, timestamp)
    
    # 9. 保存模型
    print(f"\n[7] 保存模型")
    print("-"*40)
    model_path = os.path.join(models_dir, f'din_model_{timestamp}.pth')
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_users': mapping_info['num_users'],
            'num_news': mapping_info['num_news'],
            'num_categories': num_categories,
            'num_dense': len(dense_cols)
        },
        'training_history': trainer.history,
        'final_metrics': {
            'best_val_auc': best_auc,
            'test_auc': results['auc'],
            'test_log_loss': results['log_loss']
        }
    }, model_path)
    
    print(f"  模型保存到: {model_path}")
    
    # 10. 对比
    print(f"\n[8] 与DeepFM对比")
    print("-"*40)
    print(f"  DeepFM AUC = 0.7493")
    print(f"  DIN 最佳验证AUC = {best_auc:.4f}")
    print(f"  DIN 测试集AUC = {results['auc']:.4f}")
    
    if results['auc'] > 0.74:
        print(f"  ✅ DIN 达到项目目标 (AUC > 0.74)")
    else:
        print(f"  ⚠️ DIN 暂未达到目标，需要继续优化")
    
    print("\n" + "="*60)
    print("DIN训练完成！")
    print("="*60)
    
    return model, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DIN模型训练')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    args = parser.parse_args()
    
    if args.epochs:
        EPOCHS = args.epochs
    
    model, results = main()