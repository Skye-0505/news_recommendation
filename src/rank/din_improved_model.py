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

# ==================== 改进后的配置 ====================
USER_EMB_DIM = 128
NEWS_EMB_DIM = 128
CAT_EMB_DIM = 64
HIDDEN_DIMS = [512, 256, 128]
DROPOUT = 0.3  # 增加dropout防止过拟合
BATCH_SIZE = 512
EPOCHS = 50  # 增加轮数
LEARNING_RATE = 0.00003  # 降低学习率
WEIGHT_DECAY = 0.0001
EARLY_STOP_PATIENCE = 8
VAL_RATIO = 0.2
MAX_SEQ_LEN = 30  # 增加序列长度

# ==================== 改进后的Activation Unit ====================
class ActivationUnit(nn.Module):
    """改进版的DIN注意力单元，修复BatchNorm1d维度问题"""
    def __init__(self, embedding_dim):
        super().__init__()
        # 输入维度：原始embedding * 4（拼接+点积+差）
        input_dim = embedding_dim * 4
        
        # 核心修复：使用BatchNorm1d时，先flatten seq维度，或改用LayerNorm
        # 方案：对最后一维做LayerNorm（适配3维输入）
        self.dnn = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.LayerNorm(200),  # 替换BatchNorm1d为LayerNorm，适配3维输入
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(200, 80),
            nn.LayerNorm(80),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(80, 40),
            nn.LayerNorm(40),
            nn.ReLU(),
            
            nn.Linear(40, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.dnn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, history_emb, candidate_emb):
        # 扩展候选embedding以匹配历史序列
        # history_emb: [batch_size, seq_len, embedding_dim]
        # candidate_emb: [batch_size, embedding_dim] → [batch_size, seq_len, embedding_dim]
        if len(candidate_emb.shape) == 2:
            candidate_emb = candidate_emb.unsqueeze(1).expand_as(history_emb)
        
        # 原文特征：拼接 + 点积 + 差
        product = history_emb * candidate_emb
        diff = history_emb - candidate_emb
        
        # 拼接所有特征: [batch_size, seq_len, 4*embedding_dim]
        input_tensor = torch.cat([history_emb, candidate_emb, product, diff], dim=-1)
        
        # 计算注意力得分: [batch_size, seq_len, 1] → [batch_size, seq_len]
        score = self.dnn(input_tensor)
        
        return score.squeeze(-1)

# ==================== 改进后的DIN模型 ====================
class DINImproved(nn.Module):
    def __init__(self, num_users, num_news, num_categories, num_dense):
        super().__init__()
        
        # Embedding层
        self.user_embedding = nn.Embedding(num_users, USER_EMB_DIM, padding_idx=0)
        self.news_embedding = nn.Embedding(num_news, NEWS_EMB_DIM, padding_idx=0)
        self.cat_embedding = nn.Embedding(num_categories, CAT_EMB_DIM, padding_idx=0)
        
        # 激活单元
        self.activation_unit = ActivationUnit(NEWS_EMB_DIM)
        
        # 特征融合层
        self.fusion = nn.Linear(NEWS_EMB_DIM + CAT_EMB_DIM, NEWS_EMB_DIM)
        
        # DNN层
        # 输入：user_emb + cand_emb + interest_emb + dense_features
        input_dim = USER_EMB_DIM + NEWS_EMB_DIM + NEWS_EMB_DIM + num_dense
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in HIDDEN_DIMS:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))  # 这里是2维输入，BatchNorm1d可用
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(DROPOUT))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.dnn = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                if 'embedding' in name:
                    nn.init.normal_(param, std=0.01)
                else:
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, user_idx, candidate_idx, history_idx, mask, dense, category_idx):
        # 1. Embedding
        user_emb = self.user_embedding(user_idx)  # [batch_size, USER_EMB_DIM]
        cand_emb = self.news_embedding(candidate_idx)  # [batch_size, NEWS_EMB_DIM]
        
        # 2. 融合类别信息
        cat_emb = self.cat_embedding(category_idx)  # [batch_size, CAT_EMB_DIM]
        cand_emb = self.fusion(torch.cat([cand_emb, cat_emb], dim=-1))  # [batch_size, NEWS_EMB_DIM]
        
        # 3. 历史序列embedding
        history_emb = self.news_embedding(history_idx)  # [batch_size, seq_len, NEWS_EMB_DIM]
        
        # 4. 注意力权重计算（修复空mask情况）
        attn_weights = self.activation_unit(history_emb, cand_emb)  # [batch_size, seq_len]
        attn_weights = attn_weights.masked_fill(~mask, -1e9)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # 5. 用户兴趣表示
        interest_emb = torch.sum(attn_weights.unsqueeze(-1) * history_emb, dim=1)  # [batch_size, NEWS_EMB_DIM]
        
        # 6. 拼接所有特征
        combined = torch.cat([user_emb, cand_emb, interest_emb, dense], dim=1)  # [batch_size, input_dim]
        
        # 7. 输出
        logits = self.dnn(combined).squeeze()  # [batch_size]
        
        return logits

# ==================== 改进后的Dataset ====================
class DINDatasetImproved(Dataset):
    def __init__(self, df, dense_cols, max_seq_len=30):
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
        category_idx = int(row['category_encoded'])  # 添加类别特征
        
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
            'category_idx': torch.LongTensor([category_idx]),
            'history_idx': torch.LongTensor(history_idx),
            'mask': torch.BoolTensor(mask),
            'dense': dense,
            'label': label
        }

# ==================== 训练器 ====================
class Trainer:
    def __init__(self, model, train_loader, val_loader, device, pos_weight=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 优化：避免遍历整个loader计算pos_weight，改用预计算值
        if pos_weight is None:
            # 若未传入，用默认值（也可从df中统计，更高效）
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
        
        # 使用AdamW优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        
        # 学习率调度器（修复verbose警告）
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3
        )
        
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_auc': [], 'val_auc': []
        }
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in self.train_loader:
            user_idx = batch['user_idx'].to(self.device).squeeze()
            candidate_idx = batch['candidate_idx'].to(self.device).squeeze()
            category_idx = batch['category_idx'].to(self.device).squeeze()
            history_idx = batch['history_idx'].to(self.device)
            mask = batch['mask'].to(self.device)
            dense = batch['dense'].to(self.device)
            label = batch['label'].to(self.device).squeeze()
            
            logits = self.model(user_idx, candidate_idx, history_idx, mask, dense, category_idx)
            loss = self.criterion(logits, label)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            pred = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.extend(pred)
            all_labels.extend(label.cpu().numpy())
        
        avg_loss = total_loss / len(self.train_loader)
        auc = roc_auc_score(all_labels, all_preds)
        
        return avg_loss, auc
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                user_idx = batch['user_idx'].to(self.device).squeeze()
                candidate_idx = batch['candidate_idx'].to(self.device).squeeze()
                category_idx = batch['category_idx'].to(self.device).squeeze()
                history_idx = batch['history_idx'].to(self.device)
                mask = batch['mask'].to(self.device)
                dense = batch['dense'].to(self.device)
                label = batch['label'].to(self.device).squeeze()
                
                logits = self.model(user_idx, candidate_idx, history_idx, mask, dense, category_idx)
                loss = self.criterion(logits, label)
                
                total_loss += loss.item()
                
                pred = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(pred)
                all_labels.extend(label.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        auc = roc_auc_score(all_labels, all_preds)
        
        return avg_loss, auc
    
    def train(self, epochs):
        print(f"\n开始训练，设备: {self.device}")
        print(f"训练集批次: {len(self.train_loader)}, 验证集批次: {len(self.val_loader)}")
        if hasattr(self.criterion, 'pos_weight'):
            print(f"正负样本权重: {self.criterion.pos_weight.item():.2f}")
        
        best_auc = 0.0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            train_loss, train_auc = self.train_epoch()
            val_loss, val_auc = self.validate()
            
            # 学习率调度（修复verbose警告）
            self.scheduler.step(val_auc)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_auc'].append(train_auc)
            self.history['val_loss'].append(val_loss)
            self.history['val_auc'].append(val_auc)
            
            # 打印epoch信息
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}, AUC: {train_auc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 早停逻辑
            if val_auc > best_auc:
                best_auc = val_auc
                best_model_state = self.model.state_dict().copy()
                best_history = {
                    'train_loss': self.history['train_loss'].copy(),
                    'train_auc': self.history['train_auc'].copy(),
                    'val_loss': self.history['val_loss'].copy(),
                    'val_auc': self.history['val_auc'].copy()
                }
                patience_counter = 0
                print(f"  → 验证AUC提升至 {best_auc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    print(f"\n早停触发！最佳验证AUC: {best_auc:.4f}")
                    break
        
        # 加载最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.history = best_history
        print(f"\n训练完成！最佳验证AUC: {best_auc:.4f}")
        return best_auc

# ==================== 评估和可视化 ====================
def evaluate_and_visualize(model, val_loader, device, history, results_dir, timestamp):
    """补充评估和可视化逻辑"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            user_idx = batch['user_idx'].to(device).squeeze()
            candidate_idx = batch['candidate_idx'].to(device).squeeze()
            category_idx = batch['category_idx'].to(device).squeeze()
            history_idx = batch['history_idx'].to(device)
            mask = batch['mask'].to(device)
            dense = batch['dense'].to(device)
            label = batch['label'].to(device).squeeze()
            
            logits = model(user_idx, candidate_idx, history_idx, mask, dense, category_idx)
            pred = torch.sigmoid(logits)
            
            all_preds.append(pred.cpu().numpy())
            all_labels.append(label.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # 计算指标
    auc = roc_auc_score(all_labels, all_preds)
    logloss = log_loss(all_labels, all_preds)
    
    print(f"\n最终评估结果:")
    print(f"  AUC: {auc:.4f}")
    print(f"  LogLoss: {logloss:.4f}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 损失曲线
    axes[0,0].plot(history['train_loss'], 'b-', label='Train Loss')
    axes[0,0].plot(history['val_loss'], 'r-', label='Val Loss')
    axes[0,0].set_title('Loss Curve')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # AUC曲线
    axes[0,1].plot(history['train_auc'], 'b-', label='Train AUC')
    axes[0,1].plot(history['val_auc'], 'r-', label='Val AUC')
    axes[0,1].set_title('AUC Curve')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('AUC')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 预测概率分布
    axes[1,0].hist(all_preds[all_labels==0], bins=50, alpha=0.5, label='负样本', density=True)
    axes[1,0].hist(all_preds[all_labels==1], bins=50, alpha=0.5, label='正样本', density=True)
    axes[1,0].set_title('Prediction Distribution')
    axes[1,0].set_xlabel('Predicted Probability')
    axes[1,0].set_ylabel('Density')
    axes[1,0].legend()
    
    # ROC曲线
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    axes[1,1].plot(fpr, tpr, 'b-', label=f'AUC = {auc:.4f}')
    axes[1,1].plot([0,1], [0,1], 'r--', label='Random')
    axes[1,1].set_title('ROC Curve')
    axes[1,1].set_xlabel('FPR')
    axes[1,1].set_ylabel('TPR')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'din_improved_{timestamp}.png'), dpi=300)
    plt.close()
    
    return {'auc': auc, 'logloss': logloss}

# ==================== 主函数 ====================
def main():
    print("="*60)
    print("DIN改进版 - CTR预估模型训练")
    print("="*60)
    
    # 设置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, '../../data/processed/train_din_samples.csv')
    mapping_path = os.path.join(current_dir, '../../data/processed/mapping_info.pkl')
    models_dir = os.path.join(current_dir, '../models')
    results_dir = os.path.join(current_dir, '../results')
    
    # 创建目录
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 加载数据
    print(f"\n[1] 加载数据")
    df = pd.read_csv(csv_path)
    print(f"  数据集: {df.shape[0]} 样本")
    print(f"  正样本比例: {df['label'].mean():.4f}")
    
    # 预计算正负样本权重（更高效）
    pos_count = df['label'].sum()
    neg_count = len(df) - pos_count
    pos_weight = neg_count / max(pos_count, 1)
    
    # 加载映射信息
    with open(mapping_path, 'rb') as f:
        mapping_info = pickle.load(f)
    
    # 定义特征列
    dense_cols = ['hist_click_count', 'hist_ctr', 'title_length',
                  'impression_count', 'click_count', 'ctr']
    
    # 划分数据集
    train_df, val_df = train_test_split(
        df, test_size=VAL_RATIO, random_state=42, stratify=df['label']
    )
    
    print(f"\n[2] 数据划分")
    print(f"  训练集: {len(train_df)} 样本")
    print(f"  验证集: {len(val_df)} 样本")
    
    # 创建DataLoader
    train_dataset = DINDatasetImproved(train_df, dense_cols, MAX_SEQ_LEN)
    val_dataset = DINDatasetImproved(val_df, dense_cols, MAX_SEQ_LEN)
    
    # 修复num_workers=0（避免多进程问题）
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 创建模型
    print(f"\n[3] 创建改进版DIN模型")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  设备: {device}")
    
    num_categories = int(df['category_encoded'].max()) + 1
    
    model = DINImproved(
        num_users=mapping_info['num_users'],
        num_news=mapping_info['num_news'],
        num_categories=num_categories,
        num_dense=len(dense_cols)
    ).to(device)
    
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练
    print(f"\n[4] 开始训练")
    trainer = Trainer(model, train_loader, val_loader, device, pos_weight=pos_weight)
    best_auc = trainer.train(epochs=EPOCHS)
    
    # 评估和可视化
    print(f"\n[5] 评估和可视化")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    eval_results = evaluate_and_visualize(model, val_loader, device, trainer.history, results_dir, timestamp)
    
    # 保存模型
    print(f"\n[6] 保存模型")
    model_path = os.path.join(models_dir, f'din_improved_{timestamp}.pth')
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_users': mapping_info['num_users'],
            'num_news': mapping_info['num_news'],
            'num_categories': num_categories,
            'num_dense': len(dense_cols)
        },
        'best_val_auc': best_auc,
        'final_eval': eval_results,
        'training_history': trainer.history
    }, model_path)
    
    print(f"  模型保存到: {model_path}")
    print(f"\n最佳验证AUC: {best_auc:.4f} | 最终测试AUC: {eval_results['auc']:.4f}")

if __name__ == "__main__":
    # 解析命令行参数（可选）
    parser = argparse.ArgumentParser(description='DIN Improved Model Training')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Training epochs')
    args = parser.parse_args()
    
    EPOCHS = args.epochs
    main()