# MIND新闻推荐系统 - CTR预估模型对比

## 项目简介

本项目基于微软MIND（MIcrosoft News Dataset）新闻推荐数据集，实现并对比了三种经典的CTR（Click-Through Rate）预估模型：

- **LR (Logistic Regression)**：逻辑回归，带有人工交叉特征的线性模型基线
- **DeepFM**：端到端学习的深度因子分解机，自动学习特征交互
- **DIN (Deep Interest Network)**：深度兴趣网络，通过注意力机制建模用户历史行为序列

项目完整实现了从数据预处理、特征工程到模型训练、评估和可视化的全流程，旨在探索不同模型在新闻推荐场景下的表现差异。


## 快速开始

### 1. 环境准备

```bash
pip install pandas numpy scikit-learn torch matplotlib joblib
```

### 2. 数据处理

首先运行数据处理脚本，从原始数据中提取特征并生成训练样本：

```bash
python data_processor.py
```

该脚本会：
- 加载原始behaviors.tsv和news.tsv文件
- 提取用户特征（历史点击数、活跃时间、历史CTR等）
- 提取新闻特征（标题长度、是否有摘要、类别编码等）
- 生成三个版本的数据集：
  - `train_samples.csv`：包含人工交叉特征（用于LR）
  - `train_deepfm_samples.csv`：无人工交叉特征（用于DeepFM）
  - `train_din_samples.csv`：包含用户历史序列（用于DIN）
- 保存特征映射信息供模型使用

### 3. 模型训练

按顺序执行三个模型训练脚本：

```bash
# 1. 训练LR基线模型
python baseline_lr.py

# 2. 训练DeepFM模型
python deepfm_model.py

# 3. 训练DIN模型
python din.py （基础版，auc0.72+）
python din_improved_model.py (改进版，auc0.8+)

```

每个模型训练完成后，会在`models/`目录保存模型文件，在`results/`目录生成可视化图表。

## 模型设计

### LR (Logistic Regression)

**设计思路**：
- 作为项目基线模型，使用简单的逻辑回归
- 依赖人工构造的交叉特征来捕捉特征间的交互关系

**人工交叉特征（8个）**：
- `category_match`：类别匹配特征
- `ctr_gap`/`ctr_gap_abs`：用户CTR与新闻CTR的差距
- `click_popularity_interaction`：点击数与曝光数的交互
- `hour_match`：用户活跃时间与新闻发布时间匹配度
- `weekend_news_match`：周末新闻偏好
- `title_length_pref`：标题长度偏好
- `ctr_ratio`：CTR比率
- `quality_interaction`：用户质量与新闻质量的交互

**特点**：简单、可解释性强，但依赖特征工程的质量

---

### DeepFM

**设计思路**：
- 结合因子分解机（FM）和深度神经网络（DNN）
- FM部分学习二阶特征交互，DNN部分学习高阶非线性关系
- **无需人工构造交叉特征**，模型自动学习特征交互

**模型结构**：
- FM一阶部分：线性组合所有特征
- FM二阶部分：通过隐向量内积学习特征对交互
- DNN部分：拼接Embedding后通过多层神经网络
- 输出：FM部分 + DNN部分 + 偏置项

**特点**：端到端学习，自动特征交叉，效果通常优于LR

---

### DIN (Deep Interest Network)

**设计思路**：
- 针对推荐场景中用户兴趣多样化的特点
- 引入**注意力机制**，根据候选新闻动态计算用户历史行为的重要性
- 建模用户兴趣的演进和多样性

**核心组件**：
- **Activation Unit**：计算每个历史新闻与候选新闻的相关性得分
- **用户兴趣表示**：通过注意力权重加权求和历史新闻Embedding
- **Embedding层**：用户、新闻、类别的Embedding表示
- **DNN层**：拼接用户Embedding、候选新闻Embedding、兴趣表示和数值特征

**特点**：能够捕捉用户兴趣的动态变化，对长序列数据效果好

## 模型对比

| 模型 | 特征工程 | 核心优势 | 适用场景 |
|------|---------|---------|---------|
| **LR** | 需要人工构造交叉特征 | 简单、可解释、训练快 | 快速验证、业务需要可解释性 |
| **DeepFM** | 无需人工交叉特征 | 自动学习特征交互，端到端 | 特征关系复杂、追求效果 |
| **DIN** | 需要用户历史序列 | 建模用户兴趣多样性，注意力机制 | 用户行为丰富、兴趣多样化 |

## 预期结果

运行完整流程后，各模型在测试集上的AUC表现：

- **LR**：≈ 0.734（有人工交叉特征）
- **DeepFM**：≈ 0.749（无人工交叉特征）
- **DIN**：≈ 0.812（建模用户序列）

## 自定义训练参数

各模型支持命令行参数调整：

```bash
# DeepFM
python deepfm_model.py --embedding_dim 10 --epochs 30 --sample_fraction 1.0

# DIN
python din.py --epochs 30
```
## DIN模型改进点总结

### 改进前后的核心差异对比

| 改进点 | 原始版本 | 改进版本 | 提升效果 |
|--------|---------|---------|---------|
| **最终AUC** | ~0.72 | **0.8+** | **+8%以上** |

---

### 1. 🐛 **注意力单元维度错误修复（最关键）**

**原始问题**：
```python
# 输入是3维 [batch, seq_len, features]
# 但线性层错误地压扁了维度，导致不同样本的历史商品混在一起算分
score = self.dnn(input)  # [batch*seq_len, features] → 计算错误！
```

**改进方案**：
```python
# 使用LayerNorm适配3维输入，保持维度正确
self.dnn = nn.Sequential(
    nn.Linear(input_dim, 200),
    nn.LayerNorm(200),  # LayerNorm可处理3维输入
    nn.ReLU(),
    # ...
)
```
**效果**：AUC提升 **+0.03~0.04**

---

### 2. 🎯 **差特征（Diff Feature）引入**

**原始**：只用历史、候选、点积（3个特征）
**改进**：增加差特征 `history - candidate`（4个特征）

```python
# 原始
input = torch.cat([history_emb, candidate_emb, product], dim=-1)

# 改进
diff = history_emb - candidate_emb
input = torch.cat([history_emb, candidate_emb, product, diff], dim=-1)
```
**效果**：AUC提升 **+0.005~0.01**

---

### 3. 🔄 **类别特征融合**

**原始**：只用新闻ID embedding
**改进**：融合新闻类别信息

```python
# 原始
cand_emb = self.news_embedding(candidate_idx)

# 改进
cat_emb = self.cat_embedding(category_idx)
cand_emb = self.fusion(torch.cat([cand_emb, cat_emb], dim=-1))
```
**效果**：AUC提升 **+0.01~0.015**

---

### 4. ⚙️ **超参数优化**

| 参数 | 原始 | 改进 | 作用 |
|------|------|------|------|
| User Emb Dim | 64 | **128** | 更大容量，学更细粒度特征 |
| News Emb Dim | 64 | **128** | 同上 |
| Cat Emb Dim | 32 | **64** | 同上 |
| Dropout | 0.2 | **0.3** | 更强正则化，防过拟合 |
| Learning Rate | 1e-4 | **3e-5** | 更稳定训练 |
| Max Seq Len | 20 | **30** | 利用更多历史行为 |
| Epochs | 30 | **50** | 充分收敛 |
| Early Stop | 5 | **8** | 避免过早停止 |

**效果**：AUC提升 **+0.01**

---

### 5. 📉 **优化器改进**

**原始**：
```python
self.optimizer = torch.optim.Adam(...)
```

**改进**：
```python
self.optimizer = torch.optim.AdamW(...)  # 正确实现权重衰减
self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(...)  # 学习率调度
```
**效果**：AUC提升 **+0.005**

---

### 6. 🎨 **初始化策略优化**

**原始**：所有层统一Xavier初始化
**改进**：不同层差异化初始化

```python
if 'embedding' in name:
    nn.init.normal_(param, std=0.01)  # embedding用小标准差
else:
    nn.init.xavier_normal_(param)      # 其他层用Xavier
```
**效果**：训练更稳定，收敛更快

---

### 7. 🚀 **训练效率优化**

**原始**：每轮遍历计算正负样本权重
**改进**：预计算权重，避免重复计算

```python
# 原始（低效）
pos_count = sum(batch['label'].sum() for batch in train_loader)

# 改进（高效）
pos_count = df['label'].sum()  # 一次性计算
```
**效果**：训练速度提升，权重更稳定

---

### 改进效果总结

| 改进类别 | AUC提升 | 重要性 |
|---------|---------|--------|
| 维度错误修复 | +0.03~0.04 | ⭐⭐⭐ 最关键 |
| 类别特征融合 | +0.01~0.015 | ⭐⭐ 很重要 |
| 差特征 | +0.005~0.01 | ⭐⭐ 很重要 |
| 超参数调优 | +0.01 | ⭐⭐ 很重要 |
| 优化器改进 | +0.005 | ⭐ 有帮助 |
| 初始化优化 | 稳定性提升 | ⭐ 有帮助 |
| **总计** | **+0.08+** | **AUC 0.72 → 0.8+** |


## 注意事项

1. 首次运行需确保MIND数据集已下载并放置在正确的目录 data/raw
2. 数据处理脚本会生成较大文件，确保有足够磁盘空间