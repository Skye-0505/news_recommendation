# data_processor.py
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import pickle

# MINDDataProcessor:得到用户行为数据和新闻数据并添加列名
class MINDDataProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
    def load_behaviors(self):
        """加载用户行为数据"""
        behaviors = pd.read_csv(
            f'{self.data_dir}/behaviors.tsv',
            sep='\t',
            header=None,
            names=['impression_id', 'user_id', 'time', 'history', 'impressions']
        )
        return behaviors
    
    def load_news(self):
        """加载新闻数据"""
        news = pd.read_csv(
            f'{self.data_dir}/news.tsv',
            sep='\t',
            header=None,
            encoding='utf-8',
            names=['news_id', 'category', 'subcategory', 'title', 
                   'abstract', 'url', 'title_entities', 'abstract_entities']
        )
        return news

def extract_features(behaviors, news_df):
    """
    提取用户基础特征
    Returns: DataFrame with user_id and features
    """
    features = []
    # 新闻热度特征计算
    news_stats = defaultdict(lambda: {'impressions': 0, 'clicks': 0})
    
    for _, row in behaviors.iterrows():
        user_id = row['user_id']
        history = str(row['history'])
        time_str = row['time']
        impressions = str(row['impressions'])

        # 特征：
        # 历史点击新闻数量
        hist_count = 0 if history == 'nan' else len(history.split())
        
        # 活跃时间（小时）
        try:
            hour = datetime.strptime(time_str, '%m/%d/%Y %I:%M:%S %p').hour
        except:
            hour = 12  # 默认值
        
        # 工作日/周末（简化）
        try:
            weekday = datetime.strptime(time_str, '%m/%d/%Y %I:%M:%S %p').weekday()
            is_weekend = 1 if weekday >= 5 else 0
        except:
            is_weekend = 0

        # 用户历史CTR
        total_impressions = 0
        total_clicks = 0
        if impressions != 'nan':
            for item in impressions.split():
                if '-' in item:
                    total_impressions += 1 #曝光记录
                    news_id, label = item.split('-')
                    news_stats[news_id]['impressions'] += 1
                    if label == '1':
                        total_clicks += 1 #点击记录
                        news_stats[news_id]['clicks'] += 1
        
        # 用户CTR：用户 U1 本次会话看了 5 条新闻，点了 2 条 → CTR=0.4	
        hist_ctr = np.clip(total_clicks / max(total_impressions, 0.01), 0, 1)
        
        features.append({
            'user_id': user_id,
            'hist_click_count': hist_count,
            'active_hour': hour,
            'is_weekend': is_weekend,
            'hist_ctr': hist_ctr
        })
        
    user_features_df = pd.DataFrame(features)
    user_features_df = user_features_df.drop_duplicates(subset=['user_id'], keep='first')

    """
    提取新闻基础特征
    """

    news_df = news_df.copy()
    
    # 特征1: 标题长度（字符数）
    news_df['title_length'] = news_df['title'].fillna('').apply(len)
    
    # 特征2: 是否有摘要
    news_df['has_abstract'] = news_df['abstract'].notna().astype(int)
    
    # 特征3: 子类别编码（暂时用简单的标签编码）    
    le_sub = LabelEncoder()
    le_cat = LabelEncoder()
    news_df['subcategory_encoded'] = le_sub.fit_transform(news_df['subcategory'].fillna('unknown'))
    
    # 类别编码（大部分是'news'，但可能有其他）
    news_df['category_encoded'] = le_cat.fit_transform(news_df['category'].fillna('news'))

    # 转换为DataFrame
    stats_list = []
    for news_id, stats in news_stats.items():
        stats['news_id'] = news_id
        stats['ctr'] = np.clip(stats['clicks'] / max(stats['impressions'], 1), 0, 1) # 新闻CTR：新闻 N1 被 100 个用户看到，被 20 人点击 → CTR=0.2
        stats_list.append(stats)
    
    stats_df = pd.DataFrame(stats_list)

    # 合并到新闻特征
    news_features_df = news_df[['news_id', 'title_length', 'has_abstract', 
                                'subcategory_encoded', 'category_encoded']]
    
    news_features_df = pd.merge(news_features_df, stats_df, 
                               on='news_id', how='left').fillna(0)
    
    # 重命名列
    news_features_df = news_features_df.rename(columns={
        'impressions': 'impression_count',
        'clicks': 'click_count'
    })
    
    return user_features_df, news_features_df

def add_cross_features(df):
    """
    添加交叉特征（8个核心特征）
    """
    print("添加交叉特征...")
    
    # 1. 类别匹配特征（用户历史点击次数>0 且 新闻CTR>0 时匹配）
    df['category_match'] = ((df['hist_click_count'] > 0) & 
                           (df['ctr'] > 0)).astype(int)
    
    # 2. CTR差距特征
    df['ctr_gap'] = df['hist_ctr'] - df['ctr']
    df['ctr_gap_abs'] = abs(df['ctr_gap'])
    
    # 3. 点击-热度交互特征
    df['click_popularity_interaction'] = df['hist_click_count'] * df['impression_count']
    
    # 4. 时间匹配特征（假设新闻发布时间为随机，实际中应从数据推断）
    np.random.seed(42)  # 固定随机种子确保可复现
    df['news_publish_hour'] = np.random.randint(0, 24, len(df))
    df['hour_match'] = 1 - abs(df['active_hour'] - df['news_publish_hour']) / 24
    
    # 5. 周末新闻匹配特征（假设某些类别在周末更受欢迎）
    weekend_categories = set([0])
    df['weekend_news_match'] = (df['is_weekend'] * 
                               df['subcategory_encoded'].isin(weekend_categories)).astype(int)
    
    # 6. 标题长度匹配特征（用户平均偏好 vs 当前新闻）
    # 简化：假设用户偏好中等长度标题（50-100字）
    df['title_length_pref'] = np.where(
        (df['title_length'] >= 50) & (df['title_length'] <= 100), 1, 0
    )
    
    # 7. CTR比率特征
    df['ctr_ratio'] = np.clip(df['hist_ctr'] / (df['ctr'] + 0.01), 0, 10)  # 避免除零
    
    # 8. 用户-新闻质量交互特征
    df['quality_interaction'] = df['hist_ctr'] * df['ctr']
    
    print(f"    添加了 {len([c for c in df.columns if 'cross' in c.lower() or 'match' in c.lower() or 'gap' in c.lower() or 'interaction' in c.lower() or 'ratio' in c.lower()])} 个交叉特征")
    
    # 列出所有交叉特征
    cross_features = [
        'category_match', 'ctr_gap', 'ctr_gap_abs', 
        'click_popularity_interaction', 'hour_match', 
        'weekend_news_match', 'title_length_pref', 
        'ctr_ratio', 'quality_interaction'
    ]
    print(f"    交叉特征列表: {cross_features}")
    
    return df

def extract_user_sequences(behaviors_df, max_seq_len=50):
    """
    提取用户历史点击序列（为DIN准备）
    
    Args:
        behaviors_df: 用户行为DataFrame
        max_seq_len: 最大序列长度（保留最近N次点击）
    
    Returns:
        user_sequences: 字典 {用户ID: [新闻ID1, 新闻ID2, ...]}
    """
    print("提取用户历史行为序列...")
    
    user_sequences = {}
    
    # 按用户分组，每组内按时间排序
    behaviors_df = behaviors_df.sort_values(['user_id', 'time'])
    
    for user_id, group in behaviors_df.groupby('user_id'):
        clicked_items = []
        
        for _, row in group.iterrows():
            # 1. 添加历史点击（之前的会话）
            history = str(row['history'])
            if history != 'nan':
                clicked_items.extend(history.split())
            
            # 2. 添加当前会话中的点击
            impressions = str(row['impressions'])
            if impressions != 'nan':
                for item in impressions.split():
                    if '-' in item:
                        news_id, label = item.split('-')
                        if label == '1':  # 只记录点击的
                            clicked_items.append(news_id)
        
        # 只保留最近的 max_seq_len 个点击
        if len(clicked_items) > max_seq_len:
            clicked_items = clicked_items[-max_seq_len:]
        
        if len(clicked_items) > 0:
            user_sequences[user_id] = clicked_items
    
    # 打印统计信息
    seq_lengths = [len(seq) for seq in user_sequences.values()]
    print(f"  提取了 {len(user_sequences)} 个用户的序列")
    print(f"  平均序列长度: {np.mean(seq_lengths):.1f}")
    print(f"  最大序列长度: {max(seq_lengths)}")
    print(f"  最小序列长度: {min(seq_lengths)}")
    
    return user_sequences

def create_training_samples(behaviors_df, user_features_df, news_features_df, 
                           user_sequences=None, max_seq_len=20):
    """
    直接创建包含所有特征的训练样本（一步到位）
    Returns: DataFrame 包含所有特征
    """
   
    # 1. 先创建基础样本（复用原来的逻辑）
    samples = []
    for idx, row in behaviors_df.iterrows():

        user_id = row['user_id']
        impressions = str(row['impressions'])
        
        if impressions == 'nan':
            continue
            
        for item in impressions.split():
            if '-' in item:
                news_id, label = item.split('-')
                samples.append({
                    'user_id': user_id,
                    'news_id': news_id,
                    'label': int(label)
                })
    
    samples_df = pd.DataFrame(samples)
    print(f"  基础样本数: {len(samples_df)}")
    
    # 2. 合并用户特征
    print("  合并用户特征...")
    enriched_df = pd.merge(samples_df, user_features_df, on='user_id', how='left')
    
    # 3. 合并新闻特征
    print("  合并新闻特征...")
    enriched_df = pd.merge(enriched_df, news_features_df, on='news_id', how='left')
    # 处理缺失值
    enriched_df = enriched_df.fillna(0)
    # DeepFM数据（无交叉特征）
    deepfm_df = enriched_df.copy()
    # LR数据（有交叉特征）
    enriched_df = add_cross_features(enriched_df)

    # ========== 创建DIN数据 ==========
    din_samples = []
    
    # 如果没有传入序列数据，临时提取
    if user_sequences is None:
        print("    警告：未提供用户序列，临时提取...")
        user_sequences = extract_user_sequences(behaviors_df, max_seq_len)
    
    # 为每个样本添加历史序列
    for idx, row in enriched_df.iterrows():
        user_id = row['user_id']
        news_id = row['news_id']
        label = row['label']
        
        # 获取该用户的历史序列
        hist_seq = user_sequences.get(user_id, [])
        
        # 如果历史序列包含当前新闻，移除它（避免数据泄漏）
        if news_id in hist_seq:
            hist_seq = [item for item in hist_seq if item != news_id]
        
        # 确保序列长度不超过max_seq_len
        if len(hist_seq) > max_seq_len:
            hist_seq = hist_seq[-max_seq_len:]
        
        # 创建DIN样本
        din_sample = row.to_dict().copy()
        
        # 添加序列信息
        din_sample['history_seq'] = ','.join(hist_seq)
        din_sample['seq_length'] = len(hist_seq)
        din_sample['seq_mask'] = ','.join(['1'] * len(hist_seq) + ['0'] * (max_seq_len - len(hist_seq)))
        
        # 如果序列不足，填充到max_seq_len
        if len(hist_seq) < max_seq_len:
            padded_seq = hist_seq + ['<PAD>'] * (max_seq_len - len(hist_seq))
            din_sample['history_seq'] = ','.join(padded_seq)
        
        din_samples.append(din_sample)
    
    din_df = pd.DataFrame(din_samples)
    # 创建字符串id映射
    # 1. 创建新闻ID映射（必须！）
    all_news_ids = set()
    all_news_ids.update(samples_df['news_id'].unique())
    for seq in user_sequences.values():
        all_news_ids.update(seq)
    
    news_id2idx = {news_id: idx+1 for idx, news_id in enumerate(sorted(all_news_ids))}
    news_id2idx['<PAD>'] = 0
    
    # 2. 创建用户ID映射（可选，但推荐）
    all_users = samples_df['user_id'].unique()
    user_id2idx = {user: idx+1 for idx, user in enumerate(sorted(all_users))}
    user_id2idx['<UNK>'] = 0
    
    # 3. 为DIN添加索引列
    din_df['user_idx'] = din_df['user_id'].map(user_id2idx).fillna(0).astype(int)
    din_df['candidate_news_idx'] = din_df['news_id'].map(news_id2idx).fillna(0).astype(int)
    
    # 4. 历史序列转索引
    def seq_to_idx(seq_str):
        if pd.isna(seq_str) or seq_str == '':
            return []
        news_ids = seq_str.split(',')
        return [news_id2idx.get(nid, 0) for nid in news_ids]
    
    din_df['history_seq_idx'] = din_df['history_seq'].apply(seq_to_idx)
    
    # 5. 保存映射
    mapping_info = {
        'news_id2idx': news_id2idx,
        'user_id2idx': user_id2idx,
        'num_news': len(news_id2idx),
        'num_users': len(user_id2idx)
    }
    
    # ========== 返回三个数据集 ==========
    print(f"\n  最终数据集维度:")
    print(f"    LR数据集: {enriched_df.shape}")
    print(f"    DeepFM数据集: {deepfm_df.shape}")
    print(f"    DIN数据集: {din_df.shape}")

    return enriched_df, deepfm_df, din_df, mapping_info

def save_features(user_features, news_features, samples, deepfm_samples, din_samples, mapping_info, output_dir='../../data/processed'):
    """保存特征到文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    user_features.to_csv(f'{output_dir}/user_features.csv', index=False)
    news_features.to_csv(f'{output_dir}/news_features.csv', index=False)
    samples.to_csv(f'{output_dir}/train_samples.csv', index=False)
    deepfm_samples.to_csv(f'{output_dir}/train_deepfm_samples.csv', index=False)
    din_samples.to_csv(f'{output_dir}/train_din_samples.csv', index=False)

    # 保存mapping_info
    with open(f'{output_dir}/mapping_info.pkl', 'wb') as f:
        pickle.dump(mapping_info, f)
    
    # 同时保存一个可读的文本版本（方便查看）
    with open(f'{output_dir}/mapping_info.txt', 'w') as f:
        f.write(f"num_users: {mapping_info['num_users']}\n")
        f.write(f"num_news: {mapping_info['num_news']}\n")
        f.write("news_id2idx样例:\n")
        for news_id, idx in list(mapping_info['news_id2idx'].items())[:10]:
            f.write(f"  {news_id}: {idx}\n")
    
    print(f"特征已保存到 {output_dir} 目录")
    print(f"用户特征: {len(user_features)} 行")
    print(f"新闻特征: {len(news_features)} 行")
    print(f"lr训练样本: {len(samples)} 行")
    print(f"deepfm训练样本: {len(samples)} 行")
    print(f"DIN训练样本: {len(din_samples)} 行")

# 1. 加载数据
current_dir = os.path.dirname(os.path.abspath(__file__))
behaviors_file_path = os.path.join(current_dir, '../../data/raw/MINDsmall_train')
processor = MINDDataProcessor(behaviors_file_path)
behaviors = processor.load_behaviors()
news = processor.load_news()

 # 2. 提取特征
user_features, news_features = extract_features(behaviors, news)
print(f"提取了 {len(user_features)} 个用户的特征")
print(f"提取了 {len(news_features)} 篇新闻的特征")
print(user_features.head())
print(news_features.head())

# 3. 提取用户序列
user_sequences = extract_user_sequences(behaviors, max_seq_len=20)

# 4. 创建训练样本
samples, deepfm_samples, din_samples, mapping_info = create_training_samples(behaviors, user_features, news_features, user_sequences, max_seq_len=20)


processed_path = os.path.join(current_dir, '../../data/processed')
# 5.保存
save_features(user_features, news_features, samples, deepfm_samples, din_samples, mapping_info, processed_path)
