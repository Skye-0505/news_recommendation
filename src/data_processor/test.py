# 你可以用这段代码快速查看数据规模
import pandas as pd

train_behaviors_path = '../../data/raw/MINDsmall_train/behaviors.tsv'
train_news_path = '../../data/raw/MINDsmall_train/news.tsv'

def inspect_data():
    # 1. 查看文件是否存在
    import os
    print("检查文件...")
    print(f"behaviors.tsv 存在: {os.path.exists(train_behaviors_path)}")
    print(f"news.tsv 存在: {os.path.exists(train_news_path)}")
    
    # 2. 读取前几行
    print("\nbehaviors.tsv 前3行:")
    behaviors = pd.read_csv(train_behaviors_path, sep='\t', header=None, nrows=3)
    print(behaviors)
    
    print("\nnews.tsv 前3行:")
    news = pd.read_csv(train_news_path, sep='\t', header=None, nrows=3, encoding='utf-8')
    print(news)
    
    # 3. 基本信息
    # 读取数据
    train_behaviors = pd.read_csv(train_behaviors_path, sep='\t', header=None)
    train_news = pd.read_csv(train_news_path, sep='\t', header=None)

    # 计算正负样本比例
    impressions = train_behaviors[4].str.split()
    positive = 0
    total = 0
    for imp in impressions:
        for item in imp:
            if item.endswith('-1'):
                positive += 1
            total += 1

    print(f"总展示次数: {total:,}")
    print(f"正样本数(点击): {positive:,}")
    print(f"负样本数(未点击): {total-positive:,}")
    print(f"点击率(CTR): {positive/total:.2%}")

    print(f"训练集行为记录数: {len(train_behaviors):,}")
    print(f"训练集新闻数: {len(train_news):,}")
if __name__ == "__main__":
    inspect_data()