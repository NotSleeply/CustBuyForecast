"""
`recall_sort.py` - **召回排序系统**
1. 共现召回
2. Item2Vec召回 
3. LightGBM排序
4. 多路召回融合
"""
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from gensim.models import Word2Vec
import lightgbm as lgb
from typing import Dict, Any


class RecallSortRecommender:
    def __init__(self):
        self.co_counts: Dict[Any, Dict[Any, float]] = defaultdict(lambda: defaultdict(float))
        self.w2v_model = None
        self.ranker = None
        self.item_popularity = None
        
    def build_covisit_matrix(self, train_df):
        """构建共现矩阵"""
        print("正在构建商品共现矩阵...")
        total_users = train_df['buyer_admin_id'].nunique()
        print(f"总用户数: {total_users}")
        
        for idx, (buyer, group) in enumerate(train_df.groupby('buyer_admin_id')):
            if idx % 10000 == 0:
                print(f"共现矩阵构建进度: {idx}/{total_users} 用户 ({idx/total_users*100:.1f}%)")
            
            items = list(group.sort_values('irank')['item_id'])
            for i in range(len(items)):
                for j in range(i+1, len(items)):
                    # 时间距离权重衰减
                    w = 1.0 / (1 + abs(j-i))
                    self.co_counts[items[i]][items[j]] += w
                    self.co_counts[items[j]][items[i]] += w
        print(f"共现矩阵构建完成，覆盖商品数: {len(self.co_counts)}")
    
    def train_item2vec(self, train_df):
        """训练Item2Vec模型"""
        print("正在训练Item2Vec模型...")
        sentences = []
        total_users = train_df['buyer_admin_id'].nunique()
        print(f"准备Item2Vec训练序列，总用户数: {total_users}")
        
        for idx, (buyer, group) in enumerate(train_df.groupby('buyer_admin_id')):
            if idx % 10000 == 0:
                print(f"Item2Vec序列收集进度: {idx}/{total_users} 用户")
            
            items = list(group.sort_values('irank')['item_id'])
            # 将item_id转换为字符串，确保gensim兼容性
            sentences.append([str(item) for item in items])
        
        print(f"Item2Vec总序列数: {len(sentences)}")
        if not sentences:
            print("警告：没有训练序列，跳过Item2Vec训练")
            return
            
        print("开始Item2Vec模型训练...")
        self.w2v_model = Word2Vec(
            sentences, 
            vector_size=64, 
            window=5, 
            min_count=2, 
            workers=4,
            epochs=10
        )
        print(f"Item2Vec训练完成，词汇表大小: {len(self.w2v_model.wv)}")
    
    def get_covisit_candidates(self, history_items, topn=50):
        """共现召回候选商品"""
        scores: Dict[Any, float] = defaultdict(float)
        for item in history_items:
            if item in self.co_counts:
                for similar_item, weight in self.co_counts[item].items():
                    if similar_item not in history_items:  # 过滤已购买
                        scores[similar_item] += weight
        
        candidates = [item for item, _ in Counter(scores).most_common(topn)]
        # 调试信息
        if len(candidates) == 0:
            print(f"共现召回警告：历史商品 {history_items} 无候选商品")
        
        return candidates
    
    def get_item2vec_candidates(self, history_items, topn=50):
        """Item2Vec召回候选商品"""
        if not self.w2v_model:
            return []
            
        scores: Dict[Any, float] = defaultdict(float)
        valid_items = 0
        for item in history_items:
            item_str = str(item)
            if item_str in self.w2v_model.wv:
                valid_items += 1
                try:
                    similar_items = self.w2v_model.wv.most_similar(item_str, topn=topn)
                    for similar_item_str, similarity in similar_items:
                        # 将字符串转回原始类型
                        try:
                            similar_item = int(similar_item_str)
                        except ValueError:
                            similar_item = similar_item_str
                        
                        if similar_item not in history_items:
                            scores[similar_item] += float(similarity)
                except KeyError:
                    continue
                except Exception as e:
                    print(f"Item2Vec相似度计算错误: {e}")
                    continue
        
        candidates = [item for item, _ in Counter(scores).most_common(topn)]
        # 调试信息
        if valid_items == 0:
            print(f"Item2Vec召回警告：历史商品 {history_items} 无有效商品")
        elif len(candidates) == 0:
            print(f"Item2Vec召回警告：历史商品 {history_items} 无候选商品")
                    
        return candidates
    
    def generate_candidates(self, test_df, topn=100):
        """为测试用户生成候选商品"""
        print("正在生成用户候选商品...")
        user_candidates = {}
        total_users = test_df['buyer_admin_id'].nunique()
        print(f"测试用户总数: {total_users}")
        
        def process_user(buyer_group):
            buyer, group = buyer_group
            history_items = list(group.sort_values('irank')['item_id'])
            covisit_cands = self.get_covisit_candidates(history_items, topn//2)
            i2v_cands = self.get_item2vec_candidates(history_items, topn//2)
            candidates = list(dict.fromkeys(covisit_cands + i2v_cands))
            return buyer, candidates

        buyer_groups = list(test_df.groupby('buyer_admin_id'))
        with ThreadPoolExecutor(max_workers=8) as executor: 
            for idx, result in enumerate(executor.map(process_user, buyer_groups)):
                buyer, candidates = result
                user_candidates[buyer] = candidates
                if idx % 1000 == 0:
                    print(
                        f"候选生成进度: {idx}/{total_users} 用户 ({idx/total_users*100:.1f}%)")

        if user_candidates:
            avg_candidates = np.mean([len(v) for v in user_candidates.values()])
            print(f"候选商品生成完成，平均每用户候选数: {avg_candidates:.1f}")
        return user_candidates
    
    def prepare_ranking_data(self, train_df):
        """准备排序训练数据"""
        print("正在准备排序训练数据...")
        train_samples = []
        
        # 构建用户购买历史
        user_history = train_df.groupby('buyer_admin_id')['item_id'].apply(set).to_dict()
        total_users = len(user_history)
        print(f"排序训练用户数: {total_users}")
        

        def process_user(buyer_group):
            buyer, group = buyer_group
            items = list(group.sort_values('irank')['item_id'])
            samples = []
            if len(items) < 3:
                  return samples
            history = items[1:]
            target = items[0]
            candidates = self.get_covisit_candidates(
                history, 30) + self.get_item2vec_candidates(history, 30)
            candidates = list(dict.fromkeys(candidates))
            if not candidates:
                return samples
            samples.append([buyer, target, 1])
            neg_samples = [c for c in candidates if c != target][:20]
            for neg_item in neg_samples:
                samples.append([buyer, neg_item, 0])
            return samples
        
        buyer_groups = list(train_df.groupby('buyer_admin_id'))
        from itertools import chain
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(process_user, buyer_groups))
        train_samples = list(chain.from_iterable(results))

        print(f"最终训练样本数: {len(train_samples)}")
        if not train_samples:
            print("警告：没有生成排序训练样本")
            return pd.DataFrame(columns=['buyer_admin_id', 'item_id', 'label'])
        return pd.DataFrame(train_samples, columns=['buyer_admin_id', 'item_id', 'label'])
    
    def extract_ranking_features(self, df_rank, train_df):
        """提取排序特征"""
        print(f"正在提取排序特征，样本数: {len(df_rank)}")
        
        if df_rank.empty:
            print("排序特征提取：输入数据为空")
            return df_rank
            
        # 构建统计特征
        user_history = train_df.groupby('buyer_admin_id')['item_id'].apply(set).to_dict()
        
        # 特征1: 是否购买过该商品
        df_rank['feat_bought'] = df_rank.apply(
            lambda x: 1 if x['item_id'] in user_history.get(x['buyer_admin_id'], set()) else 0, 
            axis=1
        )
        
        # 特征2: 商品热门度
        if self.item_popularity is None:
            self.item_popularity = train_df['item_id'].value_counts().to_dict()
        
        df_rank['feat_popularity'] = df_rank['item_id'].map(self.item_popularity).fillna(0)
        df_rank['feat_popularity'] = np.log1p(df_rank['feat_popularity'].astype(float))  # 对数变换
        
        print(f"排序特征提取完成:")
        print(f"  - feat_bought 平均值: {df_rank['feat_bought'].mean():.3f}")
        print(f"  - feat_popularity 平均值: {df_rank['feat_popularity'].mean():.3f}")
        
        return df_rank
    
    def train_ranker(self, train_df):
        """训练LightGBM排序模型"""
        print("正在训练排序模型...")
        df_rank = self.prepare_ranking_data(train_df)
        
        if df_rank.empty:
            print("警告：排序训练数据为空，使用默认排序策略")
            return
            
        df_rank = self.extract_ranking_features(df_rank, train_df)
        
        # 准备训练数据
        feature_cols = ['feat_bought', 'feat_popularity']
        X = df_rank[feature_cols].values
        y = df_rank['label'].values
        
        # 每个用户的样本数量（用于分组）
        group_sizes = df_rank.groupby('buyer_admin_id').size().values
        
        print(f"排序训练数据统计:")
        print(f"  - 特征维度: {X.shape}")
        print(f"  - 正样本数: {np.sum(y==1)}")
        print(f"  - 负样本数: {np.sum(y==0)}")
        print(f"  - 用户组数: {len(group_sizes)}")
        
        # 训练排序模型
        self.ranker = lgb.LGBMRanker(
            objective='lambdarank',
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42
        )
        
        try:
            print("开始LightGBM排序模型训练...")
            self.ranker.fit(X, y, group=group_sizes)
            print("排序模型训练完成")
        except Exception as e:
            print(f"排序模型训练失败: {e}")
            self.ranker = None
    

    def predict_and_rank(self, user_candidates, train_df, top_n=30):
        """对候选商品进行排序预测"""
        print("正在进行排序预测...")
        user_recommendations = {}

        # 热门商品作为兜底
        hot_items = list(train_df['item_id'].value_counts().head(100).index)
        user_history = train_df.groupby('buyer_admin_id')['item_id'].apply(set).to_dict()

        # 确保item_popularity已初始化
        if self.item_popularity is None:
            self.item_popularity = train_df['item_id'].value_counts().to_dict()

        total_users = len(user_candidates)
        print(f"排序预测用户数: {total_users}")
        
        no_candidates_count = 0
        no_ranker_count = 0
        success_count = 0
        fallback_count = 0

        for idx, (buyer, candidates) in enumerate(user_candidates.items()):
            if idx % 1000 == 0:
                print(f"排序预测进度: {idx}/{total_users} 用户 ({idx/total_users*100:.1f}%)")
            
            if not candidates or self.ranker is None:
                if not candidates:
                    no_candidates_count += 1
                if self.ranker is None:
                    no_ranker_count += 1
                    
                # 无候选商品或无排序模型时使用热门商品
                history = user_history.get(buyer, set())
                recommendations = [
                    item for item in hot_items if item not in history][:top_n]
            else:
                # 构建特征
                test_data = []
                for item in candidates:
                    test_data.append([buyer, item, 0])  # label无关紧要

                df_test = pd.DataFrame(test_data, columns=[
                                      'buyer_admin_id', 'item_id', 'label'])
                df_test = self.extract_ranking_features(df_test, train_df)

                # 预测排序分数
                feature_cols = ['feat_bought', 'feat_popularity']
                X_test = df_test[feature_cols].values

                try:
                    scores = self.ranker.predict(X_test)
                    # 确保scores是numpy数组
                    scores = np.array(scores, dtype=float)

                    # 排序并取top_n
                    sorted_indices = np.argsort(-scores)  # 降序排序
                    recommendations = [candidates[i]
                                      for i in sorted_indices[:top_n]]
                    success_count += 1
                except Exception as e:
                    print(f"排序预测失败: {e}")
                    fallback_count += 1
                    # 降级到基于热门度的简单排序
                    item_scores = [(item, self.item_popularity.get(item, 0) if self.item_popularity else 0)
                                  for item in candidates]
                    item_scores.sort(key=lambda x: x[1], reverse=True)
                    recommendations = [item for item, _ in item_scores[:top_n]]

                # 不足时用热门商品补充
                if len(recommendations) < top_n:
                    history = user_history.get(buyer, set())
                    for item in hot_items:
                        if item not in recommendations and item not in history:
                            recommendations.append(item)
                            if len(recommendations) >= top_n:
                                break

            user_recommendations[buyer] = recommendations[:top_n]

        print(f"排序预测完成统计:")
        print(f"  - 无候选商品用户数: {no_candidates_count}")
        print(f"  - 无排序模型用户数: {no_ranker_count}")
        print(f"  - 排序预测成功用户数: {success_count}")
        print(f"  - 降级处理用户数: {fallback_count}")
        
        if user_recommendations:
            avg_recs = np.mean([len(v) for v in user_recommendations.values()])
            print(f"排序预测完成，平均推荐数: {avg_recs:.1f}")
        return user_recommendations


def recall_sort_recommend(train, test):
    """召回排序推荐主函数"""
    print("=== 开始召回排序推荐系统 ===")
    recommender = RecallSortRecommender()
    
    # 1. 构建召回模型
    print("\n第1步：构建召回模型")
    recommender.build_covisit_matrix(train)
    recommender.train_item2vec(train)
    
    # 2. 训练排序模型
    print("\n第2步：训练排序模型")
    recommender.train_ranker(train)
    
    # 3. 生成候选商品
    print("\n第3步：生成候选商品")
    user_candidates = recommender.generate_candidates(test)
    
    # 4. 排序预测
    print("\n第4步：排序预测")
    recommendations = recommender.predict_and_rank(user_candidates, train)
    
    print("\n=== 召回排序推荐系统完成 ===")
    return recommendations