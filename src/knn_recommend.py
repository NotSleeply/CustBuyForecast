"""
`knn_recommend.py`  - **推荐系统** 
1. KNN算法推荐流程
2. 基于用户历史行为推荐
3. 生成推荐结果
"""
from sklearn.neighbors import NearestNeighbors

def knn_recommend(train, test, n_neighbors=30):
    """
    函数名：KNN算法推荐流程
    作用：基于KNN算法为每个用户推荐商品
    参数：
        train: 训练集（需包含 item_id, item_buy_count 等特征）
        test: 测试集（需包含 buyer_admin_id, item_id 等特征）
        n_neighbors: 推荐商品数
    输出：
        user_recommend: {buyer_admin_id: 推荐商品列表}
    """
    # 构造商品特征矩阵
    item_matrix = train.groupby('item_id')[['item_buy_count']].mean()
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(item_matrix)

    user_recommend = {}
    for uid, group in test.groupby('buyer_admin_id'):
        history = list(train[train['buyer_admin_id']
                       == uid]['item_id'].unique())
        indices = []
        for item in history:
            if item in item_matrix.index:
                _, idx = knn.kneighbors(item_matrix.loc[[item]])
                indices.extend(idx[0])
        # 去重、过滤已买
        indices = list(set(indices) - set(history))
        # 推荐商品id
        recommend_items = list(item_matrix.index[indices])[:n_neighbors]
        user_recommend[uid] = recommend_items
        print(f"KNN已处理用户：{uid}")
    return user_recommend
