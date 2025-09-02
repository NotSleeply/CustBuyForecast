"""
`knn_recommend.py`  - **推荐系统** 
1. KNN算法推荐流程
2. 基于用户历史行为推荐
3. 生成推荐结果
"""
from sklearn.neighbors import NearestNeighbors
from src.recommend import get_high_freq_items

n_neighbors = 30


def build_item_matrix(train):
    """
    函数名：build_item_matrix
    作用：构造商品特征矩阵
    参数：train - 训练数据
    返回值：商品特征矩阵
    """
    item_matrix = train.groupby('item_id')[['item_buy_count']].mean()
    return item_matrix


def get_user_history(train, uid):
    """
    函数名：get_user_history
    作用：获取用户历史购买商品列表
    参数：train - 训练数据, uid - 用户ID
    返回值：用户历史购买商品列表
    """
    return list(train[train['buyer_admin_id'] == uid]['item_id'].unique())


def knn_recommend(train, test):
    """
    函数名：KNN算法推荐流程
    作用：基于KNN算法为每个用户推荐商品
    参数：
        train: 训练集（需包含 item_id, item_buy_count 等特征）
        test: 测试集（需包含 buyer_admin_id, item_id 等特征）
    输出：
        user_recommend: {buyer_admin_id: 推荐商品列表}
    """
    # 构造商品特征矩阵
    item_matrix = build_item_matrix(train)
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(item_matrix)

    # 获取高频商品列表
    hot_items = get_high_freq_items(train)

    user_recommend = {}
    for uid, group in test.groupby('buyer_admin_id'):
        history = get_user_history(train, uid)
        indices = []
        for item in history:
            if item in item_matrix.index:
                _, idx = knn.kneighbors(item_matrix.loc[[item]])
                indices.extend(idx[0])
        # 去重、过滤已买
        indices = list(set(indices) - set(history))
        # 推荐商品id
        recommend_items = list(item_matrix.index[indices])[:n_neighbors]

        # 补齐热门商品
        if len(recommend_items) < n_neighbors:
            for hot in hot_items:
                if hot not in recommend_items and hot not in history:
                    recommend_items.append(hot)
                if len(recommend_items) == n_neighbors:
                    break
        
        user_recommend[uid] = recommend_items
        print(f"KNN已处理用户：{uid}")
    return user_recommend
