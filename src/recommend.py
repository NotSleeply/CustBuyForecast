"""
`recommend.py`  - **推荐系统** 
1. 统计高频购买商品
2. 为每个用户收集购买历史
3. 通过频次填充Top30推荐列表
"""

import numpy as np


def get_high_freq_items(train):
    """
    - 函数名: get_high_freq_items
    - 作用: 获取高频商品
    - 参数: train - 训练数据集
    - 返回值: 高频商品列表
    """
    temp = train.drop_duplicates(
        subset=['buyer_admin_id', 'item_id'], keep='first')
    item_cnts = temp.groupby(['item_id']).size().reset_index()
    item_cnts.columns = ['item_id', 'cnts']
    item_cnts = item_cnts.sort_values('cnts', ascending=False)
    items = item_cnts['item_id'].values.tolist()
    return items


def item_fillna(tmp_, items, top_n=30):
    """
    - 函数名: item_fillna
    - 作用: 填充用户商品列表
    - 参数: tmp_ - 用户商品列表, items - 高频商品列表, top_n - 填充数量
    - 返回值: 填充后的用户商品列表
    """
    tmp = tmp_.copy()
    l = len(tmp)
    if l == top_n:
        tmp = tmp
    elif l < top_n:
        m = top_n - l
        items_t = items.copy()
        for i in range(m):
            for j in range(50):
                it = items_t.pop(0)
                if it not in tmp:
                    tmp.append(it)
                    break
    elif l > top_n:
        tmp = tmp[:top_n]
    return tmp


def get_item_list(df_, items, top_n=30):
    """
    - 函数名: get_item_list
    - 作用: 生成用户商品列表
    - 参数: df_ - 输入数据, items - 高频商品列表
    - 返回值: 用户商品列表字典
    """
    df = df_.copy()
    dic = {}
    flag = 0
    for item in df[['buyer_admin_id', 'item_id']].values:
        try:
            dic[item[0]].append(item[1])
        except:
            if flag != 0:
                # 去重
                tmp = []
                for i in dic[flag]:
                    if i not in tmp:
                        tmp.append(i)
                # 填充
                tmp = item_fillna(tmp, items, top_n)
                dic[flag] = tmp
                print(f"已处理用户：{flag}")
                flag = item[0]
            else:
                flag = item[0]
            dic[item[0]] = [item[1]]
    # 最后一个用户也要去重和填充
    if flag != 0:
        tmp = []
        for i in dic[flag]:
            if i not in tmp:
                tmp.append(i)
        tmp = item_fillna(tmp, items, top_n)
        dic[flag] = tmp
        print(f"已处理用户：{flag}")
    return dic


def recommend_items_for_users(test, train, top_n=30):
    """
    - 函数名: recommend_items_for_users
    - 作用: 为每个用户推荐商品列表
    - 参数: test - 测试数据, train - 训练数据, top_n - 推荐数量
    - 返回值: 用户商品列表字典
    """
    print("统计高频item_id...")
    items = get_high_freq_items(train)
    print("高频item_id统计完成。")
    print("生成用户推荐item列表...")
    test = test.sort_values(['buyer_admin_id', 'irank'])
    dic = get_item_list(test, items, top_n)
    print("用户推荐item列表生成完成。")
    return dic