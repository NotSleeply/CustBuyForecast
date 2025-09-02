"""
`feature_engineering.py`  - **特征工程** 
1. 特征选择
2. 特征提取
3. 特征构造
"""
import pandas as pd


def add_time_features(df):
    """
    函数名：add_time_features
    作用：添加时间相关特征
    参数：df - 输入数据
    返回值：添加时间特征后的数据
    """
    # 商品被购买总次数
    item_buy_count = df.groupby('item_id').size().rename('item_buy_count')
    # 商品被多少不同用户购买
    item_user_count = df.groupby(
        'item_id')['buyer_admin_id'].nunique().rename('item_user_count')
    # 商品最近一次被购买时间
    item_last_buy = df.groupby('item_id')['date'].max().rename('item_last_buy')
    df = df.merge(item_buy_count, on='item_id', how='left')
    df = df.merge(item_user_count, on='item_id', how='left')
    df = df.merge(item_last_buy, on='item_id', how='left')
    return df


def add_user_features(df):
    """
    函数名：add_user_features
    作用：添加用户购买记录数相关特征
    参数：df - 输入数据
    返回值：添加用户特征后的数据
    """
    # 用户历史购买次数
    user_buy_count = df.groupby(
        'buyer_admin_id').size().rename('user_buy_count')
    # 用户活跃天数
    user_active_days = df.groupby('buyer_admin_id')[
        'date'].nunique().rename('user_active_days')
    # 用户最近一次购买时间
    user_last_buy = df.groupby('buyer_admin_id')[
        'date'].max().rename('user_last_buy')
    df = df.merge(user_buy_count, on='buyer_admin_id', how='left')
    df = df.merge(user_active_days, on='buyer_admin_id', how='left')
    df = df.merge(user_last_buy, on='buyer_admin_id', how='left')
    return df


def add_item_features(df):
    """
    函数名：add_item_features
    作用：添加商品购买记录数相关特征
    参数：df - 输入数据
    返回值：添加商品特征后的数据
    """
    item_buy_count = df.groupby('item_id').size().rename('item_buy_count')
    df = df.merge(item_buy_count, on='item_id', how='left')
    return df


def add_features_main(df):
    """
    函数名：add_features_main
    作用：主函数，添加所有特征
    参数：df - 输入数据
    返回值：添加所有特征后的数据
    """
    df = add_time_features(df)
    df = add_user_features(df)
    return df
