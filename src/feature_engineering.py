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
    df = df.copy()
    df['create_order_time'] = pd.to_datetime(
        df['create_order_time'], errors='coerce')
    df['hour'] = df['create_order_time'].dt.hour
    df['day'] = df['create_order_time'].dt.day
    df['weekday'] = df['create_order_time'].dt.weekday
    df['month'] = df['create_order_time'].dt.month
    return df


def add_user_features(df):
    """
    函数名：add_user_features
    作用：添加用户购买记录数相关特征
    参数：df - 输入数据
    返回值：添加用户特征后的数据
    """
    user_buy_count = df.groupby(
        'buyer_admin_id').size().rename('user_buy_count')
    df = df.merge(user_buy_count, on='buyer_admin_id', how='left')
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
