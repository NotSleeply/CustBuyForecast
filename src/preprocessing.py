"""
`preprocessing.py`  - **数据预处理**
1. 数据预处理
2. 数据加载
"""

import pandas as pd

def get_preprocessing(df_):
    """
    - 函数名: get_preprocessing
    - 作用: 对输入数据进行预处理
    - 参数: df_ - 输入数据
    - 返回值: 预处理后的数据
    """
    df = df_.copy()
    df['create_order_time'] = pd.to_datetime(
        df['create_order_time'], errors='coerce')
    df['hour'] = df['create_order_time'].dt.hour
    df['day'] = df['create_order_time'].dt.day
    df['month'] = df['create_order_time'].dt.month
    df['year'] = df['create_order_time'].dt.year
    df['date'] = (df['month'].values - 7) * 31 + df['day']
    del df['create_order_time']
    return df


def load_data(path='./data/'):
    """
    - 函数名: load_data
    - 作用: 加载数据
    - 参数: path - 数据文件路径
    - 返回值: item, test, train - 加载的数据
    """
    item = pd.read_csv(path+'Antai_hackathon_attr.csv')
    test = pd.read_csv(path+'dianshang_test.csv')
    train = pd.read_csv(path+'Antai_hackathon_train.csv')
    return item, test, train
