import pandas as pd


def get_preprocessing(df_):
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
    item = pd.read_csv(path+'Antai_hackathon_attr.csv')
    test = pd.read_csv(path+'dianshang_test.csv')
    train = pd.read_csv(path+'Antai_hackathon_train.csv')
    return item, test, train
