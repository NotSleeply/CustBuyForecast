"""
`main.py`  - **调度中心** 
1. 读取数据
2. 数据预处理
3. 特征工程
4. 推荐算法
5. 数据保存
"""

import warnings
import pandas as pd
from src.preprocessing import get_preprocessing, load_data
from src.knn_recommend import knn_recommend
from src.feature_engineering import add_features_main
from src.save_result import save_submission


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

# 1. 读取数据
item, test, train = load_data('./data/')

# 2. 预处理
print("开始数据预处理...")
train = get_preprocessing(train)
test = get_preprocessing(test)
print("数据预处理完成。")

# 3. 特征工程
print("开始特征工程...")
train = add_features_main(train)
test = add_features_main(test)
print("特征工程完成。")

# 4. 推荐算法
dic = knn_recommend(train, test)

# 5. 生成提交文件
save_submission(dic)
