"""
`main.py`  - **调度中心** 
1. 读取数据
2. 数据预处理
3. 推荐算法
4. 数据保存
"""

import warnings
import pandas as pd
from src.preprocessing import get_preprocessing, load_data
from src.knn_recommend import knn_recommend


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

# 1. 读取数据
item, test, train = load_data('./data/')

# 2. 预处理
print("开始数据预处理...")
train = get_preprocessing(train)
test = get_preprocessing(test)
print("数据预处理完成。")

# 3. 推荐算法
dic = knn_recommend(train, test, n_neighbors=30)

# 4. 生成提交文件
print("正在生成提交文件...")
temp = pd.DataFrame({'lst': dic}).reset_index()
for i in range(30):
    temp[i] = temp['lst'].apply(lambda x: x[i])
del temp['lst']
temp.to_csv('./output/username.csv', index=False, header=False)
print("提交文件已保存为 ./output/username.csv")
