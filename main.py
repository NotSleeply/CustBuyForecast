"""
`main.py`  - **调度中心** 
1. 数据预处理
2. 数据加载
3. 数据保存
"""

import warnings
import pandas as pd
from src.preprocessing import get_preprocessing, load_data
from src.recommend import get_high_freq_items, get_item_list


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

# 1. 读取数据
item, test, train = load_data('./data/')

# 2. 预处理
print("开始数据预处理...")
train = get_preprocessing(train)
test = get_preprocessing(test)
print("数据预处理完成。")

# 3. 高频item统计
print("统计高频item_id...")
items = get_high_freq_items(train)
print("高频item_id统计完成。")

# 4. 生成推荐列表
print("生成用户推荐item列表...")
test = test.sort_values(['buyer_admin_id', 'irank'])
dic = get_item_list(test, items)
print("用户推荐item列表生成完成。")

# 5. 生成提交文件
print("正在生成提交文件...")
temp = pd.DataFrame({'lst': dic}).reset_index()
for i in range(30):
    temp[i] = temp['lst'].apply(lambda x: x[i])
del temp['lst']
temp.to_csv('./output/username.csv', index=False, header=False)
print("提交文件已保存为 ./output/username.csv")
