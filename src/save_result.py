"""
`save_result.py`  - **结果保存** 
1. 保存提交文件
"""

import pandas as pd

path = './output/username.csv'
top_n = 30

def save_submission(dic):
    """
    保存推荐结果为提交文件
    参数:
        dic: {buyer_admin_id: 推荐商品列表}
    """
    print("正在生成提交文件...")
    temp = pd.DataFrame({'buyer_admin_id': list(
        dic.keys()), 'lst': list(dic.values())})
    for i in range(top_n):
        temp[i+1] = temp['lst'].apply(lambda x: x[i] if len(x) > i else -1)
    temp = temp.drop(columns=['lst'])
    temp.to_csv(path, index=False, header=False)
    print(f"提交文件已保存为 {path}")
