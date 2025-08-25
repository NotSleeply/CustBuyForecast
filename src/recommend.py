def get_high_freq_items(train):
    temp = train.loc[train.buyer_country_id == 'xx']
    temp = temp.drop_duplicates(
        subset=['buyer_admin_id', 'item_id'], keep='first')
    item_cnts = temp.groupby(['item_id']).size().reset_index()
    item_cnts.columns = ['item_id', 'cnts']
    item_cnts = item_cnts.sort_values('cnts', ascending=False)
    items = item_cnts['item_id'].values.tolist()
    return items


def item_fillna(tmp_, items):
    tmp = tmp_.copy()
    l = len(tmp)
    if l == 30:
        tmp = tmp
    elif l < 30:
        m = 30 - l
        items_t = items.copy()
        for i in range(m):
            for j in range(50):
                it = items_t.pop(0)
                if it not in tmp:
                    tmp.append(it)
                    break
    elif l > 30:
        tmp = tmp[:30]
    return tmp


def get_item_list(df_, items):
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
                tmp = item_fillna(tmp, items)
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
        tmp = item_fillna(tmp, items)
        dic[flag] = tmp
        print(f"已处理用户：{flag}")
    return dic
