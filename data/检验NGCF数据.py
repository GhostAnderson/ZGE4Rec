import pandas as pd
from collections import Counter

# 读取user_list.txt文件并获取所有的org_id
df_user_list = pd.read_csv('gowalla_NGCF/user_list.txt', sep=' ', header=None)
# org_ids = df_user_list[0].tolist()[1:]
org_ids = [int(id) for id in df_user_list[0].tolist()[1:]]

# 读取Gowalla_totalCheckins.txt文件
df = pd.read_csv('Gowalla_totalCheckins.txt', sep='\t', header=None)
df.columns = ['user', 'time', 'latitude', 'longitude', 'location']

# 创建用户的计数器
user_counts = Counter(df['user'])

# 检查所有的org_id是否都存在于Gowalla_totalCheckins.txt的用户id中
is_all_exist = all(org_id in user_counts for org_id in org_ids)
# 检查这些org_id是否在Gowalla_totalCheckins.txt的用户id中都有过10次以上的出现
is_all_appear_more_than_10 = all(user_counts[org_id] >= 10 for org_id in org_ids)
# 创建一个列表来存储无法被检索到的org_id
not_found_org_ids = [org_id for org_id in org_ids if org_id not in user_counts]
# 打印结果
if is_all_appear_more_than_10:
    print("是的")
else:
    print("否")

# 读取item_list.txt文件并获取所有的物品的org_id，转换为整数类型
df_item_list = pd.read_csv('gowalla_NGCF/item_list.txt', sep=' ', header=None)
item_org_ids = [int(id) for id in df_item_list[0].tolist()[1:]]

# 创建物品的计数器
item_counts = Counter(df['location'])

# 检查所有的物品的org_id是否都存在于Gowalla_totalCheckins.txt的物品id中
is_all_item_exist = all(item_org_id in item_counts for item_org_id in item_org_ids)

if is_all_item_exist:
    # 如果所有的物品的org_id都存在，再检查这些org_id是否在Gowalla_totalCheckins.txt的物品id中都有过10次以上的出现
    is_all_item_appear_more_than_10 = all(item_counts[item_org_id] >= 10 for item_org_id in item_org_ids)

    # 打印结果
    if is_all_item_appear_more_than_10:
        print("物品ID检查结果：是的")
    else:
        print("物品ID检查结果：否")
else:
    # 创建一个列表来存储无法被检索到的物品的org_id
    not_found_item_org_ids = [item_org_id for item_org_id in item_org_ids if item_org_id not in item_counts]

    # 打印无法被检索到的物品的org_id的数量
    print("无法被检索到的物品的org_id的数量：", len(not_found_item_org_ids))

    # 打印无法被检索到的物品的org_id
    print("无法被检索到的物品的org_id：", not_found_item_org_ids)