import pandas as pd
from collections import Counter
# 读取Gowalla_totalCheckins.txt文件
df = pd.read_csv('Gowalla_totalCheckins.txt', sep='\t', header=None)
df.columns = ['user', 'time', 'latitude', 'longitude', 'location']
# 创建一个字典来存储Gowalla_totalCheckins.txt中每个用户和物品配对的交互次数
interaction_counts = Counter(zip(df['user'], df['location']))


# 读取user_list.txt文件并获取所有的org_id
df_user_list = pd.read_csv('gowalla_NGCF/user_list.txt', sep=' ', header=None, skiprows=1)
# 读取item_list.txt文件并获取所有的物品的org_id，转换为整数类型
df_item_list = pd.read_csv('gowalla_NGCF/item_list.txt', sep=' ', header=None, skiprows=1)

# 读取user_list.txt和item_list.txt文件，并创建用户和物品的映射字典
# 
# 读取user_list.txt和item_list.txt文件，并创建用户和物品的映射字典
user_mapping = {int(mapped_id): int(org_id) for org_id, mapped_id in zip(df_user_list[0], df_user_list[1])}
item_mapping = {int(mapped_id): int(org_id) for org_id, mapped_id in zip(df_item_list[0], df_item_list[1])}
# 创建物品的反向映射字典
item_mapping_reverse = {v: k for k, v in item_mapping.items()}

# 读取train.txt文件
df_train = pd.read_csv('gowalla_NGCF/test.txt', sep='\t', header=None)

# 创建一个新的数据集来存储结果
df_new = pd.DataFrame(columns=['mapped_user_id', 'mapped_item_id', 'interaction_count'])


# 创建一个空列表来存储数据
data = []

# 遍历train.txt文件中的每一行
for index, row in df_train.iterrows():
    # 将Series对象转换为字符串，然后分割字符串并转换为整数
    row = list(map(int, row[0].split()))

    # 获取映射后的用户id和物品id
    mapped_user_id = row[0]
    mapped_item_ids = row[1:]

    # 根据映射后的用户id和物品id获取原始的用户id和物品id
    org_user_id = user_mapping[mapped_user_id]
    org_item_ids = [item_mapping[mapped_item_id] for mapped_item_id in mapped_item_ids]

    # 根据原始的用户id和物品id在字典中查找交互次数，并将结果添加到列表中
    for org_item_id in org_item_ids:
        interaction_count = interaction_counts[(org_user_id, org_item_id)]
        mapped_item_id = item_mapping_reverse[org_item_id]
        data.append({'mapped_user_id': mapped_user_id, 'mapped_item_id': mapped_item_id, 'interaction_count': interaction_count})

# 将列表转换为DataFrame，并保存到文件
df_new = pd.DataFrame(data, columns=['mapped_user_id', 'mapped_item_id', 'interaction_count'])
df_new.to_csv('new_train_dataset.txt', sep='\t', index=False)