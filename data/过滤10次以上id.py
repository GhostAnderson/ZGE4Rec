import pandas as pd
from collections import Counter

# 读取数据
df = pd.read_csv('Gowalla_totalCheckins.txt', sep='\t', header=None)
df.columns = ['user', 'time', 'latitude', 'longitude', 'location']

# 创建用户和地点的计数器
user_counts = Counter(df['user'])
location_counts = Counter(df['location'])

# 过滤出出现次数大于10的用户和地点
filtered_users = [user for user, count in user_counts.items() if count >= 10]
filtered_locations = [location for location, count in location_counts.items() if count >= 10]

# 打印结果
print('用户ID出现次数大于10的有：', filtered_users)
print('地点ID出现次数大于10的有：', filtered_locations)