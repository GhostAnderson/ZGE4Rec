import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.sparse import csr_matrix

# 1. 构造交互矩阵
df = pd.read_csv('Gowalla_totalCheckins.txt', sep='\t', header=None)
df.columns = ['user', 'time', 'latitude', 'longitude', 'location']

# 对地点进行去重并重新赋值id
unique_locations = pd.Series(df['location'].unique()).reset_index()
location_to_id = {location: idx for idx, location in unique_locations.values}
df['location_id'] = df['location'].map(location_to_id)

# 使用稀疏矩阵来存储交互矩阵，元素值为用户与地点的交互次数
interaction_matrix = csr_matrix((np.ones(len(df)), (df['user'], df['location_id'])), dtype=int)
interaction_matrix.sum_duplicates()  # 累加重复的交互

# 过滤掉与用户交互次数少于10次的地点
interaction_matrix = interaction_matrix[:, np.diff(interaction_matrix.indptr) >= 10]

# 2. 计算每个地点的向量的平均值和方差
mean_values = np.array(interaction_matrix.mean(axis=0)).reshape(-1)
var_values = np.array(interaction_matrix.power(2).mean(axis=0)).reshape(-1) - np.power(mean_values, 2)

# 3. 检验均值和方差的关系并作图
plt.figure(figsize=(10, 6))
plt.scatter(mean_values, var_values, alpha=0.5)
# plt.plot([0, max(mean_values)], [0, max(var_values)], color='red')  # y=x line
plt.xlabel('Mean')
plt.ylabel('Variance')
plt.title('Mean vs Variance')
# plt.axis('equal')
plt.xlim([0, 0.0012])
plt.ylim([0, 0.008])
plt.show()

# 3. 检验均值和方差的关系并作图
plt.figure(figsize=(10, 6))
plt.scatter(mean_values, var_values, alpha=0.5)
max_value = max(max(mean_values), max(var_values))
# max_value = max(max(mean_values), max(mean_values))
plt.plot([0, max_value], [0, max_value], color='red')  # y=x line
plt.xlabel('Mean')
plt.ylabel('Variance')
plt.title('Mean vs Variance')
# plt.axis('equal')
plt.xlim([0, 0.0012])
plt.ylim([0, 0.008])
plt.show()

# 4. 对地点出现的次数密度作图
location_counts = Counter(df['location'])
plt.figure(figsize=(10, 6))
# plt.hist(location_counts.values(), bins=40,log=True)
plt.hist(location_counts.values(), bins=np.arange(0, max(location_counts.values())+500, 250), log=True)
plt.xlabel('Number of occurrences')
plt.ylabel('Number of locations')
plt.title('Density of location occurrences')
plt.show()