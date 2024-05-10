import pandas as pd
import os
# 指定Excel文件的文件夹路径
folder_path = 'average_result'  # 替换成你的文件夹路径
# 获取文件夹下所有Excel文件的文件名
excel_files = [file for file in os.listdir(folder_path) if file.endswith('.xlsx') or file.endswith('.xls')]
# 按照文件名排序，确保按照表格序号顺序读取
# 使用int函数将文件名转换为整数，然后进行排序
# excel_files.sort(key=lambda x: int(x.split('.')[0]))

# 创建一个空列表，用于存储每个Excel文件的数据框
dfs = []
for i in range(1,40):
    df = pd.read_excel(folder_path + '/'+'test'+str(i) +'.xlsx')
    # 将数据框添加到列表中
    dfs.append(df)

# 使用pandas.concat函数将列表中的所有数据框按列合并
result = pd.concat(dfs, axis=1)

# 将结果保存为新的Excel文件
result.to_excel('merged.xlsx', index=False)