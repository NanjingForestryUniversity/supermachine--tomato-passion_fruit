import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
import copy
from math import *

#载入数据
data_path = './/merged1124.xlsx' #数据
label_path = './/label.xlsx' #标签（反射率）

D = pd.read_excel(data_path)
L = pd.read_excel(label_path)
data = np.array(D)
label = np.array(L)
print(label.shape)

# 绘制原始后图片
plt.figure(500)
x_col = data[:,0]  #数组逆序
y_col = data[:, 1:40]
plt.plot(x_col, y_col)
plt.xlabel("Wavenumber(nm)")
plt.ylabel("Absorbance")
plt.title("The spectrum of the tomato dataset",fontweight= "semibold",fontsize='x-large')
# plt.savefig('.//Result//MSC.png')
plt.show()

#随机划分数据集
x_data = np.transpose(data[:, 1:40])
print(x_data.shape)
y_data = copy.deepcopy(label)

print(y_data.shape)

test_ratio = 0.2
X_train,X_test,y_train,y_test = train_test_split(x_data,y_data,test_size=test_ratio,shuffle=True,random_state=2)

#载入数据
#PCA降维到10个维度,测试该数据最好
pca=PCA(n_components=28)  #只保留2个特征
pca.fit(X_train)
X_train_reduction = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)

# PCA降维后图片绘制
plt.figure(100)
plt.scatter(X_train_reduction[:, 0], X_train_reduction[:, 1],marker='o')
plt.xlabel("Wavenumber(nm)")
plt.ylabel("Absorbance")
plt.title("The  PCA for tomato dataset",fontweight= "semibold",fontsize='large')
# plt.savefig('.//Result//PCA.png')
plt.show()

#pls预测
pls2 = PLSRegression(n_components=5)
pls2.fit(X_train_reduction, y_train)

train_pred = pls2.predict(X_train_reduction)
pred = pls2.predict(X_test_reduction)

#计算R2
train_R2 = r2_score(train_pred,y_train)
R2 = r2_score(y_test,pred) #Y_true, Pred
print('训练R2:{}'.format(train_R2))
print('测试R2:{}'.format(R2))
#计算MSE
print('********************')
x_MSE = mean_squared_error(train_pred,y_train)
t_MSE = mean_squared_error(y_test,pred)
print('训练MSE:{}'.format(x_MSE))
print('测试MSE:{}'.format(t_MSE))

#计算RMSE
print('********************')
print('训练RMSE:{}'.format(sqrt(x_MSE)))
print('测试RMSE:{}'.format(sqrt(t_MSE)))
