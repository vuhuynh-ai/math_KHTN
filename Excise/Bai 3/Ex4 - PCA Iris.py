"""=============================================================================
   Ex4: PCA - sklearn
      a) Đọc dữ liệu từ Iris.xls vào dataframe
      b) Tìm correlation matrix, trực quan hóa   
      c) Dùng PCA giảm xuống còn 2 chiều (ban đầu 4 chiều, không kể cột loại iris)
      d) Trực quan hóa dữ liệu sau khi giảm chiều
============================================================================="""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

##------------------------------------------------------------------------------
print('\n*** a) Đọc dữ liệu từ Iris.xls vào dataframe:')
##------------------------------------------------------------------------------
folder = 'Data/Bai 3/'
data = pd.read_excel(folder + 'Iris.xls')
print(data.head())

##------------------------------------------------------------------------------
print('\n*** b) Tìm correlation matrix, trực quan hóa:')
##------------------------------------------------------------------------------
corr = data.corr()
print('   - Ma trận hiệp phương sai', corr.shape, ': \n', corr)
sns.heatmap(corr, xticklabels = corr.columns.values, 
                  yticklabels = corr.columns.values)

##------------------------------------------------------------------------------
print('\n*** c) Thực hiện giảm chiều dữ liệu, k= 2, với sklearn.PCA:')
##------------------------------------------------------------------------------
A = data[['sepallength','sepalwidth','petallength','petalwidth']].values
print('   - Ma trận A: \n', A[0:5], '\n')

pca = PCA(2)
pca.fit(A)

# access values and vectors
# components_ : array, shape (n_components, n_features)
# Các trục chính trong không gian feature, biểu thị
# các hướng của phương sai tối đa trong dữ liệu
# explained_variance_ : array, shape (n_components,)
# Số lượng phương sai được giải thích bởi từng thành phần được chọn.
print('PCA.Components:\n', pca.components_)
print('PCA.Shape: ', pca.components_.shape)
print('PCA.Explained variance: ', pca.explained_variance_)
print('PCA.Explained variance shape: ', pca.explained_variance_.shape)

# transform data
B = pca.transform(A)
print('         - Ma trận B_T', B.shape, ': \n', B[0:5], '\n')
print(pca.explained_variance_ratio_)

##------------------------------------------------------------------------------
print('\n*** d) Trực quan hóa dữ liệu (KHÔNG phân loại):')
##------------------------------------------------------------------------------
principalDf = pd.DataFrame(data = B, columns = ['PC 1', 'PC 2'])
print(principalDf.head(), '\n')

principalDf = pd.DataFrame(data = B, columns = ['PC 1', 'PC 2'])
print(principalDf.head(), '\n')

# Trực quan hóa dữ liệu (KHÔNG phân loại)
plt.figure(figsize = (8, 6))
sns.jointplot(x = 'PC 1', y = 'PC 2', data = principalDf)              
plt.show()

# Lấy cột phân loại (Types) trong file dữ liệu
y = np.array(data.iris)
y = pd.DataFrame(data = y, columns = ['Types'])

# Ghép cột phân loại (Class) vào ma trận PCA
finalDf = pd.concat([principalDf, y], axis = 1)
print(finalDf.head(), '\n')

# Trực quan hóa dữ liệu (CÓ phân loại)
plt.figure(figsize = (8, 8))
sns.scatterplot(x = 'PC 1', y = 'PC 2', data = finalDf, hue = 'Types', legend = 'full')              
plt.show()
