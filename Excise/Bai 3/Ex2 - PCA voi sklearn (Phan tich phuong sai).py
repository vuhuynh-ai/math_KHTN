"""=============================================================================
Ex2: PCA - sklearn --> mở rộng thêm phân tích phương sai để xác định k
    a) Đọc tập tin dữ liệu Student_12f.xls vào dataframe.
    b) Áp dụng phương pháp PCA để giảm xuống k chiều (2 < k < 12).
       Giải thích nguyên nhân hay cơ sở về số chiều được giảm.
    c) Giảm chiều xuống còn k = 2 và trực quan hóa dữ liệu. Nhận xét kết quả.
============================================================================="""
#%%
import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import seaborn           as sns

from sklearn.decomposition import PCA

print('=============================================================')
print('*** a) Đọc tập tin dữ liệu vào dataframe                  ***')
print('=============================================================')

folder = './Data/Bai 3/'
data   = pd.read_excel(folder + 'Classification_12f_C.xls')
print(data.head(), '\n')

print('=============================================================')
print('*** b) Áp dụng PCA để giảm xuống còn k chiều (2 < k < 12) ***')
print('=============================================================')
#   https://stackoverflow.com/questions/32857029/python-scikit-learn-pca-explained-variance-ratio-cutoff
#   - The pca.explained_variance_ratio_ returns a vector of the variance explained by each dimension.
#   - The pca.explained_variance_ratio_[i] gives the variance explained solely by the i+1st dimension.
#   - The pca.explained_variance_ratio_.cumsum() will return a vector x 
#     such that x[i] returns the cumulative variance explained by the first i+1 dimensions.

#   (1) PCA().components_: Chuyển vị của ma trận vectơ riêng EigenVectors.T
#   (2) PCA().explained_variance_: Các giá trị riêng
#   (3) PCA().explained_variance_ratio_: Tỷ lệ phương sai so với dữ liệu gốc
#   (4) Hàm numpy.cumsum()

##------------------------------------------------------------------------------
print('CÁCH 1: Chọn k dựa trên đồ thị biểu diễn phương sai tích lũy ')
print('-------------------------------------------------------------')
##------------------------------------------------------------------------------
pca = PCA().fit(data)

# Vẽ đồ thị biểu diễn % phương sai tích lũy theo số features
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Số lượng features')
plt.ylabel('Variance (%)')
plt.title('Đồ thị biểu diễn % phương sai tích lũy theo số features')
plt.show()
# Nhận xét:
#   - #f =  0: giữ lại    0%  phương sai so với dữ liệu gốc
#   - #f =  3: giữ lại ~ 85%  phương sai so với dữ liệu gốc
#   - #f >  3: giữ lại > 90%  phương sai so với dữ liệu gốc

print('Phân tích chi tiết theo k: \n')
for k in range(1, 13):
    pca = PCA(k)
    pca.fit(data)
      
    var = pca.explained_variance_ratio_.sum() * 100
    print('   * Với k = %2d' %k, '--> phương sai tích lũy %.2f%%' %var)

    # Test - BEGIN
    print('      - Ma trận trị riêng LAMBDA', pca.explained_variance_.shape)   
    print('      - Chuyển vị của ma trận vectơ riêng P_T', pca.components_.shape, ': \n')
    # Test - END


##------------------------------------------------------------------------------
print('-------------------------------------------------------------')
print('CÁCH 2: Chọn k dựa trên ngưỡng phương sai tích lũy mong muốn ')
print('-------------------------------------------------------------')
##------------------------------------------------------------------------------
print('   * Giả sử muốn giữ lại 90%')
threshold = .9
percent   = threshold * 100
pca       = PCA(threshold)

pca.fit(data)
k   = pca.n_components_
var = sum(pca.explained_variance_ratio_) * 100
print('      - Muốn phương sai tích lũy >= %.2f%%' %percent, 'thì k >= %d' %k,
      ' (k = %d' %k, '--> %.2f%%)' %var, '\n')

print('   * Phân tích chi tiết theo ngưỡng phương sai:')
A = np.array([.5, .6, .7, .8, .9, .95, .99])
for x in A:
    percent   = x * 100
    pca       = PCA(x)

    pca.fit(data)
    k   = pca.n_components_
    var = sum(pca.explained_variance_ratio_) * 100
    print('      - Muốn phương sai tích lũy >= %.2f%%' %percent, 'thì k >= %2d' %k,
          ' (k = %2d' %k, '--> %.2f%%)' %var)

print('\n=============================================================')
print('*** c) Giảm chiều còn k = 2 và trực quan hóa dữ liệu      ***')
print('=============================================================')
k   = 2
pca = PCA(k)
pca.fit(data)

# transform data
B = pca.transform(data)
principalDf = pd.DataFrame(data = B, columns = ['PC 1', 'PC 2'])
print('- Ma trận B_T', principalDf.head(), '\n')

# Trực quan hóa dữ liệu (KHÔNG phân loại)
plt.figure(figsize = (8, 6))
sns.jointplot(x = 'PC 1', y = 'PC 2', data = principalDf)              
plt.show()

# Lấy cột phân loại (Class) trong file dữ liệu
y = np.array(data.Class)
y = pd.DataFrame(data = y, columns = ['Class'])

# Ghép cột phân loại (Class) vào ma trận PCA
finalDf = pd.concat([principalDf, y], axis = 1)
print(finalDf.head(), '\n')

# Trực quan hóa dữ liệu (CÓ phân loại)
plt.figure(figsize = (8, 8))
sns.scatterplot(x = 'PC 1', y = 'PC 2', data = finalDf, hue = 'Class', legend = 'full')              
plt.show()


print('========================================================')
print('*** d) CHUẨN HÓA dữ liệu trước khi thực hiện PCA     ***')
print('========================================================')
from sklearn.preprocessing import StandardScaler
k = 2
pca_norm  = PCA(k)
data_norm = StandardScaler().fit_transform(data)
pca_norm.fit(data_norm)

# transform data
B_norm = pca_norm.transform(data_norm)
print('- Ma trận B_T', B_norm.shape)

principalDf_norm = pd.DataFrame(data = B_norm, columns = ['PC 1', 'PC 2'])
print(principalDf_norm.head(), '\n')

# Lấy cột phân loại (Class) trong file dữ liệu
y = np.array(data.Class)
y = pd.DataFrame(data = y, columns = ['Class'])

# Ghép cột phân loại (Class) vào ma trận PCA
finalDf_norm = pd.concat([principalDf_norm, y], axis = 1)
print(finalDf_norm.head(), '\n')

# Trực quan hóa dữ liệu (CÓ phân loại)
plt.figure(figsize = (8, 8))
sns.scatterplot(x = 'PC 1', y = 'PC 2', data = finalDf_norm, hue = 'Class', legend = 'full')              
plt.show()
