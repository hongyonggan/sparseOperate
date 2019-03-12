# 导入包
from scipy.sparse import csr_matrix


def sparse_to_dense_with_default(sparse_matrics, default_value=0):
    '''
    接受稀疏矩阵，scipy.sparse，进行0值及默认值填充
    sparse_matrics：需要转换的稀疏矩阵。
    default_value：转换稠密矩阵后的默认值，默认为0。
    '''
    if default_value==0:
        return sparse_matrics.todense()
    else:
        # 获取稀疏矩阵的值
        data = sparse_matrics.data
        # 让矩阵变换为与你默认值大小的值
        new_data = (data/data)*(-default_value)
        # 获取值的位置
        indices = sparse_matrics.indices
        indptr = sparse_matrics.indptr
        # 生成新的稀疏矩阵
        csr = csr_matrix((new_data, indices, indptr), shape=sparse_matrics.shape)
        result_dense = csr.todense()+default_value+sparse_matrics.todense()
        return result_dense

    
# demo
X = [[1, 0, None], [0, 1, 0], [0, 0, 1]]
# 构建稀疏矩阵
sparse_matrics = csr_matrix(X, dtype='float')
# 把全部 0值及空缺值填充为 -1
result_dense = sparse_to_dense_with_default(sparse_matrics, -1)
print(result_dense)
# matrix([[ 1., -1., -1.],
#        [-1.,  1., -1.],
#        [-1., -1.,  1.]])
