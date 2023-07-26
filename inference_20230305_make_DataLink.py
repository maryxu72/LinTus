from sklearn.decomposition import PCA
import numpy as np
import os
from tqdm import tqdm
import pickle

np_files = os.listdir('./npy_group')

for npn in tqdm(np_files, total=len(np_files), leave=True):
    out1 = np.load(os.path.join('./npy_group', npn))

    if 'out_all' not in vars():
        out_all = out1
    else:
        out_all = np.concatenate((out_all, out1), axis=0)

del out1
# 保存整个数据输出
# np.save('./npy_group/out_all_total.npy', out_all)
# 设置数据链表参数
wd = 1
wz = 100000000
savepath = f'{wd}_{wz}out'
if not os.path.exists(savepath):
    os.mkdir(savepath)
# PCA降维
# pca = PCA(n_components=wd)
# pca.fit(out_all)
# with open(os.path.join(savepath, '1pca.pkl'), 'wb') as f:
#     pickle.dump(pca, f)
# 载入PCA
max_out_all = np.load(os.path.join(savepath, f'1max_{wd}_{wz}.npy'))

min_out_all = np.load(os.path.join(savepath, f'1min_{wd}_{wz}.npy'))

with open(os.path.join(savepath, '1pca.pkl'), 'rb') as f1:
    pca = pickle.load(f1)
out_all1 = pca.transform(out_all)

# max_out_all = np.max(out_all1, axis=0)
# print(max_out_all)
# np.save(os.path.join(savepath, f'1max_{wd}_{wz}.npy'), max_out_all)
# # input(max_out_all)
# min_out_all = np.min(out_all1, axis=0)
# print(min_out_all)
# np.save(os.path.join(savepath, f'1min_{wd}_{wz}.npy'), min_out_all)
# input(min_out_all)
out_all1 = (out_all1 - min_out_all) / (max_out_all - min_out_all)
out_all1 = np.around(out_all1 * (wz-1)).astype(int)
numlenmax = 0
DataLink = np.full([wz for i in range(wd)], {})
for idx1, ix1 in tqdm(enumerate(out_all1), total=len(out_all1), leave=True):
    ix1str = ','.join(str(j) for j in ix1)
    if DataLink[eval(ix1str)]:
        DataLink[eval(ix1str)]['address'].append(idx1)
    else:
        DataLink[eval(ix1str)] = {'address': [idx1]}
        # DataLink[eval(ix1str)] = {'address': [idx1], 'move': []}

    # print(DataLinked1[out_all[i][0]][out_all[i][1]][out_all[i][2]][out_all[i][3]][out_all[i][4]])

    if len(DataLink[eval(ix1str)]['address']) > numlenmax:
        numlenmax = len(DataLink[eval(ix1str)]['address'])
print(numlenmax)
np.save(os.path.join(savepath, f'inference0619_{wd}_{wz}.npy'), DataLink)
