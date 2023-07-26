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
zu = [[1, 100000000], [2, 10000], [3, 500], [4, 100], [5, 50], [9, 9]]
for z1 in zu:
    # Set data list parameters
    wd = z1[0]
    wz = z1[1]
    savepath = f'{wd}_{wz}out'
    print(savepath, 'start')
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    # PCA dimensional reduction
    pca = PCA(n_components=wd)
    pca.fit(out_all)
    with open(os.path.join(savepath, 'pca.pkl'), 'wb') as f:
        pickle.dump(pca, f)
    out_all1 = pca.transform(out_all)

    max_out_all = np.max(out_all1, axis=0)
    np.save(os.path.join(savepath, f'max_{wd}_{wz}.npy'), max_out_all)

    min_out_all = np.min(out_all1, axis=0)
    np.save(os.path.join(savepath, f'min_{wd}_{wz}.npy'), min_out_all)

    out_all1 = (out_all1 - min_out_all) / (max_out_all - min_out_all)
    out_all1 = np.around(out_all1 * (wz - 1)).astype(int)
    numlenmax = 0
    DataLink = np.full([wz for i in range(wd)], {})
    for idx1, ix1 in tqdm(enumerate(out_all1), total=len(out_all1), leave=True):
        ix1str = ','.join(str(j) for j in ix1)
        if DataLink[eval(ix1str)]:
            DataLink[eval(ix1str)]['address'].append(idx1)
        else:
            DataLink[eval(ix1str)] = {'address': [idx1], 'move': []}

        # print(DataLinked1[out_all[i][0]][out_all[i][1]][out_all[i][2]][out_all[i][3]][out_all[i][4]])

        if len(DataLink[eval(ix1str)]['address']) > numlenmax:
            numlenmax = len(DataLink[eval(ix1str)]['address'])
    print(numlenmax)
    np.save(os.path.join(savepath, f'inference0424_{wd}_{wz}.npy'), DataLink)
    print(savepath, 'finish')
