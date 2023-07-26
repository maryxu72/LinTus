import numpy as np
import pickle
import os


f = open('train_data.csv', 'r', encoding='utf-8')
with open("train_data_index.id", 'rb') as fp:
    offsets = pickle.load(fp)

fr = open('./1_100000000out/1000.txt', 'r')
l1 = eval(fr.read())
fr.close()

ll1 = l1['idx']

DataLink = np.load('./1_100000000out/inference0617_1_100000000.npy', allow_pickle=True)
savepath = './1_100000000out/' + '101'
for ix1 in ll1:
    ix1str = ','.join(str(j) for j in ix1)
    Dk = DataLink[eval(ix1str)]['address']

    if not os.path.exists(savepath):
        os.mkdir(savepath)
    with open(savepath+'/'+ix1str+'.txt', 'w') as ff:
        for i1 in Dk:
            f.seek(offsets[i1], 0)
            s1 = f.readline()
            s1 = s1.split(',')
            s1 = ','.join(s1[1:])
            s1 = s1.replace('\n', '').strip('"')
            ff.write(s1+'\n')
