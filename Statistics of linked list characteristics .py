import numpy as np
from tqdm import tqdm
import os
# original data
sps = ['1_100000000out', '2_10000out', '3_500out', '4_100out', '5_50out', '9_9out']
fps = ['1_100000000out/inference0619_1_100000000.npy', '2_10000out/inference0424_2_10000.npy',
       '3_500out/inference0424_3_500.npy', '4_100out/inference0424_4_100.npy',
       '5_50out/inference0424_5_50.npy', '9_9out/inference0424_9_9.npy']
# 2000W泛化数据
# sps = ['2000w_1_100000000out', '2000w_2_10000out', '2000w_3_500out', '2000w_4_100out', '2000w_5_50out', '2000w_9_9out']
# fps = ['2000w_1_100000000out/inference0602_1_100000000.npy', '2000w_2_10000out/inference0602_2_10000.npy',
#        '2000w_3_500out/inference0602_3_500.npy', '2000w_4_100out/inference0602_4_100.npy',
#        '2000w_5_50out/inference0602_5_50.npy', '2000w_9_9out/inference0602_9_9.npy']
for fp, save_path in zip(fps, sps):
    l1 = np.load(fp, allow_pickle=True)

    len1 = {'num': 0}
    len10 = {'num': 0, 'idx': []}
    len100 = {'num': 0}
    len1000 = {'num': 0, 'idx': []}
    len10000 = {'num': 0, 'idx': []}
    len_h = {'num': 0, 'idx': []}

    for i, x in tqdm(np.ndenumerate(l1), total=len(l1)):

        if x:
            lx = len(x['address'])
            if lx == 1:
                len1['num'] += 1
            elif 2 <= lx < 10:
                len10['num'] += 1
                len10['idx'].append(i)
            elif 10 <= lx < 100:
                len100['num'] += 1
            elif 100 <= lx < 1000:
                len1000['num'] += 1
                len1000['idx'].append(i)
            elif 1000 <= lx < 10000:
                len10000['num'] += 1
                len10000['idx'].append(i)
            elif 10000 <= lx:
                len_h['num'] += 1
                len_h['idx'].append(i)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with open(os.path.join(save_path, '10.txt'), 'w') as f10:
        f10.write(str(len10))

    with open(os.path.join(save_path, '1000.txt'), 'w') as f1000:
        f1000.write(str(len1000))

    with open(os.path.join(save_path, '10000.txt'), 'w') as f10000:
        f10000.write(str(len10000))

    with open(os.path.join(save_path, 'h.txt'), 'w') as fh:
        fh.write(str(len_h))

    print('The number of linked list addresses with number 1 is:', len1['num'])
    print('The number of linked list addresses with number 2-9 is:', len10['num'])
    print('The number of linked list addresses with number 10-99 is:', len100['num'])
    print('The number of linked list addresses with number 100-999 is:', len1000['num'])
    print('The number of linked list addresses with number 1000-9999 is:', len10000['num'])
    print('The number of linked list addresses with more than 10,000 is:', len_h['num'])

    # 总项数
    total1 = len1['num'] + len10['num'] + len100['num'] + len1000['num'] + len10000['num'] + len_h['num']

    txt1 = [
        f"The number of linked list addresses with number 1 is:{len1['num']}\n",
        f"The number of linked list addresses with number 2-9 is:{len10['num']}\n",
        f"The number of linked list addresses with number 10-99 is:{len100['num']}\n",
        f"The number of linked list addresses with number 100-999 is:{len1000['num']}\n",
        f"The number of linked list addresses with number 1000-9999 is:{len10000['num']}\n",
        f"The number of linked list addresses with more than 10,000 is:{len_h['num']}\n",
        f"The total number of items is:{total1}\n",
        f"Average number per item:{82668056 / total1}"
    ]
    with open(os.path.join(save_path, 'cul.txt'), 'w', encoding='UTF-8') as fc:
        for txt2 in txt1:
            fc.write(str(txt2))
