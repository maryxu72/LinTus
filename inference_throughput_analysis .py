import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import random
from einops import rearrange, repeat
from tqdm import tqdm
from sklearn.decomposition import PCA

f = open('train_data.csv', 'r', encoding='utf-8')
with open("train_data_index.id", 'rb') as fp:
    offsets = pickle.load(fp)
df_encode = pd.read_csv('chars_encode.csv', index_col=0)


def get_encode_tenser(x):
    encode_1 = [eval(df_encode[i][0]) for i in x]
    mm = np.array(encode_1)
    if mm.ndim != 2:
        mm = np.zeros((1, 7))

    random_list = 550 - len(mm)
    m = cat_np(mm, random_list)
    my_tensor = torch.tensor(m)

    return my_tensor


def get_str(idx, offset2):
    f.seek(offset2[idx], 0)
    l1 = f.readline()
    l1 = l1.split(',')
    l1 = ','.join(l1[1:])
    l1 = l1.replace('\n', '').strip('"')
    l1 = list(l1)
    l1 = l1[:550]
    # l1.insert(0, '&')
    # l1.append('&1')
    return l1


def get_random(total_amount, quantities):
    
    amount_list = []
    person_num = quantities
    cur_total_amount = total_amount
    for _ in range(quantities - 1):
        amount = random.randint(0, cur_total_amount // person_num * 2)
       
        cur_total_amount -= amount
        person_num -= 1
        amount_list.append(amount)
    amount_list.append(cur_total_amount)
    # print(sum(amount_list))
    return amount_list


def cat_np(m, random_list):
   
    # if len(m) > 20:
    #     change_num = 10 
    #     m[np.random.choice(len(m), change_num, replace=False)] = np.array(
    #         [np.random.randint(0, 2, size=7) for i in range(change_num)])

    m2 = np.zeros((random_list, 7))

    m = np.append(m, m2, axis=0)
    return m


class Mydata(Dataset):
    def __init__(self, offset, ):
        self.offset = offset

    def __getitem__(self, idx):
        l1 = get_str(idx, self.offset)
        t1 = get_encode_tenser(l1)
        # train_data_final = train_data_final.to(t.float32)
        # train_data_final = train_data_final.detach()
        return t1.float()

    def __len__(self):
        return len(self.offset)  # 826681*100


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):

        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask:
           
            attention = attention.masked_fill_(attn_mask, -np.inf)
        # 计算softmax
        attention = self.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        # 和V做点积
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=256, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(self.dim_per_head * num_heads, model_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.layer_norm = nn.LayerNorm(model_dim)
        self.re = nn.ReLU(inplace=True)

    def forward(self, key, value, query, attn_mask=None):
       
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        # input(key.size())
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)
        # input(context)
        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)
        output = self.re(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention


class Trans(nn.Module):

    def __init__(self):
        super(Trans, self).__init__()
        self.ModelList = nn.ModuleList([MultiHeadAttention() for _ in range(6)])
        self.linear_out1 = nn.Linear(256, 128)
        self.linear_out2 = nn.Linear(128, 64)
        self.sigmoid = nn.Sigmoid()
        self.re = nn.ReLU(inplace=True)
        self.poem = nn.Parameter(torch.randn(1, 551, 256), requires_grad=True)
        self.class_token = nn.Parameter(torch.randn(1, 1, 256), requires_grad=True)
        self.linear_k = nn.Linear(7, 256)
        # self.linear_q = nn.Linear(16, 512)
        # self.linear_v = nn.Linear(16, 512)

    def forward(self, x):
        x = self.linear_k(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.class_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.poem + x
        # xq = self.linear_q(x)
        # xv = self.linear_v(x)
        x, attention = self.ModelList[0](x, x, x)
        for i in range(1, len(self.ModelList)):
            x, attention = self.ModelList[i](x, x, x)
        # vectorization

        x = x[:, 0, :]
        x = x.view(-1, 256)
        x = self.linear_out1(x)
        x = self.re(x)
        x = self.linear_out2(x)

        # linear and sigmoid
        x = self.sigmoid(x)

        return x, attention


class data_unit():
    def __init__(self):
        self.address = []
        self.move = []


def make_array(*d, ):
    """创建多为数据链表"""
    return [make_array(*d[1:]) for _ in range(d[0])] if d else {'address': [], 'move': []}


if __name__ == '__main__':

    model = Trans().cuda()
    model = torch.load('model_demo2.pth')


    batch_size = 400
    epoch = 128

    train_times = 0
    out_all = np.zeros((1, 64))
    # min_all_loss = 99999
    for i in range(epoch):
        start = datetime.now()
        all_loss = 0
        l11 = len(offsets)
        rate1 = 128
        i1 = i % rate1
        if i1 == rate1 - 1:
            offset1 = offsets[l11 // rate1 * i1:]
        else:
            offset1 = offsets[l11 // rate1 * i1:l11 // rate1 * (i1 + 1)]
        # 处理20000条
        offset1 = offset1[:20000]
        dataset_train = Mydata(
            offset=offset1,
        )

       
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=20,
                                      drop_last=False)
       
        wd = 1
        wz = 100000000
        savepath = f'{wd}_{wz}out'
        max_out_all = np.load(os.path.join(savepath, f'max_{wd}_{wz}.npy'))

        min_out_all = np.load(os.path.join(savepath, f'min_{wd}_{wz}.npy'))

        with open(os.path.join(savepath, 'pca.pkl'), 'rb') as f:
            pca = pickle.load(f)

        out_all_s = np.zeros((1, 64))
        batch_times = 0
        min1_time = timedelta(minutes=1)
        for idx, data, in tqdm(enumerate(dataloader_train), total=len(dataloader_train), leave=True):
            s_time = datetime.now()
            data = data.cuda()
            out, _ = model(data)
            out1 = out.view(data.size()[0], 64).detach().cpu().numpy()
            out1 = pca.transform(out1)
            out1 = (out1 - min_out_all) / (max_out_all - min_out_all)
            out1 = np.around(out1 * (wz - 1)).astype(int)
            f_time = datetime.now()
            cost_time = cost_time + f_time - s_time if 'cost_time' in vars() else f_time - s_time
            batch_times += 1
        print('cost_time: ', cost_time)
        tm = batch_times * batch_size / (cost_time / min1_time)
        print('throughput/min: ', tm)
        throughput_per_min = throughput_per_min + tm if 'throughput_per_min' in vars() else tm
        print('average throughput/min: ', throughput_per_min / (i + 1))

        with open('吞吐量计算.txt', 'a') as ttl:
            ttl.write(f'epoch:{i+1},\ncost_time: {cost_time},\nthroughput/min:{tm},\naverage throughput/min:{throughput_per_min / (i + 1)}\n-------------------------------------------\n')
        del cost_time




