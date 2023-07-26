import pandas as pd
import numpy as np
import re
from datetime import datetime
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import random
from einops import rearrange, repeat
from tqdm import tqdm
from sklearn.decomposition import PCA
import json

f = open('train_data.csv', 'r', encoding='utf-8')
with open("train_data_index.id", 'rb') as fp:
    offsets = pickle.load(fp)
# df_encode = pd.read_csv('chars_encode.csv', index_col=0)
with open("chars_encode.pickle", "rb") as file:
    encode_dict = pickle.load(file)
# df_sample = pd.read_csv('sample_data1.csv', index_col=0)
with open('./100str/100str_original.json', 'r') as file:
    # 从文件中读取 JSON 字符串
    json_str = file.read()
    str_original = json.loads(json_str)

with open('./100str/100str_preposition.json', 'r') as file:
    # 从文件中读取 JSON 字符串
    json_str = file.read()
    str_preposition = json.loads(json_str)

with open('./100str/100str_keywordchanged.json', 'r') as file:
    # 从文件中读取 JSON 字符串
    json_str = file.read()
    str_keywordchanged = json.loads(json_str)

dfc = pd.read_csv('chars0119.csv', index_col=0)
cc1 = dfc.chars
cc2 = cc1.to_list()


def get_encode_tenser(x):
    encode_1 = [encode_dict[i] for i in x if i in cc2]
    mm = np.array(encode_1)
    if mm.ndim != 2:
        mm = np.zeros((1, 7))

    random_list = 550 - len(mm)
    m = cat_np(mm, random_list)
    my_tensor = torch.tensor(m)

    return my_tensor


def get_str(idx):
    # l2 = re.findall(r"[\w']+|[.,!?;:']", df_sample.iloc[idx, 0].strip("[]").strip("'"))
    l2 = list()
    l2 = [s.lower() for s in l2 if isinstance(s, str) == True]
    return l2


def get_str2(idx, df_sample1):
    txt1 = df_sample1.iloc[idx, 0].strip("[]").strip("'")
    l1 = list(txt1)
    l1 = l1[:550]
    return l1


def get_random(total_amount, quantities):
    """将输入值随机分割"""
    amount_list = []
    person_num = quantities
    cur_total_amount = total_amount
    for _ in range(quantities - 1):
        amount = random.randint(0, cur_total_amount // person_num * 2)
        # 每次减去当前随机值，用剩余值进行下次随机获取
        cur_total_amount -= amount
        person_num -= 1
        amount_list.append(amount)
    amount_list.append(cur_total_amount)
    # print(sum(amount_list))
    return amount_list


def cat_np(m, random_list):
    """数据随机前后补零增强"""
    # if len(m) > 20:
    #     change_num = 10  # 设置修改字母数
    #     m[np.random.choice(len(m), change_num, replace=False)] = np.array(
    #         [np.random.randint(0, 2, size=7) for i in range(change_num)])

    m2 = np.zeros((random_list, 7))

    m = np.append(m, m2, axis=0)
    return m


class Mydata(Dataset):
    def __init__(self, df_sample1, ):
        self.df_sample = df_sample1

    def __getitem__(self, idx):
        l1 = get_str2(idx, self.df_sample)
        t1 = get_encode_tenser(l1)
        # train_data_final = train_data_final.to(t.float32)
        # train_data_final = train_data_final.detach()
        return t1.float()

    def __len__(self):
        return len(self.df_sample)


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
            # 给需要mask的地方设置一个负无穷
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
        # multi-head attention之后需要做layer norm
        self.layer_norm = nn.LayerNorm(model_dim)
        self.re = nn.ReLU(inplace=True)

    def forward(self, key, value, query, attn_mask=None):
        # 残差连接
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


def infer_batch(model1, data1):
    """以batch为单位推理"""
    ld1 = len(data1)
    if ld1 != 400:
        ld2 = 400 - ld1
        data2 = torch.randint(0, 2, (ld2, 550, 7))
        data1 = torch.cat((data1, data2), dim=0)
    data1 = data1.cuda()

    out_1, _ = model1(data1)
    out_1 = out_1[:ld1]
    return out_1


def cu_cos(s1, s2):
    li1 = [i for i in s1 if i in cc2]
    l2 = [i for i in s2 if i in cc2]
    ll = 550 - len(li1)
    li1 = li1 + (['kong'] * ll)
    ll = 550 - len(l2)
    l2 = l2 + (['kong'] * ll)
    e1 = np.array([encode_dict[i] for i in li1])
    e2 = np.array([encode_dict[i] for i in l2])
    cosi0 = (e1*e2).sum()/(e1.sum()*e2.sum())*1000
    return cosi0


if __name__ == '__main__':

    model = Trans().cuda()
    model = torch.load('model_demo2.pth')
    # 设置参数
    wd = 1
    wz = 100000000
    savepath = f'{wd}_{wz}out'
    max_out_all = np.load(os.path.join(savepath, f'max_{wd}_{wz}.npy'))

    min_out_all = np.load(os.path.join(savepath, f'min_{wd}_{wz}.npy'))

    with open(os.path.join(savepath, 'pca.pkl'), 'rb') as f1:
        pca = pickle.load(f1)
    # 加载数据链表
    DataLink = np.load(os.path.join(savepath, 'inference0424_1_100000000.npy'), allow_pickle=True)


    total_cosi1 = 0
    total_cosi2 = 0
    outs_100 = []
    outs_100_pca = []
    for idx in tqdm(range(len(str_original))):
        # for data in tqdm(dataloader_train, total=len(dataloader_train), leave=True):
        # 载入数据
        # txt1 = df_sample.iloc[idx, 0].strip("[]").strip("'")
        txts = [str_original[idx], str_preposition[idx], str_keywordchanged[idx]]
        outs = []
        outs_pca = []
        # print('查询输入为：', txt1)
        for txt1 in txts:
            l1 = list(txt1)
            l1 = l1[:550]
            t1 = get_encode_tenser(l1)
            data = t1.float()
            data = torch.unsqueeze(data, dim=0)

            out = infer_batch(model, data)
            out1 = out.view(data.size()[0], 64).detach().cpu().numpy()

            outs.append(out1)

            out1 = pca.transform(out1)
            out1 = (out1 - min_out_all) / (max_out_all - min_out_all)
            out1 = np.around(out1 * (wz - 1)).astype(int)

            outs_pca.append(out1[0])

        outs_100.append(outs)
        outs_100_pca.append(outs_pca)
    outs_100 = np.array(outs_100).reshape((100, 3, 64))
    print('original - preposition:', np.absolute(outs_100[:, 0]-outs_100[:, 1]).sum())
    print('original - keywordchanged:', np.absolute(outs_100[:, 0] - outs_100[:, 2]).sum())
    range_100_pre = 0
    range_100_kwc = 0
    for out in outs_100_pca:
        ix1 = int(out[0])
        ix2 = int(out[1])
        ix3 = int(out[2])
        len_range = 0
        if ix1 < ix2:
            for ix in range(ix1, ix2):
               len_range += 1 if DataLink[ix] else 0
        else:
            for ix in range(ix2, ix1):
               len_range += 1 if DataLink[ix] else 0
        range_100_pre += len_range
        len_range = 0
        if ix1 < ix3:
            for ix in range(ix1, ix3):
               len_range += 1 if DataLink[ix] else 0
        else:
            for ix in range(ix3, ix1):
               len_range += 1 if DataLink[ix] else 0
        range_100_kwc += len_range

    print('平均修改介词搜寻有效范围：', range_100_pre / 100)
    print('平均修改关键词搜寻有效范围：', range_100_kwc / 100)
