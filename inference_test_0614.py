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


# def get_str(idx, offset2):
#     f.seek(offset2[idx], 0)
#     l1 = f.readline()
#     l1 = l1.split(',')
#     l1 = ','.join(l1[1:])
#     l1 = l1.replace('\n', '').strip('"')
#     l1 = list(l1)
#     l1 = l1[:550]
#     # l1.insert(0, '&')
#     # l1.append('&1')
#     return l1


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


# class Mydata(Dataset):
#     def __init__(self, offset, ):
#         self.offset = offset
#
#     def __getitem__(self, idx):
#         l1 = get_str(idx, self.offset)
#         t1 = get_encode_tenser(l1)
#         # train_data_final = train_data_final.to(t.float32)
#         # train_data_final = train_data_final.detach()
#         return t1.float()
#
#     def __len__(self):
#         return len(self.offset)  # 826681*100


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


if __name__ == '__main__':
    # y = torch.rand(512, 16)
    # y__= torch.rand(512, 16)
    # input(torch.matmul(y.view(y__.size(0), 16), y__.view(16,y__.size(0))).size())
    # DataLink = np.full([15, 15, 15, 15, 15, 15, 15], {})
    model = Trans().cuda()
    model = torch.load('model_demo2.pth')
    # 设置参数
    wd = 1
    wz = 100000000
    savepath = f'{wd}_{wz}out'
    max_out_all = np.load(os.path.join(savepath, f'1max_{wd}_{wz}.npy'))

    min_out_all = np.load(os.path.join(savepath, f'1min_{wd}_{wz}.npy'))

    with open(os.path.join(savepath, '1pca.pkl'), 'rb') as f1:
        pca = pickle.load(f1)
   
    DataLink = np.load(os.path.join(savepath, 'inference0619_1_100000000.npy'), allow_pickle=True)

    # model = torch.nn.DataParallel(model, device_ids=[0,1,2])
    # ll2 = "Angels & Airwaves continues forward with their 80s-tinged blending of punk vocals, anthemic and chest-beating U2 choruses, and the guitar shadings and conceptual nature of classic Pink Floyd. All in all, some of my favorite musical touchstones that have been part of the soundtrack to my life.""Call To Arms"" opens in classic AvA style, widescreen and epic, beautiful chiming guitars and pulsing keyboards building up into a huge song with big choruses and gorgeous Hammond organ.""Everything's Magic"" has become sort've ""our song"" between my girlfriend and I. I heard it for the first time just days before we met, and it was played quite a bit in the car during our first few months. Everytime we hear it come on in a department store or the mall, we give eachother this look and a smile that is priceless. There's a nice pop-punk quality to it (which is different from anything on their first album), although the chorus is a pure adrenaline rush reminescent of early Asia.""Love Like Rockets"" is another ""classic AvA"" song, with those trademark sequenced keyboards and drums. The intro is perfectly epic, utilising sound samples of President Eisenhower, astronauts, and various radio chatter and feedback that brings to mind the Space Race of the 50's and 60's. Tom seems to be making a parallel between t"
    idxs = [322084, 447468, 1497390, 2106120, 3244857, 3498935, 5590217, 57602117, 57890883, 58155008, 58535982]
    for idx in idxs:
        # 载入数据
        f.seek(offsets[idx], 0)
        l1 = f.readline()
        l1 = l1.split(',')
        l1 = ','.join(l1[1:])

        l1 = "The Killer Wore Leather is set deep in the heart of the BDSM subculture, but you don't need to be kinky to appreciate this book. It probably helps, but if you're part of any subculture that looks weird to outsiders (the SCA, SF fandom, Civil War re-enactment...or as another reviewer said, soccer moms), you'll be able to appreciate the cliques, the in-fighting, and yet the instinct to huddle together and defend against attacks from outside. It's all rather over the top, and yet I've met the most outrageous of these characters at BDSM events.And in the end, for all the humor, and the fun cozy-mystery vibe, you do f"
        # 处理载入数据
        l1 = l1.replace('\n', '').strip('"')
        l1 = list(l1)
        l1 = l1[:550]
        t1 = get_encode_tenser(l1)
        data = t1.float()
        data = torch.unsqueeze(data, dim=0)
        out = infer_batch(model, data)
        # data = data.cuda()
        #
        # out, _ = model(data)
        out1 = out.view(data.size()[0], 64).detach().cpu().numpy()

        # ll1 = "I definitely have liked this knife whose high quality steel holds an edge like no other I have had for the money. Definitely recommend it.  Seriously, compare the steel in this knife to comparable Gerber.Ok, let me make this more informative... I also like the clip on this smaller-sized knife.  Other clips can grab on things (e.g., upholstery, overalls) and cause you to lose the knife as I have experienced it with other knives.  This one seems much less prone to this behavior."
        #
        # ll1 = ll1.replace('\n', '').strip('"')
        # l1 = list(ll1)
        # l1 = l1[:550]
        # t1 = get_encode_tenser(l1)
        # data = t1.float()
        # data = torch.unsqueeze(data, dim=0)
        # data = data.cuda()
        # out, _ = model(data)
        # out2 = out.view(data.size()[0], 64).detach().cpu().numpy()
        #
        # out3 = out1 - out2
        out1 = pca.transform(out1)
        out1 = (out1 - min_out_all) / (max_out_all - min_out_all)
        out1 = np.around(out1 * (wz - 1)).astype(int)

        ix1str = ','.join(str(j) for j in out1[0])
        if DataLink[eval(ix1str)]:
            print(DataLink[eval(ix1str)]['address'])
            for idx1 in DataLink[eval(ix1str)]['address']:
                f.seek(offsets[idx1], 0)
                l11 = f.readline()
                l11 = l11.split(',')
                l11 = ','.join(l11[1:])
                l11 = l11.replace('\n', '').strip('"')
                print(l11)
        else:
            # DataLink[eval(ix1str)] = {'address': [1], 'move': []}
            print(ix1str, 'in DataLink is empty')

