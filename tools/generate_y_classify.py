from tqdm import tqdm
from config import Config
import numpy as np
from tools.dataset import MyDataset


config = Config()
dataset = MyDataset()
train, valid, test = dataset.split_btc_batch()
y_list = []
print('开始读取训练集数据')
for x, y in tqdm(dataset.get_batch_data(train), total=len(train)):
    y_list.extend(y)

print('开始读取验证集数据')
for x, y in tqdm(dataset.get_batch_data(valid), total=len(valid)):
    y_list.extend(y)

print('开始读取测试集数据')
for x, y in tqdm(dataset.get_batch_data(test), total=len(test)):
    y_list.extend(y)

y_list.sort()  # 对y取值进行排序
for i in range(config.classify_num):
    start = int(len(y_list) / config.classify_num * i)
    end = int(len(y_list) / config.classify_num * (i+1))
    if i == 0:
        print('if y < {}:'.format(round(y_list[end], 2)))
    elif i < config.classify_num - 1:
        print('if {} <= y < {}:'.format(round(y_list[start], 2), round(y_list[end], 2)))

    else:
        print('if y >= {}:'.format(y_list[start]))
    print('\treturn {}'.format(i))