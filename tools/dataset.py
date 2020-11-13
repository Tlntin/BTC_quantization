import time
import pandas as pd
import numpy as np
from config import Config
import os
from tqdm import tqdm
from tools.data_preprocess import classify_y
from sklearn.preprocessing import StandardScaler
import pickle
from collections import Counter


config = Config()


class MyDataset(object):
    def __init__(self):
        self.btc_df = pd.read_csv(config.btc_gz_path)
        # print(btc_df.head())
        print('开始读取target数据,预计用时5秒')
        start_time = time.time()
        if os.path.exists(config.target_csv_path):
            self.target_df = pd.read_csv(config.target_csv_path, index_col=0)
        else:
            self.target_df = pd.read_csv(config.target_gzip_path, index_col=0)
        end_time = time.time()
        print('target数据读取完成,用时{:.2f}秒'.format(end_time - start_time))

    @staticmethod
    def split_btc_batch():
        """
        分割BTC_df数据
        """
        if not os.path.exists(os.path.join(config.pickle_dir, 'train.pkl')):
            raise Exception('请运行tools文件夹的data_preprocess.py文件生成相关文件')
        else:
            # 读取文件
            with open(os.path.join(config.pickle_dir, 'train.pkl'), 'rb') as f:
                train_index = pickle.load(f)
            with open(os.path.join(config.pickle_dir, 'valid.pkl'), 'rb') as f:
                valid_index = pickle.load(f)
            with open(os.path.join(config.pickle_dir, 'test.pkl'), 'rb') as f:
                test_index = pickle.load(f)
        print('训练集{}条,验证集{}条,测试集{}条'.format(len(train_index), len(valid_index), len(test_index)))
        return train_index, valid_index, test_index

    def get_batch_data(self, batch_index_list):
        """
        批量获取数据
        :param batch_index_list: batch开始索引构建的列表
        """
        scale = StandardScaler()
        for i in range(0, len(batch_index_list) - config.batch_size, config.batch_size):
            # 保险起见,减去一个batch_size
            x_batch = []
            y_batch = []
            for size in range(config.batch_size):
                index = batch_index_list[i + size]
                x = self.btc_df.iloc[index - config.sequence_num: index, 1: 32].astype('float32').values
                y = self.btc_df.iloc[index, 32]
                y = classify_y(y)  # y 归类
                y = np.array(y)
                y_batch.append(y)
                # 获取额外的数据
                date_time = self.btc_df.iloc[index, 0]
                target_list = self.target_df.loc[date_time].values.astype('float32')
                # 正式合并
                x = np.insert(x, 31, values=target_list, axis=1)  # x修改为31纬数据
                # 标准化
                x = scale.fit_transform(x)
                x_batch.append(x)
            x_batch, y_batch = np.array(x_batch, dtype='float32'), np.array(y_batch, dtype='int64')
            yield x_batch, y_batch


if __name__ == '__main__':
    data = MyDataset()
    train, valid, test = data.split_btc_batch()
    x1 = y1 = None
    y_list1 = []
    for x1, y1 in tqdm(data.get_batch_data(test), total=len(test) // config.batch_size):
        print(x1.shape)
        break
    #     y_list1.extend(y1)
    # dict1 = Counter(y_list1)
    # print(dict1)