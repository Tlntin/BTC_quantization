import time
import pandas as pd
import numpy as np
from config import Config
import random
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pickle

config = Config()


class MyDataset(object):
    def __init__(self):
        self.btc_df = pd.read_csv(config.btc_path)
        # print(btc_df.head())
        print('开始读取target数据,预计用时5秒')
        start_time = time.time()
        if os.path.exists(config.target_csv_path):
            self.target_df = pd.read_csv(config.target_csv_path, index_col=0)
        else:
            self.target_df = pd.read_csv(config.target_gzip_path, index_col=0)
        end_time = time.time()
        print('target数据读取完成,用时{:.2f}秒'.format(end_time - start_time))

    def filter_btc(self):
        total_df = self.btc_df.iloc[76632: len(self.btc_df) - config.sequence_num].copy()  # 开始时间2016-3-16
        result_index = []
        print('开始筛选数据集')
        for index in tqdm(total_df.index.tolist()):
            date_time = self.btc_df.iloc[index, 0]
            target_list = self.target_df.loc[date_time].values.tolist()
            now_target = target_list[config.sequence_num-1]
            if now_target > 0.4 and now_target > np.mean(target_list[int(len(target_list) * 0.5):]):
                result_index.append(index)
        print('数据集筛选完成')
        return result_index

    def split_btc_batch(self):
        """
        分割BTC_df数据
        """
        if not os.path.exists(os.path.join(config.data_dir, 'train.pkl')):
            index_list = self.filter_btc()
            random.shuffle(index_list)
            # 分割数据集
            train_length = int(len(index_list) * 0.6)
            valid_length = int(len(index_list) * 0.2)
            train_index = index_list[:train_length]
            valid_index = index_list[train_length: train_length + valid_length]
            test_index = index_list[train_length + valid_length:]
            # 写入数据集
            with open(os.path.join(config.data_dir, 'train.pkl'), 'wb') as f:
                pickle.dump(train_index, f)
            with open(os.path.join(config.data_dir, 'valid.pkl'), 'wb') as f:
                pickle.dump(valid_index, f)
            with open(os.path.join(config.data_dir, 'test.pkl'), 'wb') as f:
                pickle.dump(test_index, f)
        else:
            # 读取文件
            with open(os.path.join(config.data_dir, 'train.pkl'), 'rb') as f:
                train_index = pickle.load(f)
            with open(os.path.join(config.data_dir, 'valid.pkl'), 'rb') as f:
                valid_index = pickle.load(f)
            with open(os.path.join(config.data_dir, 'test.pkl'), 'rb') as f:
                test_index = pickle.load(f)
        print('训练集{}条,验证集{}条,测试集{}条'.format(len(train_index), len(valid_index), len(test_index)))
        return train_index, valid_index, test_index

    def get_batch_data(self, batch_index_list):
        """
        批量获取数据
        :param batch_index_list: batch开始索引构建的列表
        """
        scale = StandardScaler()
        for i in range(0, len(batch_index_list) - config.batch_size, config.batch_size ):
            # 保险起见,减去一个batch_size
            x_batch = []
            y_batch = []
            for size in range(config.batch_size):
                index = batch_index_list[i + size]
                x = self.btc_df.iloc[index - config.sequence_num: index, 1:].astype('float32').values
                y = np.round(self.btc_df.iloc[index + config.sequence_pre_num, 4] /
                             self.btc_df.iloc[index, 4] - 1, 4) * 100
                # 获取额外的数据
                date_time = self.btc_df.iloc[index, 0]
                target_list = self.target_df.loc[date_time].values.astype('float32')
                # 正式合并
                x = np.insert(x, 31, values=target_list, axis=1)  # x修改为31纬数据
                # 标准化
                x = scale.fit_transform(x)
                x_batch.append(x)
                y = self.classify_y(y)
                y = np.array(y)
                y_batch.append(y)
            x_batch, y_batch = np.array(x_batch, dtype='float32'), np.array(y_batch, dtype='int64')
            yield x_batch, y_batch

    @staticmethod
    def classify_y(y):
        """
        对y的值大小进行归类,目前为10分类,该分类基于y的排序后的区间值决定
        """
        if y < -0.64:
            return 0
        if -0.64 <= y < -0.11:
            return 1
        if -0.11 <= y < 0.24:
            return 2
        if 0.24 <= y < 0.84:
            return 3
        if y >= 0.84:
            return 4


if __name__ == '__main__':
    data = MyDataset()
    train, valid, test = data.split_btc_batch()
    x1 = y1 = None
    for x1, y1 in tqdm(data.get_batch_data(train), total=len(train)):
        break
    print(x1.shape)
    print(x1)
    print('===')
    print(y1)
