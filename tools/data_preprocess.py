from tqdm import tqdm
import pandas as pd
from config import Config as config
import numpy as np
import talib
import time
import os
import random
import pickle


class DataPreProcess(object):
    """
    数据预处理过程
    """
    def __init__(self):
        # 读取原始csv文件
        if not os.path.exists(config.btc_gz_path):
            btc_df = pd.read_csv(config.btc_csv_path)
            self.btc_df = self.get_all_target(btc_df)
            self.get_y_label()  # 计算y的值
            self.btc_df.to_csv(config.btc_gz_path, index=False, compression='gzip')
        else:
            self.btc_df = pd.read_csv(config.btc_gz_path)
        print('开始读取target数据,预计用时5秒')
        start_time = time.time()
        if os.path.exists(config.target_csv_path):
            self.target_df = pd.read_csv(config.target_csv_path, index_col=0)
        else:
            self.target_df = pd.read_csv(config.target_gzip_path, index_col=0)
        end_time = time.time()
        print('target数据读取完成,用时{:.2f}秒'.format(end_time - start_time))

    @staticmethod
    def get_all_target(df1):
        """
        计算除了vpin外的所有指标
        :param
        """
        # 调用talib计算MACD指标
        df1['macd'], df1['macd_signal'], df1['macd_hist'] = talib.MACD(df1.close.values, fastperiod=12, slowperiod=26,
                                                                       signalperiod=9)
        # 调用talib计算rsi与动量
        df1['rsi'] = talib.RSI(df1.close.values, timeperiod=12)  # RSI的天数一般是6、12、24
        df1['mom'] = talib.MOM(df1.close.values, timeperiod=5)
        # 布林带
        df1['upperband'], df1['middleband'], df1['lowerband'] = talib.BBANDS(df1.close.values, timeperiod=5, nbdevup=2,
                                                                             nbdevdn=2, matype=0)

        # 希尔伯特变换
        df1['line1'] = talib.HT_TRENDLINE(df1.close.values)
        # KAMA考夫曼的自适应移动平均线
        df1['kama'] = talib.KAMA(df1.close.values, timeperiod=30)
        # SAR抛物线指标
        df1['sar'] = talib.SAR(df1.high.values, df1.low.values, acceleration=0, maximum=0)
        # CCI指标
        df1['cci'] = talib.CCI(df1.high.values, df1.low.values, df1.close.values, timeperiod=14)
        # KDJ中的KD指标
        df1['slowk'], df1['slowd'] = talib.STOCH(df1.high.values, df1.low.values, df1.close.values,
                                                 fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3,
                                                 slowd_matype=0)
        # ULTOSC终极波动指标
        # 一种多方位功能的指标，除了趋势确认及超买超卖方面的作用之外，它的“突破”讯号不仅可以提供最适当的交易时机之外，更可以进一步加强指标的可靠度。
        df1['ultosc'] = talib.ULTOSC(df1.high.values, df1.low.values, df1.close.values, timeperiod1=7, timeperiod2=14,
                                     timeperiod3=28)
        # WILLR威廉指标
        # 市场处于超买还是超卖状态。股票投资分析方法主要有如下三种：基本分析、技术分析、演化分析。在实际应用中，它们既相互联系，又有重要区别。
        df1['wtllr'] = talib.WILLR(df1.high.values, df1.low.values, df1.close.values, timeperiod=14)
        # AD累积/派发线
        # 平衡交易量指标，以当日的收盘价位来估算成交流量，用于估定一段时间内该证券累积的资金流量。
        df1['ad'] = talib.AD(df1.high.values, df1.low.values, df1.close.values, df1.volume.values)
        # ADOSC震荡指标
        df1['adosc'] = talib.ADOSC(df1.high.values, df1.low.values, df1.close.values, df1.volume.values, fastperiod=3,
                                   slowperiod=10)
        # OBV能量潮
        # 通过统计成交量变动的趋势推测股价趋势
        df1['obv'] = talib.OBV(df1.close.values, df1.volume.values)
        # ATR真实波动幅度均值
        # 以 N 天的指数移动平均数平均後的交易波动幅度。
        df1['atr'] = talib.ATR(df1.high.values, df1.low.values, df1.close.values, timeperiod=14)

        # HT_DCPERIOD希尔伯特变换-主导周期
        # 将价格作为信息信号，计算价格处在的周期的位置，作为择时的依据。
        df1['line2'] = talib.HT_DCPERIOD(df1.close.values)

        # HT_DCPHASE希尔伯特变换-主导循环阶段
        df1['line3'] = talib.HT_DCPHASE(df1.close.values)

        # HT_ PHASOR希尔伯特变换-希尔伯特变换相量分量
        df1['inphase'], df1['quadrature'] = talib.HT_PHASOR(df1.close.values)

        # HT_ SINE希尔伯特变换-正弦波
        df1['sine'], df1['leadsine'] = talib.HT_SINE(df1.close.values)
        return df1

    def get_y_label(self):
        """
        计算y的标签原始标签值
        :param
        """
        print('开始计算y的标签值')
        result_y = []
        for i in tqdm(range(len(self.btc_df) - 4)):
            y = np.round((self.btc_df.loc[i + 4, 'close'] / self.btc_df.loc[i, 'close'] - 1) * 100, 4)
            result_y.append(y)
        result_y.extend([np.nan] * 4)
        self.btc_df['label'] = result_y
        print('y的标签值计算完成')

    def filter_btc(self):
        """
        筛选索引
        :param
        """
        total_df = self.btc_df.iloc[76632: len(self.btc_df) - config.sequence_num].copy()  # 开始时间2016-3-16
        index_list = []
        y_list = []
        print('开始筛选数据集')
        for index in tqdm(total_df.index.tolist()):
            y = self.btc_df.iloc[index, 32]
            date_time = self.btc_df.iloc[index, 0]
            target_list = self.target_df.loc[date_time].values.tolist()
            now_target = target_list[config.sequence_num-1]
            if now_target > 0.4 and now_target > np.mean(target_list[int(len(target_list) * 0.5):]):
                index_list.append(index)
                y_list.append(y)
        print('数据集筛选完成')
        return index_list, y_list

    def get_y_classify(self):
        """
        利用筛选后的y，生成y的分类函数
        :param
        """
        index_list, y_list = self.filter_btc()
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

    def get_split_index(self):
        """
        生成安装等比例生成训练集、验证集、测试集的索引，防止样本不均衡
        :param
        """
        # 获取筛选后的索引
        index_list, y_list = self.filter_btc()
        result_data = []
        # 对y进行分类
        for index in index_list:
            raw_y = self.btc_df.loc[index, 'label']
            y = classify_y(raw_y)
            result_data.append([index, y])
        # 正式切割数据集
        df2 = pd.DataFrame(result_data, columns=['index2', 'label'])
        train_index = []
        valid_index = []
        test_index = []
        for label in range(config.classify_num):
            temp_df = df2[df2.label == label]
            temp_index = temp_df.index2.tolist()
            random.shuffle(temp_index)  # 随机打散
            train_length = int(len(temp_index) * 0.6)
            valid_length = int(len(temp_index) * 0.2)
            train_index.extend(temp_index[:train_length])
            valid_index.extend(temp_index[train_length: train_length + valid_length])
            test_index.extend(temp_index[train_length + valid_length:])
        # 写入到pickle
        with open(os.path.join(config.pickle_dir, 'train.pkl'), 'wb') as f:
            pickle.dump(train_index, f)
        with open(os.path.join(config.pickle_dir, 'valid.pkl'), 'wb') as f:
            pickle.dump(valid_index, f)
        with open(os.path.join(config.pickle_dir, 'test.pkl'), 'wb') as f:
            pickle.dump(test_index, f)


def classify_y(y):
    """
    对y的值大小进行归类,目前为5分类,该分类方法由上面的get_y_classify生成
    """
    if y < -0.37:
        return 0
    if -0.37 <= y < -0.07:
        return 1
    if -0.07 <= y < 0.12:
        return 2
    if 0.12 <= y < 0.44:
        return 3
    if y >= 0.4377:
        return 4


if __name__ == '__main__':
    data = DataPreProcess()
    # 第一步：筛选后并生成classify_y函数
    # data.get_y_classify()
    # 第二步：筛选并切割索引示范
    data.get_split_index()
