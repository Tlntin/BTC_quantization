import os
import pandas as pd
import flowrisk as fr
from tqdm import tqdm
import numpy as np
from datetime import datetime
from multiprocessing import Pool
from config import Config


config = Config()
"""
创建日期：2020-11-6
用途：使用多线程计算vpin值（下面简称target值）
注意点：1、为了模拟实盘功能，所以每次计算将保留所有target值
      2、默认保留1000/256条数据，也就是1000/256条target值
"""


class TargetConfig(fr.BulkVPINConfig):
    """
    用于target系数计算前的准备工作
    """
    def __init__(self, date_title: str, price_tile: str, volume_title: str, avg_amount):
        """
        初始化构造函数
        :param date_title: 日期的列名称，如'date'
        :param price_tile: 选取的价格列名称，如：'close'取收盘价
        :param volume_title: 选取的交易量列：如：'volume'
        :param avg_amount: 平均每个篮子最大能放的容量
        """
        self.TIME_BAR_TIME_STAMP_COL_NAME = date_title
        self.TIME_BAR_PRICE_COL_NAME = price_tile
        self.TIME_BAR_VOLUME_COL_NAME = volume_title
        self.BUCKET_MAX_VOLUME = avg_amount
        self.N_TIME_BAR_FOR_INITIALIZATION = 50


def get_target(df2, max_amount):
    """
    此函数用于计算target系数
    :param df2: DataFrame格式
    :param max_amount: 篮子最大能放的容量，取上一年的平均每日交易量除以篮子数量
    :return: 返回vpin列表
    """
    columns_list = ['date_time', 'close', 'volume', max_amount]
    target_list = False
    target_config = TargetConfig(*columns_list)
    try:
        target_estimator = fr.BulkVPIN(target_config)  # 估计target
        target_df = target_estimator.estimate(df2)  # 计算target
        target_list = np.round(target_df.vpin.values, 4).tolist()
    except Exception as err:
        print(err)
        return False
    finally:
        return target_list


def sync_start(df2, start_id, end_id, task=0):
    print('Run task %s (%s)...' % (task, os.getpid()))
    index_list = list(range(config.sequence_num))
    result_list = []
    df_day1 = df2.loc[start_id - 366 * 24 * 4: start_id, :].groupby('day').sum()  # 取当前日期到366天之前的数据，更新数据
    avg_volume = df_day1.iloc[:len(df_day1) - 1, :]['volume'].mean()  # 放弃最后一天的数据
    for i in tqdm(range(start_id, end_id)):
        if i % (24*4) == 0:  # 如果是当日的零时整点
            df_day1 = df2.loc[i - 366 * 24 * 4: i, :].groupby('day').sum()  # 取当前日期到366天之前的数据，更新数据
            avg_volume = df_day1.iloc[:len(df_day1)-1, :]['volume'].mean()  # 放弃最后一天的数据
        target_df = df2.loc[i - config.sequence_num + 1:i, ['date_time', 'close', 'volume']].copy()   # 获取1000/256条数据
        if len(target_df) != config.sequence_num:
            print('=' * 20)
            print('target数据有误')
            print('=' * 20)
        target_df['volume'] = target_df['volume'] + 0.00001  # 防止交易量分母为零
        target_df.index = index_list
        target_list = get_target(target_df, avg_volume / 50)  # 计算target指标
        if bool(target_list):
            if len(target_list) > 1:
                result_list.append(target_list)
            else:
                result_list.append([0] * config.sequence_num)
        else:
            result_list.append([0] * config.sequence_num)
    df_result = pd.DataFrame(result_list)
    df2_copy = df2.loc[start_id: end_id-1, :].copy()  # 复制一份，后面再尝试合并
    df_result.index = df2_copy.index  # 索引一致，方便后期排序以及合并
    df_result.insert(0, 'date_time', df2_copy.date_time)
    file_name = '{}_{}_{}.csv'.format(task, start_id, end_id)
    file_path = os.path.join(config.temp_dir, file_name)
    df_result.to_csv(file_path, index=True, encoding='utf-8-sig')
    print(file_path, '保存成功')


def run():
    """
    正式运行多线程
    """
    df1 = pd.read_csv(os.path.join(config.data_dir, 'BTC.csv'), encoding='utf-8')
    df1['day'] = df1['date_time'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d'))
    interval = int((len(df1) - 76632) / config.thread_num)  # 计算单个线程间隔,  开始时间2016-3-16
    print('=' * 20)
    print('即将开启多线程，当前CPU核心设置为{}个'.format(config.cpu_num))
    print('当前共有{}条数据，共有{}个线程，每个线程处理{}条数据'.format(len(df1), config.thread_num, interval))
    print('=' * 20)
    print('Parent process %s.' % os.getpid())
    p = Pool(config.cpu_num)
    start = 76632
    for i in range(config.thread_num):
        if i < config.thread_num - 1:
            end = start + interval
        else:
            end = len(df1)
        df3 = df1.iloc[start - 366 * 24 * 4: end, :].copy()  # 先复制一份，省的占用资源
        p.apply_async(sync_start, args=(df3, start, end, i))
        start = end
    p.close()
    p.join()
    print('All subprocesses done.')


def merge_df():
    """
    合并temp文件夹的DataFrame文件
    """
    file_list = os.listdir(config.temp_dir)
    print('开始保存文件到csv')
    file_list.sort(key=lambda x: int(x.split('_')[0]))
    for i, file_name in enumerate(tqdm(file_list)):
        # 临时加入,用于去除无用信息
        if i < 13:
            continue
        temp_df = pd.read_csv(os.path.join(config.temp_dir, file_name), index_col=0)
        temp_df.iloc[:, 1:] = temp_df.iloc[:, 1:].astype('float32')  # 转float32省空间
        if i == 13:
            temp_df.to_csv(config.target_csv_path, mode='w+', index=False, encoding='utf-8-sig')
        else:
            temp_df.to_csv(config.target_csv_path, mode='a+', index=False, encoding='utf-8-sig', header=False)
    print("csv文件保存成功")
    df1 = pd.read_csv(config.target_csv_path, encoding='utf-8-sig')
    df1.to_csv(config.target_gzip_path, index=False, encoding='utf-8-sig', compression='gzip')
    print("gzip文件保存成功")


if __name__ == '__main__':
    pass
    # run()
    merge_df()




