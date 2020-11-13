import os


class Config(object):
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(parent_dir, 'data')
    temp_dir = os.path.join(data_dir, 'temp')
    btc_path = os.path.join(data_dir, 'btc.gz')  # 比特币文件所在路径
    sequence_num = 256  # 单个数据拥有的数据,4*46也就是46小时
    # sequence_interval = 4  # 每隔4个取一个
    sequence_pre_num = 16  # 预测时间段为4h(实际就是4*4, 因为已经包含了间隔)
    target_csv_path = os.path.join(data_dir, 'target_{}.csv'.format(sequence_num))  # target指标所在路径，csv文件
    target_gzip_path = os.path.join(data_dir, 'target_{}.gz'.format(sequence_num))  # target指标所在路径，压缩文件
    thread_num = 64  # 线程数量
    cpu_num = 8  # 处理器线程，根据你的电脑实际写
    classify_num = 5  # 分类类别
    batch_size = 100
    epochs = 400
    model_dir = os.path.join(data_dir, 'model')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    checkpoint_file = os.path.join(model_dir, 'checkpoint.pth.tar')
    best_checkpoint_file = os.path.join(model_dir, 'best_checkpoint.pth.tar')


if not os.path.exists(Config.data_dir):
    os.mkdir(Config.data_dir)

if not os.path.exists(Config.temp_dir):
    os.mkdir(Config.temp_dir)
