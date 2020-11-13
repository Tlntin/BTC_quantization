from config import Config
import time
import torch
import copy
import os
from torch import optim, nn
from tools.model import resnet18
from tools.dataset import MyDataset
from tools.utils import clip_gradient, save_checkpoint
from tqdm import tqdm

config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


def train_model(dataset, train_index, valid_index,  model, criterion, optimizer):
    since = time.time()
    best_loss = 10
    best_acc = 0
    epochs_since_improvement = 0
    start_epoch = 0
    # 如果存在上次的训练记录
    if os.path.exists(config.checkpoint_file):
        checkpoint = torch.load(config.checkpoint_file)
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer = checkpoint['optimizer']

    batch_length = len(train_index) // config.batch_size - 1
    print('=' * 20)
    print(' ---- training batch -----')
    print('=' * 20)
    train_loss_result = []
    valid_loss_result = []
    train_acc_result = []
    valid_acc_result = []
    for epoch in range(start_epoch, config.epochs):
        # --- 训练模式 --- #
        model.train()
        running_train_loss = 0
        running_train_correct = 0
        running_train_num = 0
        for step, (x, y) in tqdm(enumerate(dataset.get_batch_data(train_index)), total=batch_length):
            running_train_num += x.shape[0]  # 对x进行计数
            x = torch.from_numpy(x).to(device)
            y = torch.from_numpy(y).to(device)
            outputs = model(x)
            y_pred = torch.argmax(outputs, 1)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            clip_gradient(optimizer, 1.0)  # 切割梯度
            optimizer.step()
            running_train_loss += loss.item() * x.size(0)
            running_train_correct += torch.sum(y_pred == y.data)
        epoch_train_loss = running_train_loss / running_train_num
        epoch_train_acc = running_train_correct.double() / running_train_num
        train_loss_result.append(epoch_train_loss)
        train_acc_result.append(epoch_train_acc)
        # ---  开始验证模型 --- #
        epoch_valid_loss, epoch_valid_acc = valid_model(dataset, valid_index, model, criterion)
        valid_loss_result.append(epoch_valid_loss)
        valid_acc_result.append(epoch_valid_acc)
        print('epoch: {} / {} train_loss:{:.4f} train_acc:{:.4f}; valid_loss:{:.4f} valid_acc:{:.4f}'.format(epoch + 1,
              config.epochs, epoch_train_loss, epoch_train_acc, epoch_valid_loss, epoch_valid_acc))

        # -- 保存最佳模型，以valid loss为准 -- #
        is_best = epoch_valid_loss < best_loss
        best_acc = max(epoch_valid_acc, best_acc)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            best_loss = epoch_valid_loss
            epochs_since_improvement = 0
        time_elapsed = time.time() - since  # 计算时间间隔
        print('当前训练共用时{:.0f}分{:.0f}秒'.format(time_elapsed // 60, time_elapsed % 60))
        print('目前最高的准确率为：{}'.format(best_acc))
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_acc, is_best)
    return model, train_loss_result, valid_loss_result, train_acc_result, valid_acc_result


def valid_model(dataset, valid_index, model, criterion):
    # --- 验证模式 --- #
    running_valid_loss = 0
    running_valid_correct = 0
    running_valid_num = 0
    batch_length = len(valid_index) // config.batch_size - 1
    print('=' * 20)
    print(' ---- predicting batch -----')
    print('=' * 20)
    model.eval()  # 开始验证模式
    with torch.no_grad():  # 省掉计算图
        for x, y in tqdm(dataset.get_batch_data(valid_index), total=batch_length):
            running_valid_num += x.shape[0]
            x = torch.from_numpy(x).to(device)
            y = torch.from_numpy(y).to(device)
            outputs = model(x)
            y_pred = outputs.argmax(dim=1)
            loss = criterion(outputs, y)
            running_valid_loss += loss.item() * x.size(0)
            running_valid_correct += torch.sum(y_pred == y.data)
    epoch_valid_loss = running_valid_loss / running_valid_num
    epoch_valid_acc = running_valid_correct.double() / running_valid_num
    return epoch_valid_loss, epoch_valid_acc


if __name__ == '__main__':
    model1 = resnet18()
    optimizer1 = optim.Adam(model1.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-09, weight_decay=0.0002)
    model1 = model1.to(device)
    criterion1 = nn.CrossEntropyLoss().to(device)
    dataset1 = MyDataset()
    train_df1, valid_df1, test_df1 = dataset1.split_btc_batch()
    train_model(dataset1, train_df1, valid_df1, model1, criterion1, optimizer1)
