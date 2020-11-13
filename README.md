## 利用Pytorch做比特币金融量化预测
### 使用说明
1. 获取比特币历史价格（推荐15m线，data文件夹已包含），参考[地址](https://github.com/Tlntin/open_block_api)

2. 修改config.py的信息，注意修改batch_size与cpu_num，前者根据你的显卡显存，后者根据你的处理器性能。

3. 计算vpin系数，运行tools/computer_target.py，将会生成一个tar.gz文件，里面包含了任意时刻计算的vpin历史数据。

4. 修改config.py的分类类别，默认分成5类。如果你要改成其它类别，需要修改y的分类依据，如果不修改可以跳过5、6条。

5. 重新制定y的分类依据，暂时注释掉tools/dataset.py 86~95行，97行，这样就可以直接获取到y的原始信息。

6. 运行tools.generate_y_classify.py，统计y的分布，生成分类函数，然后替换tools.dataset.py下classify_y函数。

7. 运行train_model.py即可。

### 注意事项
1. 目前5分类效果不佳，在加入$\lambda$防止过拟合后，准确率目前只有30~50%，3分类可能效果更好一些。

2. 后期尝试加入其它金融指标，测试是否能够获得更好的分类效果。