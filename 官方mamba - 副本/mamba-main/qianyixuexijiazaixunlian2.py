import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import scale
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba import Mamba, MambaConfig
import os
import argparse
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
# 参数设置
# 创建一个ArgumentParser对象，用于解析命令行参数
parser = argparse.ArgumentParser()

# 添加命令行参数
# 添加是否使用CUDA的参数，默认为True
parser.add_argument('--use-cuda', default=True, help='CUDA training.')
# 添加随机种子参数，默认为1
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
# 添加训练轮数参数，默认为200
parser.add_argument('--epochs', type=int, default=600, help='Number of epochs to train.')
# 添加学习率参数，默认为0.01
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
# 添加权重衰减参数，默认为1e-6
parser.add_argument('--wd', type=float, default=1e-6, help='Weight decay (L2 loss on parameters).')
# 添加表示维度参数，默认为32
parser.add_argument('--hidden', type=int, default=32, help='Dimension of representations')
# 添加层数参数，默认为4
parser.add_argument('--layer', type=int, default=4, help='Num of layers')
# 添加任务类型参数，默认为'SOH'
parser.add_argument('--task', type=str, default='SOH', help='RUL or SOH')
# 添加案例类型参数，默认为'B'
parser.add_argument('--case', type=str, default='B', help='A or B')
# 添加预训练标志参数，默认为True
parser.add_argument('--pretrain', action='store_true', default=True, help='Flag to pretrain the model')
# 添加预训练模型路径参数，默认为指定路径
parser.add_argument('--pretrain-model', type=str, default=r'C:\Users\32468\Desktop\mamba双头\mamba双头尝试\迁移权重重重\Model_num599_now_loss0.0003.pth', help='Path to pre-trained model')

# 解析命令行参数
args = parser.parse_args()

# 根据命令行参数设置是否使用CUDA，CUDA需在系统上可用
args.cuda = args.use_cuda and torch.cuda.is_available()

# 设置随机种子
def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


set_seed(args.seed, args.cuda)


# 评估指标
def evaluation_metric(y_test, y_hat):
    MSE = mean_squared_error(y_test, y_hat)
    RMSE = MSE ** 0.5
    MAE = mean_absolute_error(y_test, y_hat)
    R2 = r2_score(y_test, y_hat)
    print('%.4f %.4f %.4f %.4f' % (MSE, RMSE, MAE, R2))


# 定义一个函数用于读取数据
def ReadData(path, xlsx, task):
    """
    读取数据并动态加载对应的 predictions.npy 文件，将其作为新特征添加。
    """
    # 加载 Excel 文件中的数据
    f = os.path.join(path, xlsx)
    data = pd.read_excel(f)
    tf = len(data)
    y = data[task].values  # 提取目标值
    if args.task == 'RUL':
        y = y / tf  # 对目标值进行归一化处理

    # 提取原始特征（去掉 'RUL' 和 'SOH' 列）
    x = data.drop(['RUL', 'SOH'], axis=1).values  # 原始特征（7个）


    # 动态加载对应的 predictions.npy 文件
    # 根据 Excel 文件名加载相应的预测特征文件
    predictions_path = os.path.join(
        r'C:\Users\32468\Desktop\mamba双头\mamba双头尝试\mamba估计的数值',
        f"predictions_{xlsx.split('.')[0].split('_')[1]}.npy"  # 提取文件编号作为预测文件名
    )
    predictions = np.load(predictions_path)  # 加载预测值
    predictions = predictions.reshape(-1, 1)  # 转换为列向量

    # 确保 predictions 的长度与 x 的行数一致
    assert predictions.shape[0] == x.shape[0], f"Predictions file {predictions_path} 的样本数量与数据集不一致！"

    # 将 predictions 作为新特征添加到 x 中
    x = np.hstack((x, predictions))  # 合并特征，现在 x 有 8 列
    x = scale(x)  # 对特征进行归一化处理
    print(f"Feature matrix shape: {x.shape}")
    print(f"Target vector shape: {y.shape}")

    # 设置滑动窗口参数
    X_WINDOW = 10  # 输入窗口长度
    Y_WINDOW = 3   # 输出窗口长度

    # 创建滑动窗口
    X = []
    Y = []
    max_index = len(x) - (X_WINDOW + Y_WINDOW) + 1
    for i in range(max_index):
        x_seq = x[i: i + X_WINDOW]  # shape: (10, number_of_feature_columns)
        y_seq = y[i + X_WINDOW: i + X_WINDOW + Y_WINDOW]  # shape: (3,)

        X.append(x_seq)
        Y.append(y_seq)

    # 将列表转换为 NumPy 数组
    X = np.array(X)  # shape: (num_samples, 10, number_of_feature_columns)
    Y = np.array(Y)  # shape: (num_samples, 3)

    print("Final input X shape:", X.shape)
    print("Final output Y shape:", Y.shape)
    return X, Y


# 定义一个网络结构类，继承自nn.Module
class MambaHead(nn.Module):
    """
    封装Mamba模型并添加一个小的线性层"head"，以产生所需的(B, 3, 2)输出。
    """
    def __init__(self, mamba_model, label_length=3, label_dim=2):
        # 调用父类的构造函数
        super().__init__()
        # 将传入的Mamba模型赋值给实例变量
        self.mamba = mamba_model
        # 设置标签长度，例如3
        self.label_length = label_length
        # 设置标签维度，例如2
        self.label_dim = label_dim
        # 获取Mamba模型的隐藏层大小
        d_model = mamba_model.config.d_model
        # 定义一个线性层，将隐藏层大小映射到标签维度
        self.head = nn.Linear(d_model, label_dim)  # 对每个时间步应用

    def forward(self, x):
        """
        x: 输入数据，形状为(B, seq_length=10, d_model)
        返回: 输出数据，形状为(B, label_length=3, label_dim=2)
        """
        # 1) 将数据通过Mamba模型，输出形状为(B, L, D)，L与输入相同
        mamba_out = self.mamba(x)  # (B, L, D)

        # 2) 取出最后'label_length'个时间步
        out_sliced = mamba_out[:, -self.label_length:, :]  # (B, 3, D)

        # 3) 对每个时间步应用线性层 -> (B, 3, label_dim)
        out_pred = self.head(out_sliced)  # (B, 3, 2)

        # 返回预测结果和切片后的Mamba输出
        return out_pred


# 定义一个带有包装器的网络结构类，继承自nn.Module
class MambaHeadWithWrapper(nn.Module):
    """
    封装Mamba模型并添加一个小的线性层"head"，以产生所需的(B, 3, 2)输出。
    在线性层之前添加了冻结，dropout和sigmoid激活。
    """
    def __init__(self, mamba_model, label_length=3, label_dim=2, dropout_prob=0.5):
        # 调用父类的构造函数
        super().__init__()
        # 将传入的Mamba模型赋值给实例变量
        self.mamba = mamba_model
        # 设置标签长度，例如3
        self.label_length = label_length
        # 设置标签维度，例如2
        self.label_dim = label_dim

        # 冻结Mamba模型的参数，使其在训练过程中不被更新
        for param in self.mamba.parameters():
            param.requires_grad = False

        # 添加dropout层，以减少过拟合
        self.dropout = nn.Dropout(dropout_prob)

        # 添加一个全连接层，输入特征数为7，输出特征数为1
        self.fc = nn.Linear(7, 1)

        # 添加sigmoid激活函数，将输出压缩到0和1之间
        self.ReLU = nn.ReLU()

    def forward(self, x):
        """
        x: 输入数据，形状为(B, seq_length=10, d_model)
        返回: 输出数据，形状为(B, label_length=3, label_dim=2)
        """
        # 1) 将数据通过Mamba模型，形状为(B, L, D)，L与输入相同
        # 使用torch.no_grad()上下文管理器来冻结Mamba模型
        with torch.no_grad():
            mamba_out, out_sliced = self.mamba(x)  # (B, L, D)

        # 应用dropout层
        out_sliced = self.dropout(out_sliced)  # (B, 3, D)

        # 应用全连接层
        out_sliced = self.fc(out_sliced)  # (B, 3, 1)
        # 应用sigmoid激活函数
        out_pred = self.ReLU(out_sliced)  # (B, 3, 1)

        # 由于sigmoid激活函数的输出是(B, 3, 1)，需要将其扩展到(B, 3, 2)
        # 这里假设有一个维度扩展的操作，但是代码中没有明确实现
        # 通常，这可以通过使用unsqueeze或view等操作来实现

        return out_pred  # 注意：这里的输出形状应该是(B, 3, 1)，而不是(B, 3, 2)

# 定义一个函数，用于训练模型并进行预测



def PredictWithData(mamba_model, trainX, trainy, testX, patience=5, delta=0.001):
    # 创建一个MambaHeadWithWrapper实例，封装了传入的mamba_model，并设置标签长度为3，标签维度为1，dropout概率为0.5
    clf = mamba_model

    # # 假设 mamba 是你的模型实例
    # for name, param in clf.named_parameters():
    #     # 如果参数的名字包含 'head'，就不冻结这个层的参数
    #     if 'head' not in name:
    #         param.requires_grad = False
    #     else:
    #         param.requires_grad = True

    # 创建一个Adam优化器，只优化clf中需要梯度的参数，设置学习率为args.lr，权重衰减为args.wd
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, clf.parameters()), lr=args.lr, weight_decay=args.wd)

    # 将训练数据和测试数据转换为torch.Tensor，并设置为浮点类型
    xt = torch.from_numpy(trainX).float()
    xv = torch.from_numpy(testX).float()

    # 将训练标签转换为torch.Tensor，并设置为浮点类型，同时增加一个维度
    yt = torch.from_numpy(trainy).float().unsqueeze(-1)

    # 如果args.cuda为True，则将模型和数据移动到GPU上
    if args.cuda:
        clf = clf.cuda()
        xt = xt.cuda()
        xv = xv.cuda()
        yt = yt.cuda()

    # 创建一个学习率调度器，每50个epoch将学习率减半
    scheduler = StepLR(opt, step_size=50, gamma=0.5)

    # 初始化早停机制的相关变量
    best_loss = float('inf')
    epochs_no_improve = 0

    for e in range(args.epochs):
        # 设置模型为训练模式
        clf.train()

        # 通过模型传递训练数据
        z = clf(xt)

        # 计算预测值和真实值之间的均方误差
        loss = F.mse_loss(z, yt)

        # 清空优化器的梯度
        opt.zero_grad()

        # 反向传播计算梯度
        loss.backward()

        # 更新模型参数
        opt.step()

        # 更新学习率
        scheduler.step()

        # 每10个epoch打印一次当前epoch和损失值
        if e % 10 == 0:
            print(f'Epoch {e} | Loss: {loss.item()}')

        # 检查是否需要早停
        if loss < best_loss - delta:
            best_loss = loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # 每个epoch结束后保存模型
        save_dir = '迁移的权重0109'
        model_filename = f"{save_dir}/model_epoch_{e + 1}_loss_{loss.item():.4f}.pth"
        torch.save(clf.state_dict(), model_filename)
        print(f"Saved model at epoch {e + 1} with loss {loss.item():.4f}")

        # 如果连续patience个epoch没有改善，则停止训练
        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {e + 1}')
            break

    # 设置模型为评估模式
    clf.eval()

    # 使用模型进行测试数据的预测
    with torch.no_grad():  # 确保在评估模式下不计算梯度
        mat = clf(xv)

    # 如果使用了GPU，将预测结果移动回CPU
    if args.cuda:
        mat = mat.cpu()

    # 打印预测结果的形状
    print(mat.shape)

    # 将预测结果转换为numpy数组，并返回
    yhat = mat.detach().numpy()
    return yhat


# 主程序入口
path = r'C:\Users\32468\Desktop\mamba双头\mamba双头尝试\data\Case' + args.case
args.case == 'B'
# 使用 'CS236final.xlsx' 和 'CS237final.xlsx' 进行训练
xt1, yt1 = ReadData(path, 'CS2_36_final_重提_update.xlsx', args.task)
xt2, yt2 = ReadData(path, 'CS2_37_final_重提_update.xlsx', args.task)
trainX = np.vstack((xt1, xt2))
trainy = np.vstack((yt1, yt2))

# 使用 'CS238final.xlsx' 进行预测
testX, testy = ReadData(path, 'CS2_38_final_重提_update.xlsx', args.task)


# 1) 设置Mamba配置
config = MambaConfig(
    d_model=8,  # 必须与X.shape[-1]匹配，即特征向量的维度
    n_layers=4,  # Mamba层的数量
    d_state=16,  # 状态向量的维度
    expand_factor=2,  # 扩展因子，用于计算卷积核的大小
    d_conv=4,  # 卷积核的维度
    pscan=True,  # 是否使用并行扫描
)

# 2) 创建Mamba模型实例
mamba_model = Mamba(config)

# 3) 将Mamba模型封装在一个"head"中，使其输出形状为(B, 3, 2)
clf = MambaHead(mamba_model, label_length=3, label_dim=1)

# 检查是否提供了预训练的模型路径
pretrained_mamba = None
if args.pretrain_model:
    try:
        print(f"加载预训练模型权重: {args.pretrain_model}")

        # 1️⃣ 加载预训练的权重
        state_dict = torch.load(args.pretrain_model, map_location='cuda' if args.cuda else 'cpu')

        # 2️⃣ 去除 "mamba." 前缀，因为模型可能是在不同的命名空间下保存的
        state_dict = {key.replace('mamba.', ''): value for key, value in state_dict.items()}

        # 3️⃣ 加载权重到clf模型中，设置 strict=False，忽略不匹配的权重
        clf.load_state_dict(state_dict, strict=False)
        print("预训练模型加载成功")
    except Exception as e:
        # 如果加载模型过程中出现异常，打印异常信息
        print(f"加载模型失败: {e}")


# 使用迁移学习
predictions_transfer = PredictWithData(clf,trainX, trainy, testX)

# 打印评估结果
print('MSE RMSE MAE R2')
print(testy.shape)
print(predictions_transfer.shape)
evaluation_metric(testy, predictions_transfer.squeeze(-1))

# 绘制对比图
plt.figure()
plt.plot(testy, label='True')
plt.plot(predictions_transfer.squeeze(-1), label='Prediction (Transfer)')
plt.title(f'{args.task} Prediction with Transfer Learning')
plt.xlabel('Cycle')
plt.ylabel(f'{args.task} Value')
plt.legend()
plt.show()