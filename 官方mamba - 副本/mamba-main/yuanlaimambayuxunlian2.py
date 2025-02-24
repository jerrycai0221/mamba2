import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import scale
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_simple import Mamba
import os
import argparse

# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=True, help='CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=600, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-6, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32, help='Dimension of representations')
parser.add_argument('--layer', type=int, default=4, help='Num of layers')
parser.add_argument('--task', type=str, default='SOH', help='RUL or SOH')
parser.add_argument('--case', type=str, default='B', help='A or B')
parser.add_argument('--pretrain', action='store_true', default=True, help='Flag to pretrain the model')
parser.add_argument('--pretrain-model', type=str, default=r'C:\Users\32468\Desktop\mamba\MambaLithium-main\迁移权重\Model_num164_now_loss0.0328.pth', help='Path to pre-trained model')

args = parser.parse_args()
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


# 读取数据
from sklearn.preprocessing import MinMaxScaler


from sklearn.preprocessing import MinMaxScaler

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



# 网络结构
class MambaHead(nn.Module):
    """
    Wraps the Mamba model and a small Linear "head" to produce the desired (B, 3, 2) outputs.
    """
    def __init__(self, mamba_model, label_length=3, label_dim=2,dropout_prob=0.3, kernel_size=3):
        super().__init__()
        self.mamba = mamba_model
        self.label_length = label_length  # e.g. 3
        self.label_dim = label_dim        # e.g. 2
        # A linear head that maps from D (hidden size) to 'label_dim' (e.g. 2)
        d_model = mamba_model.config.d_model
        self.conv = nn.Conv1d(in_channels=label_dim, out_channels=label_dim, kernel_size=kernel_size, padding=1)
        self.dropout = nn.Dropout(dropout_prob)
        self.head = nn.Linear(d_model, label_dim)  # applies to each timestep

    def forward(self, x):
        """
        x: (B, seq_length=10, d_model)
        returns: (B, label_length=3, label_dim=2)
        """
        # 1) Pass data through Mamba, shape (B, L, D)
        mamba_out = self.mamba(x)  # (B, L, D), same L as input

        # 2) Take the last 'label_length' timesteps
        out_sliced = mamba_out[:, -self.label_length:, :]  # (B, 3, D)

        # 3) Apply the linear head across each timestep -> (B, 3, label_dim)
        out_pred = self.head(out_sliced)  # (B, 3, 2)

        return out_pred

# 定义加权 Smooth L1 损失函数
def weighted_smooth_l1_loss(input, target, weight=None):
    """
    加权 Smooth L1 损失
    :param input: 模型预测值
    :param target: 实际值
    :param weight: 权重向量
    :return: 加权损失值
    """
    loss = F.smooth_l1_loss(input, target, reduction='none')  # 不进行 reduction，保留逐点损失
    if weight is not None:
        loss = loss * weight  # 根据权重调整损失值
    return loss.mean()  # 最终对所有损失取平均


# 预训练 Mamba 模型
def pretrain_mamba_model(trainX, trainy):
    save_dir = '迁移权重重重'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1) Set up the Mamba Config
    config = Mamba(
        d_model=8,  # must match X.shape[-1]
        n_layers=4,  # how many Mamba layers
        d_state=16,
        expand_factor=2,
        d_conv=4,
        pscan=True,  # parallel scan
    )

    # 2) Create Mamba
    mamba_model = Mamba(config)

    # 3) Wrap it in a "head" that outputs shape (B, 3, 2)
    clf = MambaHead(mamba_model, label_length=3, label_dim=1,dropout_prob=0.3, kernel_size=3)
    opt = torch.optim.Adam(clf.parameters(), lr=args.lr, weight_decay=args.wd)

    xt = torch.from_numpy(trainX).float()
    yt = torch.from_numpy(trainy).float().unsqueeze(-1)

    if args.cuda:
        clf = clf.cuda()
        xt = xt.cuda()
        yt = yt.cuda()

    initial_loss = 1000

    for e in range(args.epochs):
        clf.train()
        z = clf(xt)  # 前向传播，获取预测值

        # ======== 引入权重 ========
        # 假设异常值的权重较低，正常值权重为 1
        weights = torch.ones_like(yt)  # 初始化权重为 1
        weights[yt < 0.2] = 0.5  # 如果真实值小于 0.2（假设异常值），权重设为 0.5

        # 使用加权损失函数
        loss = weighted_smooth_l1_loss(z, yt, weight=weights)
        # =========================

        opt.zero_grad()
        loss.backward()
        opt.step()

        if e % 10 == 0:
            print(f'Epoch {e} | Loss: {loss.item()}')

        if loss < initial_loss:
            initial_loss = loss
            model_path = os.path.join(save_dir, f"Model_num{e}_now_loss{loss:.4f}.pth")
            torch.save(clf.state_dict(), model_path)

    return clf

# 主程序入口
path = r'C:\Users\32468\Desktop\mamba双头\mamba双头尝试\data\CaseB'
xt1, yt1 = ReadData(path, 'CS2_35_final_重提_update.xlsx', args.task)
trainX = xt1  # 使用 'CS2_35_final.xlsx' 数据进行预训练
trainy = yt1
testX, testy = ReadData(path, 'CS2_36_final_重提_update.xlsx', args.task)  # 读取测试数据

# 训练模型
clf = pretrain_mamba_model(trainX, trainy)

# 对测试数据进行预测
xt = torch.from_numpy(testX).float()  # 使用读取的测试数据
if args.cuda:
    xt = xt.cuda()
clf.eval()
with torch.no_grad():
    predictions_transfer = clf(xt).cpu().numpy()

# 评估结果
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
