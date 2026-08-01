"""
Myo 手势识别 — 双分支神经网络 (PyTorch)

架构设计:
  IMU 分支 → 2层 LSTM 提取时序特征 → 32维向量
  EMG 分支 → 1D-CNN 提取空间+波形特征 → 32维向量
  融合层   → 拼接(64维) → MLP → 手势类别数

为什么这样设计？
  - IMU 采样率 50Hz，数据是连续的运动轨迹，LSTM 擅长捕捉时序依赖关系
  - EMG 采样率 200Hz，4通道×8采样=32维，包含肌肉激活的空间模式和局部波形，CNN 更高效
  - 两种模态物理特性差异大，各自学习特征后再融合（晚期融合）比直接拼接原始数据更合理

用法:
  python train_gesture_nn.py --data-dir ./utilities --epochs 100
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ─── 常量定义 ────────────────────────────────────────────────────────────────

# IMU 10个特征：四元数(w,x,y,z) + 加速度(x,y,z) + 陀螺仪(x,y,z)
IMU_FEATURES = ["qw", "qx", "qy", "qz", "ax", "ay", "az", "gx", "gy", "gz"]
IMU_DIM = len(IMU_FEATURES)   # 10
EMG_DIM = 32                  # 4通道 × 8采样点 = 32个EMG值

# 滑动窗口参数
# 窗口长度 500ms：覆盖一个完整手势动作的时序信息
# 步长 250ms（50%重叠）：增加训练样本数量，同时保持窗口间的相关性
WINDOW_MS = 500    # 窗口长度（毫秒）
STEP_MS = 250      # 滑动步长（毫秒），50%重叠
IMU_RATE = 50      # IMU 采样率 (Hz)
EMG_RATE = 200     # EMG 采样率 (Hz)

# 将时间转换为采样点数
WINDOW_IMU = int(IMU_RATE * WINDOW_MS / 1000)   # 25 个 IMU 采样点
WINDOW_EMG = int(EMG_RATE * WINDOW_MS / 1000)   # 100 个 EMG 采样点
STEP_IMU   = int(IMU_RATE * STEP_MS / 1000)      # 12 步
STEP_EMG   = int(EMG_RATE * STEP_MS / 1000)      # 50 步


# ─── CSV 解析 ─────────────────────────────────────────────────────────────────

def parse_csv(path):
    """
    解析单个 myo_data_*.csv 文件。

    CSV 格式（由 ESP32 固件输出）：
      label,timestamp,IMU,qw,qx,qy,qz,ax,ay,az,gx,gy,gz
      label,timestamp,EMG,emg0,emg1,...,emg31

    返回: (imu_rows, emg_rows)，每行为 (label, timestamp, values)
    """
    imu_rows, emg_rows = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 跳过空行、META注释行、表头行
            if not line or line.startswith("#") or line.startswith("label,"):
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                label = int(parts[0])       # 手势标签 (1, 2, 3...)
                ts = int(float(parts[1]))   # 时间戳 (ms)
                stype = parts[2].strip().upper()  # 传感器类型: IMU 或 EMG
            except (ValueError, IndexError):
                continue

            if stype == "IMU" and len(parts) >= 13:
                try:
                    vals = [float(parts[3 + i]) for i in range(10)]
                    imu_rows.append((label, ts, vals))
                except ValueError:
                    continue
            elif stype == "EMG" and len(parts) >= 35:
                try:
                    vals = [float(parts[3 + i]) for i in range(32)]
                    emg_rows.append((label, ts, vals))
                except ValueError:
                    continue
    return imu_rows, emg_rows


def load_all_csvs(data_dir):
    """加载目录下所有 CSV 文件，返回按时间戳排序的 IMU/EMG 数据列表。"""
    all_imu, all_emg = [], []
    csv_files = sorted(glob.glob(os.path.join(data_dir, "myo_data_*.csv")))
    if not csv_files:
        csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    for path in csv_files:
        imu, emg = parse_csv(path)
        if imu and emg:
            all_imu.extend(imu)
            all_emg.extend(emg)
            print(f"  {os.path.basename(path)}: {len(imu)} IMU, {len(emg)} EMG")
    # 按时间戳排序，保证同一手势的数据是连续的
    all_imu.sort(key=lambda x: x[1])
    all_emg.sort(key=lambda x: x[1])
    return all_imu, all_emg


# ─── 滑动窗口 ─────────────────────────────────────────────────────────────────

def group_by_label(rows):
    """将 (label, timestamp, values) 按手势标签分组，组内按时间戳排序。"""
    groups = {}
    for label, ts, vals in rows:
        groups.setdefault(label, []).append((ts, vals))
    for k in groups:
        groups[k].sort(key=lambda x: x[0])
    return groups


def create_windows(imu_groups, emg_groups):
    """
    创建同步的 IMU+EMG 滑动窗口。

    核心思路：
      1. 以 IMU 时间线为基准，取 500ms 窗口（25个采样点）
      2. 在同一时间范围内查找对应的 EMG 数据
      3. EMG 采样率是 IMU 的 4 倍，所以窗口内约有 100 个 EMG 采样点
      4. 如果 EMG 不足则重采样或零填充

    返回: (X_imu, X_emg, y) 三个 numpy 数组
      - X_imu: shape (n_windows, 25, 10) — 25个时间步，每步10个特征
      - X_emg: shape (n_windows, 100, 32) — 100个时间步，每步32个EMG值
      - y:     shape (n_windows,) — 每个窗口的手势标签
    """
    X_imu, X_emg, y = [], [], []

    for label in imu_groups:
        if label not in emg_groups:
            print(f"  警告: 标签 {label} 有 IMU 但无 EMG，跳过")
            continue

        imu_series = imu_groups[label]
        emg_series = emg_groups[label]

        if len(imu_series) < WINDOW_IMU:
            continue

        emg_ts = np.array([r[0] for r in emg_series])

        # 滑动窗口：每次前进 STEP_IMU 个 IMU 采样点
        for start in range(0, len(imu_series) - WINDOW_IMU + 1, STEP_IMU):
            # 提取 IMU 窗口
            imu_window = [imu_series[start + j][1] for j in range(WINDOW_IMU)]
            t0 = imu_series[start][0]           # 窗口起始时间
            t1 = imu_series[start + WINDOW_IMU - 1][0]  # 窗口结束时间

            # 在同一时间范围内查找 EMG 数据
            mask = (emg_ts >= t0) & (emg_ts <= t1)
            emg_indices = np.where(mask)[0]

            if len(emg_indices) >= WINDOW_EMG:
                # EMG 充足：取前 WINDOW_EMG 个采样点
                emg_window = [emg_series[emg_indices[i]][1] for i in range(WINDOW_EMG)]
            elif len(emg_indices) >= 4:
                # EMG 不足但有一些：线性重采样到 WINDOW_EMG 个点
                indices = np.linspace(0, len(emg_indices) - 1, WINDOW_EMG, dtype=int)
                emg_window = [emg_series[emg_indices[i]][1] for i in indices]
            else:
                # EMG 严重不足：零填充
                emg_window = [[0.0] * EMG_DIM] * WINDOW_EMG

            X_imu.append(imu_window)
            X_emg.append(emg_window)
            y.append(label)

    return (np.array(X_imu, dtype=np.float32),
            np.array(X_emg, dtype=np.float32),
            np.array(y, dtype=np.int64))


# ─── 数据集 ───────────────────────────────────────────────────────────────────

class GestureDataset(Dataset):
    """
    PyTorch Dataset：封装 IMU+EMG 数据，并做 Z-Score 标准化。

    为什么要标准化？
      - IMU 的加速度值范围约 [-4, 4]，陀螺仪约 [-2000, 2000]
      - EMG 的值范围约 [-128, 127]
      - 不同特征尺度差异大会导致梯度更新不稳定，标准化后均值为0、标准差为1
      - 重要：验证集和测试集必须使用训练集的统计量，避免数据泄露
    """

    def __init__(self, X_imu, X_emg, y, stats=None):
        self.X_imu = torch.from_numpy(X_imu)
        self.X_emg = torch.from_numpy(X_emg)
        self.y = torch.from_numpy(y)

        if stats is None:
            # 从训练集计算标准化参数（均值和标准差）
            # dim=(0,1) 表示在 batch 和 time 维度上计算，保留每个特征维度的统计量
            self.imu_mean = self.X_imu.mean(dim=(0, 1), keepdim=True)
            self.imu_std = self.X_imu.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
            self.emg_mean = self.X_emg.mean(dim=(0, 1), keepdim=True)
            self.emg_std = self.X_emg.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
        else:
            # 使用外部传入的统计量（验证集/测试集用训练集的参数）
            self.imu_mean = stats["imu_mean"]
            self.imu_std = stats["imu_std"]
            self.emg_mean = stats["emg_mean"]
            self.emg_std = stats["emg_std"]

        # Z-Score 标准化: x' = (x - mean) / std
        self.X_imu = (self.X_imu - self.imu_mean) / self.imu_std
        self.X_emg = (self.X_emg - self.emg_mean) / self.emg_std

    def get_stats(self):
        """返回标准化参数，用于传递给验证集/测试集。"""
        return {
            "imu_mean": self.imu_mean, "imu_std": self.imu_std,
            "emg_mean": self.emg_mean, "emg_std": self.emg_std,
        }

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_imu[idx], self.X_emg[idx], self.y[idx]


# ─── 模型定义 ─────────────────────────────────────────────────────────────────

class GestureNet(nn.Module):
    """
    双分支手势分类器。

    ┌──────────────────────────────┐
    │  IMU 分支 (LSTM)             │  LSTM 通过门控机制（遗忘门、输入门、输出门）
    │  输入: [batch, 25, 10]       │  选择性地记住/遗忘时序信息，非常适合运动轨迹
    │  → LSTM(10→64) × 2层        │  这种序列建模任务。
    │  → 取最后时间步的隐藏状态     │
    │  → FC → [batch, 32]          │
    └──────────┬───────────────────┘
               │ concat → [batch, 64]
    ┌──────────┴───────────────────┐    ┌──────────────────────┐
    │  EMG 分支 (CNN)              │    │  融合分类器 (MLP)     │
    │  输入: [batch, 100, 32]      │───>│  64 → 32 → n_classes │
    │  → Conv1d → BN → ReLU → Pool│    │  + Dropout 防过拟合   │
    │  → Conv1d → BN → ReLU → Pool│    └──────────────────────┘
    │  → FC → [batch, 32]          │
    └──────────────────────────────┘

    关键概念解释：
      - BatchNorm (BN): 对每一层的输出做归一化，加速训练、提高稳定性
      - Dropout: 训练时随机丢弃一部分神经元（这里30%），防止过拟合
      - AdaptiveAvgPool: 自适应池化，无论输入长度如何，都输出固定大小
    """

    def __init__(self, n_classes, imu_dim=IMU_DIM, emg_dim=EMG_DIM):
        super().__init__()

        # ═══ IMU 分支：用 LSTM 建模运动时序 ═══
        # LSTM (Long Short-Term Memory) 是一种特殊的 RNN，能学习长距离依赖
        # num_layers=2: 堆叠两层 LSTM，第二层在第一层的输出上进一步学习
        # dropout=0.2: 两层 LSTM 之间的 dropout（不是输入/输出）
        self.imu_lstm = nn.LSTM(
            input_size=imu_dim,    # 输入维度 = 10 (四元数4 + 加速度3 + 陀螺仪3)
            hidden_size=64,        # 隐藏层维度 = 64
            num_layers=2,          # 2 层堆叠
            batch_first=True,      # 输入格式: [batch, seq_len, features]
            dropout=0.2,           # 层间 dropout
            bidirectional=False,   # 单向（只用过去的信息，不做未来预测）
        )
        # 将 LSTM 最后时间步的隐藏状态 (64维) 映射到 32 维
        self.imu_fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),             # 非线性激活函数
        )

        # ═══ EMG 分支：用 1D-CNN 提取肌肉激活模式 ═══
        # Conv1d: 一维卷积，沿时间轴滑动，提取局部波形特征
        # kernel_size=7: 卷积核大小，覆盖7个时间步，能捕捉较宽的波形模式
        # padding=3: 填充使输出长度与输入相同
        # BatchNorm1d: 对卷积输出做归一化，加速收敛
        # MaxPool1d(2): 最大池化，将序列长度减半，减少计算量并增加感受野
        self.emg_conv = nn.Sequential(
            nn.Conv1d(emg_dim, 64, kernel_size=7, padding=3),   # [32, 100] → [64, 100]
            nn.BatchNorm1d(64),                                   # 归一化
            nn.ReLU(),
            nn.MaxPool1d(2),                                      # [64, 100] → [64, 50]
            nn.Conv1d(64, 128, kernel_size=5, padding=2),         # [64, 50] → [128, 50]
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),                              # [128, 50] → [128, 8]
        )
        # 将池化后的特征展平并通过全连接层
        # 128 通道 × 8 时间步 = 1024 维 → 压缩到 32 维
        self.emg_fc = nn.Sequential(
            nn.Linear(128 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # ═══ 融合分类器 ═══
        # 将 IMU 和 EMG 的 32 维特征拼接成 64 维，再通过 MLP 分类
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),                    # 随机丢弃 30% 的神经元
            nn.Linear(32, n_classes),           # 输出 n_classes 个 logits（未归一化的分数）
        )

    def forward(self, x_imu, x_emg):
        """
        前向传播。

        参数:
          x_imu: [batch, 25, 10]  — IMU 时间序列
          x_emg: [batch, 100, 32] — EMG 时间序列

        返回:
          logits: [batch, n_classes] — 每个类别的得分（未经 softmax）
        """
        # IMU 分支: LSTM 处理时序数据
        # lstm 输出: (output, (h_n, c_n))
        # h_n[-1] 是最后一层的最后时间步隐藏状态，包含了整个序列的摘要信息
        _, (imu_h, _) = self.imu_lstm(x_imu)
        imu_out = self.imu_fc(imu_h[-1])  # [batch, 32]

        # EMG 分支: CNN 处理
        # Conv1d 期望输入 [batch, channels, length]，所以需要转置
        x_emg_t = x_emg.transpose(1, 2)              # [batch, 32, 100] → [batch, 32, 100]
        emg_feat = self.emg_conv(x_emg_t)             # [batch, 128, 8]
        emg_feat = emg_feat.flatten(1)                 # [batch, 1024]
        emg_out = self.emg_fc(emg_feat)               # [batch, 32]

        # 晚期融合: 拼接两个分支的特征向量
        fused = torch.cat([imu_out, emg_out], dim=1)  # [batch, 64]
        return self.classifier(fused)                  # [batch, n_classes]


# ─── 训练与评估 ───────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device):
    """
    训练一个 epoch（遍历所有训练数据一次）。

    流程:
      1. 前向传播: 输入数据 → 模型 → 预测结果
      2. 计算损失: CrossEntropyLoss = softmax + 负对数似然
      3. 反向传播: loss.backward() 计算每个参数的梯度
      4. 参数更新: optimizer.step() 用梯度下降更新权重
    """
    model.train()  # 设为训练模式（启用 Dropout、BatchNorm 使用 batch 统计量）
    total_loss, correct, total = 0.0, 0, 0
    for x_imu, x_emg, labels in loader:
        x_imu, x_emg, labels = x_imu.to(device), x_emg.to(device), labels.to(device)
        optimizer.zero_grad()          # 清零梯度（否则梯度会累积）
        out = model(x_imu, x_emg)      # 前向传播
        loss = F.cross_entropy(out, labels)  # 计算交叉熵损失
        loss.backward()                # 反向传播，计算梯度
        optimizer.step()               # 更新参数

        total_loss += loss.item() * labels.size(0)
        correct += (out.argmax(1) == labels).sum().item()  # argmax 取最大得分的类别
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, device):
    """
    在验证集或测试集上评估模型。

    与训练的区别:
      - model.eval(): 关闭 Dropout，BatchNorm 使用运行均值/方差
      - torch.no_grad(): 不计算梯度，节省显存
    """
    model.eval()  # 设为评估模式
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():  # 不计算梯度
        for x_imu, x_emg, labels in loader:
            x_imu, x_emg, labels = x_imu.to(device), x_emg.to(device), labels.to(device)
            out = model(x_imu, x_emg)
            preds = out.argmax(1)  # 取预测得分最高的类别
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return correct / total, np.array(all_preds), np.array(all_labels)


# ─── 主函数 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="训练 Myo 手势识别神经网络")
    parser.add_argument("--data-dir", required=True,
                        help="包含 myo_data_*.csv 文件的目录")
    parser.add_argument("--epochs", type=int, default=100, help="最大训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="初始学习率")
    parser.add_argument("--save", default="gesture_net.pth", help="模型保存路径")
    args = parser.parse_args()

    # 自动检测 GPU，有就用 GPU（快几十倍），没有就用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 第一步：加载数据 ──
    print("\n=== 加载数据 ===")
    all_imu, all_emg = load_all_csvs(args.data_dir)
    print(f"总计: {len(all_imu)} IMU 行, {len(all_emg)} EMG 行")

    if not all_imu or not all_emg:
        print("错误: 未找到数据。请确保 CSV 文件在指定目录中。")
        return

    # 按手势标签分组
    imu_groups = group_by_label(all_imu)
    emg_groups = group_by_label(all_emg)
    labels_present = sorted(set(imu_groups.keys()) & set(emg_groups.keys()))
    print(f"同时具有 IMU+EMG 的手势标签: {labels_present}")
    for lbl in labels_present:
        print(f"  标签 {lbl}: {len(imu_groups[lbl])} IMU, {len(emg_groups[lbl])} EMG 样本")

    # ── 第二步：创建滑动窗口 ──
    print("\n=== 创建滑动窗口 ===")
    X_imu, X_emg, y = create_windows(imu_groups, emg_groups)
    print(f"窗口数: {len(y)} | IMU: {X_imu.shape} | EMG: {X_emg.shape}")
    unique, counts = np.unique(y, return_counts=True)
    print(f"标签分布: {dict(zip(unique.tolist(), counts.tolist()))}")

    if len(y) < 10:
        print("错误: 窗口数量不足，无法训练。请采集更多数据。")
        return

    # ── 第三步：标签重映射 ──
    # PyTorch 的 CrossEntropyLoss 要求标签是 0 开始的连续整数
    # 例如原始标签 [1, 2, 3] → 重映射为 [0, 1, 2]
    unique_labels = np.unique(y)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y_remapped = np.array([label_map[yi] for yi in y], dtype=np.int64)
    print(f"标签映射: {label_map}")
    y = y_remapped

    # ── 第四步：划分数据集 ──
    # 70% 训练 / 15% 验证 / 15% 测试
    # 训练集: 用于更新模型参数
    # 验证集: 用于调超参数和早停（不直接参与训练，但用于决定是否停止）
    # 测试集: 最终评估，完全隔离，只在最后用一次
    n_total = len(y)
    n_test = max(1, int(n_total * 0.15))
    n_val = max(1, int(n_total * 0.15))
    n_train = n_total - n_test - n_val

    indices = np.random.permutation(n_total)  # 随机打乱
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    print(f"数据划分: 训练={len(train_idx)}, 验证={len(val_idx)}, 测试={len(test_idx)}")

    # 创建 Dataset（训练集计算标准化参数，验证/测试集复用训练集的参数）
    train_ds = GestureDataset(X_imu[train_idx], X_emg[train_idx], y[train_idx])
    stats = train_ds.get_stats()
    val_ds = GestureDataset(X_imu[val_idx], X_emg[val_idx], y[val_idx], stats=stats)
    test_ds = GestureDataset(X_imu[test_idx], X_emg[test_idx], y[test_idx], stats=stats)

    # DataLoader: 自动处理批次切分和 shuffle
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # ── 第五步：构建模型 ──
    n_classes = len(unique_labels)
    model = GestureNet(n_classes=n_classes).to(device)

    # Adam 优化器：自适应学习率，比 vanilla SGD 收敛更快
    # weight_decay: L2 正则化，防止权重过大导致过拟合
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # ReduceLROnPlateau: 当验证损失连续 patience 轮不下降时，将学习率乘以 factor
    # 好处: 训练初期用大学习率快速收敛，后期用小学习率精细调优
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, min_lr=1e-6
    )

    print(f"\n=== 模型结构 ===")
    print(model)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"参数量: {param_count:,}\n")

    # ── 第六步：训练循环 ──
    print("=== 开始训练 ===")
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_acc, _, _ = evaluate(model, val_loader, device)
        scheduler.step(1 - val_acc)  # 传入验证错误率，如果不下降则降低学习率

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # 保存最佳模型（验证集上表现最好的那次）
            torch.save({
                "model_state": model.state_dict(),
                "n_classes": n_classes,
                "labels": labels_present,
                "stats": {k: v.cpu() for k, v in stats.items()},
            }, args.save)
        else:
            patience_counter += 1

        current_lr = optimizer.param_groups[0]["lr"]
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | loss={train_loss:.4f} acc={train_acc:.3f} "
                  f"val_acc={val_acc:.3f} best={best_val_acc:.3f} lr={current_lr:.2e}")

        # 早停 (Early Stopping): 如果验证集准确率连续 25 轮没有提升，停止训练
        # 原因: 继续训练只会导致过拟合（训练集准确率上升但验证集下降）
        if patience_counter >= 25:
            print(f"早停: 第 {epoch} 轮（连续 25 轮无提升）")
            break

    print(f"\n训练完成。最佳验证准确率: {best_val_acc:.4f}")
    print(f"模型已保存到: {args.save}")

    # ── 第七步：最终评估 ──
    # 用测试集评估最终模型（测试集在整个训练过程中完全未被使用）
    print("\n=== 测试集评估 ===")
    ckpt = torch.load(args.save, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    test_acc, preds, labels = evaluate(model, test_loader, device)
    print(f"测试准确率: {test_acc:.4f}")

    try:
        from sklearn.metrics import classification_report
        # 反向映射：将 0-indexed 标签还原为原始标签用于显示
        inv_map = {v: k for k, v in label_map.items()}
        target_names = [str(inv_map[i]) for i in range(n_classes)]
        print("\n分类报告:")
        print(classification_report(labels, preds, target_names=target_names))
    except ImportError:
        print("(安装 scikit-learn 可获得详细分类报告)")


if __name__ == "__main__":
    main()
