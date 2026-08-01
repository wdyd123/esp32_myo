"""
Myo 数据验证与统计工具

功能：
  - 扫描 utilities/ 目录中的 myo_data_*.csv 文件
  - 验证数据格式是否与 train_gesture_nn.py 兼容
  - 报告各手势的样本数量和时长
  - 确保数据可直接用于训练

用法：
  python process_dataset.py
  python process_dataset.py --data-dir ../utilities
"""

import argparse
import glob
import os
from collections import defaultdict

# 默认数据目录：utilities/（与本文件同级）
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utilities")


def parse_csv(path):
    """
    解析单个 CSV 文件，返回统计信息。
    """
    imu_count = 0
    emg_count = 0
    imu_labels = defaultdict(int)
    emg_labels = defaultdict(int)
    imu_timestamps = []
    emg_timestamps = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("label,"):
                continue

            parts = line.split(",")
            if len(parts) < 4:
                continue

            try:
                label = int(parts[0])
                ts = int(float(parts[1]))
                stype = parts[2].strip().upper()
            except (ValueError, IndexError):
                continue

            if stype == "IMU" and len(parts) >= 13:
                imu_count += 1
                imu_labels[label] += 1
                imu_timestamps.append(ts)
            elif stype == "EMG" and len(parts) >= 35:
                emg_count += 1
                emg_labels[label] += 1
                emg_timestamps.append(ts)

    return {
        "imu_count": imu_count,
        "emg_count": emg_count,
        "imu_labels": dict(imu_labels),
        "emg_labels": dict(emg_labels),
        "imu_ts_range": (min(imu_timestamps), max(imu_timestamps)) if imu_timestamps else None,
        "emg_ts_range": (min(emg_timestamps), max(emg_timestamps)) if emg_timestamps else None,
    }


def scan_data_dir(data_dir):
    """扫描目录，返回所有 CSV 文件的统计信息。"""
    csv_files = sorted(glob.glob(os.path.join(data_dir, "myo_data_*.csv")))
    if not csv_files:
        csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

    all_stats = []
    for path in csv_files:
        stats = parse_csv(path)
        stats["filename"] = os.path.basename(path)
        all_stats.append(stats)

    return all_stats


def print_report(all_stats, data_dir):
    """打印数据报告。"""
    if not all_stats:
        print("未找到 CSV 文件。")
        return

    print("\n" + "=" * 70)
    print("Myo 数据验证报告")
    print("=" * 70)

    # 文件级统计
    print("\n【文件统计】")
    total_imu = 0
    total_emg = 0
    all_imu_labels = defaultdict(int)
    all_emg_labels = defaultdict(int)

    for stats in all_stats:
        fname = stats["filename"]
        imu_n = stats["imu_count"]
        emg_n = stats["emg_count"]
        total_imu += imu_n
        total_emg += emg_n

        print(f"\n  {fname}")
        print(f"    IMU: {imu_n:6d} 样本")
        print(f"    EMG: {emg_n:6d} 样本")

        if stats["imu_ts_range"]:
            t0, t1 = stats["imu_ts_range"]
            duration = (t1 - t0) / 1000.0
            print(f"    IMU 时长: {duration:.1f} 秒")

        if stats["emg_ts_range"]:
            t0, t1 = stats["emg_ts_range"]
            duration = (t1 - t0) / 1000.0
            print(f"    EMG 时长: {duration:.1f} 秒")

        for lbl, cnt in sorted(stats["imu_labels"].items()):
            all_imu_labels[lbl] += cnt
        for lbl, cnt in sorted(stats["emg_labels"].items()):
            all_emg_labels[lbl] += cnt

    # 汇总统计
    print("\n" + "=" * 70)
    print("【汇总统计】")
    print(f"  文件数: {len(all_stats)}")
    print(f"  IMU 总样本: {total_imu}")
    print(f"  EMG 总样本: {total_emg}")

    # 手势分布
    all_labels = sorted(set(all_imu_labels.keys()) | set(all_emg_labels.keys()))
    if all_labels:
        print("\n【手势分布】")
        print(f"  {'手势':>6s}  {'IMU':>8s}  {'EMG':>8s}  {'状态':>10s}")
        print("  " + "-" * 40)
        for lbl in all_labels:
            imu_n = all_imu_labels.get(lbl, 0)
            emg_n = all_emg_labels.get(lbl, 0)
            status = "OK" if (imu_n > 0 and emg_n > 0) else "incomplete"
            print(f"  {lbl:>6d}  {imu_n:>8d}  {emg_n:>8d}  {status:>10s}")

    # 兼容性检查
    print("\n" + "=" * 70)
    print("【兼容性检查】")

    issues = []
    for stats in all_stats:
        if stats["imu_count"] == 0:
            issues.append(f"  {stats['filename']}: 无 IMU 数据")
        if stats["emg_count"] == 0:
            issues.append(f"  {stats['filename']}: 无 EMG 数据")

        imu_labels_set = set(stats["imu_labels"].keys())
        emg_labels_set = set(stats["emg_labels"].keys())
        missing_emg = imu_labels_set - emg_labels_set
        missing_imu = emg_labels_set - imu_labels_set

        if missing_emg:
            issues.append(f"  {stats['filename']}: 手势 {sorted(missing_emg)} 有 IMU 但无 EMG")
        if missing_imu:
            issues.append(f"  {stats['filename']}: 手势 {sorted(missing_imu)} 有 EMG 但无 IMU")

    if issues:
        print("  ! 发现以下问题：")
        for issue in issues:
            print(issue)
    else:
        print("  OK 所有文件格式正确，可用于训练。")

    abs_data_dir = os.path.abspath(data_dir)
    print("\n" + "=" * 70)
    print("训练命令：")
    print(f"  python train_gesture_nn.py --data-dir {abs_data_dir} --epochs 100")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="验证 Myo 数据并生成统计报告",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python process_dataset.py
  python process_dataset.py --data-dir ../utilities
        """
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="包含 myo_data_*.csv 文件的目录 (默认: ../utilities)")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"错误：目录不存在: {args.data_dir}")
        return

    all_stats = scan_data_dir(args.data_dir)
    print_report(all_stats, args.data_dir)


if __name__ == "__main__":
    main()
