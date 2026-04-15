import argparse
import csv
import math
from collections import defaultdict

IMU_FIELDS = ["qw", "qx", "qy", "qz", "ax", "ay", "az", "gx", "gy", "gz"]
EMG_FIELDS = [f"emg_{i}" for i in range(16)]


def parse_samples(input_path):
    samples = []
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader, None)
        if headers is None:
            return samples

        has_label = any(h.strip().lower() == "label" for h in headers)
        label_idx = [i for i, h in enumerate(headers) if h.strip().lower() == "label"]
        label_idx = label_idx[0] if label_idx else None

        for row in reader:
            if len(row) < 2:
                continue
            timestamp_raw = row[0].strip()
            if not timestamp_raw:
                continue
            try:
                timestamp = int(float(timestamp_raw))
            except ValueError:
                continue

            row_type = row[1].strip().upper()
            label = row[label_idx].strip() if label_idx is not None and label_idx < len(row) else ""

            if row_type == "IMU":
                if len(row) < 12:
                    continue
                raw_values = row[2:12]
                if len(raw_values) != len(IMU_FIELDS):
                    continue
                try:
                    values = [float(v.strip()) for v in raw_values]
                except Exception:
                    continue
                samples.append({"type": "IMU", "ts": timestamp, "values": values, "label": label})

            elif row_type == "EMG":
                # Support both compact EMG rows and full-width rows with blank IMU slots.
                if len(row) >= 18 and len(row[2:]) == 16:
                    raw_values = row[2:18]
                elif len(row) >= 28:
                    raw_values = row[12:28]
                else:
                    continue

                try:
                    values = [float(v.strip()) for v in raw_values]
                except Exception:
                    continue
                if len(values) != len(EMG_FIELDS):
                    continue
                samples.append({"type": "EMG", "ts": timestamp, "values": values, "label": label})

    return samples


def aggregate_stats(values):
    # values is list of floats
    n = len(values)
    if n == 0:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    return {"mean": mean, "std": math.sqrt(var), "min": min(values), "max": max(values)}


def build_window_features(imu_window, emg_window, start_ts, end_ts):
    features = {
        "window_start": start_ts,
        "window_end": end_ts,
        "window_center": (start_ts + end_ts) / 2,
        "imu_count": len(imu_window),
        "emg_count": len(emg_window),
    }

    if imu_window:
        for idx, field in enumerate(IMU_FIELDS):
            values = [sample["values"][idx] for sample in imu_window]
            stats = aggregate_stats(values)
            features[f"imu_{field}_mean"] = stats["mean"]
            features[f"imu_{field}_std"] = stats["std"]
            features[f"imu_{field}_min"] = stats["min"]
            features[f"imu_{field}_max"] = stats["max"]
        # last IMU sample in the window
        last_values = imu_window[-1]["values"]
        for idx, field in enumerate(IMU_FIELDS):
            features[f"imu_{field}_last"] = last_values[idx]
    else:
        for field in IMU_FIELDS:
            features[f"imu_{field}_mean"] = math.nan
            features[f"imu_{field}_std"] = math.nan
            features[f"imu_{field}_min"] = math.nan
            features[f"imu_{field}_max"] = math.nan
            features[f"imu_{field}_last"] = math.nan

    if emg_window:
        for ch in range(16):
            values = [sample["values"][ch] for sample in emg_window]
            stats = aggregate_stats(values)
            features[f"emg_{ch}_mean"] = stats["mean"]
            features[f"emg_{ch}_std"] = stats["std"]
            features[f"emg_{ch}_min"] = stats["min"]
            features[f"emg_{ch}_max"] = stats["max"]
    else:
        for ch in range(16):
            features[f"emg_{ch}_mean"] = math.nan
            features[f"emg_{ch}_std"] = math.nan
            features[f"emg_{ch}_min"] = math.nan
            features[f"emg_{ch}_max"] = math.nan

    return features


def write_features(output_path, feature_rows):
    if not feature_rows:
        raise ValueError("No feature rows to write.")

    fieldnames = list(feature_rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in feature_rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Aggregate Myo IMU+EMG into sliding windows.")
    parser.add_argument("--input", default="myo_dataset.csv", help="Input CSV path")
    parser.add_argument("--output", default="processed_dataset.csv", help="Output CSV path")
    parser.add_argument("--window-ms", type=int, default=100, help="Window width in milliseconds")
    parser.add_argument("--step-ms", type=int, default=50, help="Window step in milliseconds")
    parser.add_argument("--min-imu", type=int, default=1, help="Minimum IMU samples required in each window")
    parser.add_argument("--min-emg", type=int, default=1, help="Minimum EMG samples required in each window")
    args = parser.parse_args()

    samples = parse_samples(args.input)
    if not samples:
        raise ValueError("No valid samples found in input CSV.")

    imu_samples = [s for s in samples if s["type"] == "IMU"]
    emg_samples = [s for s in samples if s["type"] == "EMG"]
    if not imu_samples or not emg_samples:
        raise ValueError("Input must contain both IMU and EMG samples.")

    min_ts = min(s["ts"] for s in samples)
    max_ts = max(s["ts"] for s in samples)

    window_rows = []
    start = min_ts
    while start + args.window_ms <= max_ts + 1:
        end = start + args.window_ms
        imu_window = [s for s in imu_samples if start <= s["ts"] < end]
        emg_window = [s for s in emg_samples if start <= s["ts"] < end]

        if len(imu_window) >= args.min_imu and len(emg_window) >= args.min_emg:
            window_rows.append(build_window_features(imu_window, emg_window, start, end))

        start += args.step_ms

    if not window_rows:
        raise ValueError("No windows met the minimum sample requirements.")

    write_features(args.output, window_rows)
    print(f"Wrote {len(window_rows)} aggregated windows to {args.output}")


if __name__ == "__main__":
    main()
