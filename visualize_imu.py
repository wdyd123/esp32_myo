import re
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import serial

SERIAL_PORT = "COM7"
BAUD_RATE = 115200
MAX_POINTS = 100


class IMUVisualizer:
    def __init__(self):
        self.ser = None
        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 8))
        self.lines = []
        self.data_buffers = {
            "quat": {"w": [], "x": [], "y": [], "z": []},
            "acc": {"x": [], "y": [], "z": []},
            "gyro": {"x": [], "y": [], "z": []},
        }
        self.time_steps = []
        self.counter = 0

        self.setup_plots()
        self.connect_serial()

    def connect_serial(self):
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            print(f"Connected to {SERIAL_PORT}")
        except Exception as e:
            print(f"Cannot open serial port: {e}")
            sys.exit(1)

    def setup_plots(self):
        titles = [
            "Quaternion (W, X, Y, Z)",
            "Accelerometer (X, Y, Z)",
            "Gyroscope (X, Y, Z)",
        ]
        colors_quat = ["r", "g", "b", "m"]
        colors_acc_gyro = ["r", "g", "b"]

        for i, title in enumerate(titles):
            ax = self.axs[i]
            ax.set_title(title)
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.grid(True)
            ax.set_xlim(0, MAX_POINTS)

            if i == 0:
                ax.set_ylim(-1.2, 1.2)
                for j, label in enumerate(["W", "X", "Y", "Z"]):
                    line, = ax.plot([], [], label=label, color=colors_quat[j])
                    self.lines.append(line)
            elif i == 1:
                ax.set_ylim(-4.0, 4.0)
                for j, label in enumerate(["X", "Y", "Z"]):
                    line, = ax.plot([], [], label=label, color=colors_acc_gyro[j])
                    self.lines.append(line)
            else:
                ax.set_ylim(-2000.0, 2000.0)
                for j, label in enumerate(["X", "Y", "Z"]):
                    line, = ax.plot([], [], label=label, color=colors_acc_gyro[j])
                    self.lines.append(line)

            ax.legend(loc="upper right")

        plt.tight_layout()

    def parse_line(self, line):
        try:
            quat_match = re.search(
                r"Q:\s+([-0-9.e]+)\s+([-0-9.e]+)\s+([-0-9.e]+)\s+([-0-9.e]+)",
                line,
            )
            acc_match = re.search(
                r"ACC:\s+([-0-9.e]+)\s+([-0-9.e]+)\s+([-0-9.e]+)",
                line,
            )
            gyro_match = re.search(
                r"GYRO:\s+([-0-9.e]+)\s+([-0-9.e]+)\s+([-0-9.e]+)",
                line,
            )

            if quat_match and acc_match and gyro_match:
                qw, qx, qy, qz = map(float, quat_match.groups())
                ax, ay, az = map(float, acc_match.groups())
                gx, gy, gz = map(float, gyro_match.groups())
                return {
                    "quat": [qw, qx, qy, qz],
                    "acc": [ax, ay, az],
                    "gyro": [gx, gy, gz],
                }
        except Exception:
            pass
        return None

    def update_axes(self):
        if not self.time_steps:
            return

        x_min = self.time_steps[0]
        x_max = max(self.time_steps[-1], x_min + 1)
        for ax in self.axs:
            ax.set_xlim(x_min, x_max)

    def update_plot(self, frame):
        if not self.ser or not self.ser.is_open:
            return self.lines

        try:
            line = self.ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                return self.lines

            data = self.parse_line(line)
            if not data:
                return self.lines

            self.counter += 1
            self.time_steps.append(self.counter)

            for i, val in enumerate(data["quat"]):
                key = ["w", "x", "y", "z"][i]
                self.data_buffers["quat"][key].append(val)

            for i, val in enumerate(data["acc"]):
                key = ["x", "y", "z"][i]
                self.data_buffers["acc"][key].append(val)

            for i, val in enumerate(data["gyro"]):
                key = ["x", "y", "z"][i]
                self.data_buffers["gyro"][key].append(val)

            if len(self.time_steps) > MAX_POINTS:
                self.time_steps.pop(0)
                for key in self.data_buffers["quat"]:
                    self.data_buffers["quat"][key].pop(0)
                for key in self.data_buffers["acc"]:
                    self.data_buffers["acc"][key].pop(0)
                for key in self.data_buffers["gyro"]:
                    self.data_buffers["gyro"][key].pop(0)

            for i, key in enumerate(["w", "x", "y", "z"]):
                self.lines[i].set_data(self.time_steps, self.data_buffers["quat"][key])

            for i, key in enumerate(["x", "y", "z"]):
                self.lines[4 + i].set_data(self.time_steps, self.data_buffers["acc"][key])

            for i, key in enumerate(["x", "y", "z"]):
                self.lines[7 + i].set_data(self.time_steps, self.data_buffers["gyro"][key])

            self.update_axes()

        except Exception as e:
            print(f"Error: {e}")

        return self.lines

    def run(self):
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plot, interval=20, blit=False, cache_frame_data=False
        )
        plt.show()

        if self.ser and self.ser.is_open:
            self.ser.close()


if __name__ == "__main__":
    visualizer = IMUVisualizer()
    visualizer.run()
