import sys

import matplotlib
matplotlib.use('TkAgg')  # Force Tk backend for better Windows compatibility
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import serial
import serial.tools.list_ports

SERIAL_PORT = "COM5"
BAUD_RATE = 115200
MAX_POINTS = 200
PLOT_INTERVAL_MS = 50

# CSV format from ESP32 firmware:
#   label,timestamp,IMU,qw,qx,qy,qz,ax,ay,az,gx,gy,gz
#   label,timestamp,EMG,emg0..emg31


def list_com_ports():
    """Return list of available COM port names."""
    return [p.device for p in serial.tools.list_ports.comports()]


def choose_port(default=SERIAL_PORT):
    """Let user pick a COM port. Returns port name or None to skip serial."""
    available = list_com_ports()
    if not available:
        print("No COM ports found.")
        return None

    print(f"Available COM ports: {', '.join(available)}")

    if default in available:
        choice = input(f"Enter port [{default}]: ").strip()
        if not choice:
            return default
        return choice if choice in available else default
    else:
        print(f"Default port {default} not found.")
        choice = input("Enter port (or press Enter to skip serial): ").strip()
        return choice if choice in available else None


class IMUVisualizer:
    def __init__(self, port=None, baud=BAUD_RATE, auto_start=True):
        self.ser = None
        self.port = port
        self.baud = baud
        self.serial_ok = False
        self.collecting = False  # Whether ESP32 is in collection mode

        # Data buffers
        self.time_steps = []
        self.counter = 0
        self.buf_quat = {"w": [], "x": [], "y": [], "z": []}
        self.buf_acc = {"x": [], "y": [], "z": []}
        self.buf_gyro = {"x": [], "y": [], "z": []}
        self.buf_emg_rms = {ch: [] for ch in range(4)}  # RMS per EMG channel

        # Serial line buffer (data may arrive in fragments)
        self._line_buf = ""

        # Create figure: 4 rows (quat, acc, gyro, emg)
        self.fig, self.axs = plt.subplots(4, 1, figsize=(12, 10),
                                          gridspec_kw={"height_ratios": [1, 1, 1, 0.8]})
        self.lines_quat = []
        self.lines_acc = []
        self.lines_gyro = []
        self.lines_emg = []
        self._title_base = ""

        self.setup_plots()
        self.connect_serial()

        # Auto-start: send gesture command to ESP32 to begin streaming
        if auto_start and self.serial_ok:
            self.fig.canvas.mpl_connect('key_press_event', self.on_key)
            self.start_collection()

    # ---- Serial ----

    def connect_serial(self):
        if not self.port:
            print("WARNING: No serial port specified. Running in offline mode (no data).")
            self.fig.suptitle("IMU Visualizer — NO SERIAL (offline)", color="orange", fontsize=12)
            return

        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0)
            self.ser.reset_input_buffer()
            self.serial_ok = True
            print(f"Connected to {self.port} @ {self.baud} bps")
            self._title_base = f"IMU Visualizer — {self.port}"
            self._update_title()
        except Exception as e:
            print(f"Cannot open serial port {self.port}: {e}")
            print("Continuing in offline mode (no live data).")
            self.fig.suptitle(f"IMU Visualizer — SERIAL FAILED ({self.port})", color="red", fontsize=12)

    def _update_title(self):
        status = "STREAMING" if self.collecting else "PAUSED"
        color = "green" if self.collecting else "gray"
        self.fig.suptitle(f"{self._title_base}  |  {status}  |  Space=toggle  Q=quit",
                          color=color, fontsize=12)

    def start_collection(self):
        """Send gesture '1' command to ESP32 to start data streaming."""
        if self.serial_ok and self.ser and self.ser.is_open:
            self.ser.write(b"1\n")
            self.collecting = True
            self._update_title()
            print("Sent start command to ESP32")

    def stop_collection(self):
        """Send gesture '1' command again to stop data streaming."""
        if self.serial_ok and self.ser and self.ser.is_open:
            self.ser.write(b"1\n")
            self.collecting = False
            self._update_title()
            print("Sent stop command to ESP32")

    def on_key(self, event):
        """Handle key presses in the plot window."""
        if event.key == ' ':
            if self.collecting:
                self.stop_collection()
            else:
                self.start_collection()
        elif event.key in ('q', 'Q'):
            plt.close(self.fig)

    def read_lines(self):
        """Read all available complete lines from serial, return list of parsed IMU/EMG dicts."""
        if not self.serial_ok or not self.ser or not self.ser.is_open:
            return []

        results = []
        while self.ser.in_waiting > 0:
            chunk = self.ser.read(self.ser.in_waiting).decode("utf-8", errors="ignore")
            self._line_buf += chunk

            while "\n" in self._line_buf:
                line, self._line_buf = self._line_buf.split("\n", 1)
                line = line.strip()
                parsed = self.parse_csv_line(line)
                if parsed:
                    results.append(parsed)

        return results

    @staticmethod
    def parse_csv_line(line):
        """Parse a CSV line from the ESP32 firmware.

        IMU format: label,timestamp,IMU,qw,qx,qy,qz,ax,ay,az,gx,gy,gz
        EMG format: label,timestamp,EMG,emg0,...,emg31
        Returns dict or None.
        """
        if not line or line.startswith("#") or line.startswith("label,"):
            return None

        parts = line.split(",")
        if len(parts) < 3:
            return None

        try:
            sensor_type = parts[2].strip().upper()
        except IndexError:
            return None

        if sensor_type == "IMU" and len(parts) >= 13:
            try:
                qw = float(parts[3])
                qx = float(parts[4])
                qy = float(parts[5])
                qz = float(parts[6])
                ax = float(parts[7])
                ay = float(parts[8])
                az = float(parts[9])
                gx = float(parts[10])
                gy = float(parts[11])
                gz = float(parts[12])
                return {
                    "type": "IMU",
                    "quat": [qw, qx, qy, qz],
                    "acc": [ax, ay, az],
                    "gyro": [gx, gy, gz],
                }
            except (ValueError, IndexError):
                return None

        elif sensor_type == "EMG" and len(parts) >= 35:
            try:
                values = [float(parts[3 + i]) for i in range(32)]
                # Group into 4 channels (each channel has 8 samples)
                # Compute RMS for each channel
                rms = []
                for ch in range(4):
                    ch_vals = values[ch * 8:(ch + 1) * 8]
                    rms_val = (sum(v * v for v in ch_vals) / len(ch_vals)) ** 0.5
                    rms.append(rms_val)
                return {"type": "EMG", "emg_rms": rms}
            except (ValueError, IndexError):
                return None

        return None

    # ---- Plot setup ----

    def setup_plots(self):
        colors_quat = ["r", "g", "b", "m"]
        colors_xyz = ["r", "g", "b"]
        colors_emg = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

        # Row 0: Quaternion
        ax = self.axs[0]
        ax.set_title("Quaternion (W, X, Y, Z)")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.set_xlim(0, MAX_POINTS)
        ax.set_ylim(-1.2, 1.2)
        for j, lbl in enumerate(["W", "X", "Y", "Z"]):
            line, = ax.plot([], [], label=lbl, color=colors_quat[j], linewidth=0.8)
            self.lines_quat.append(line)
        ax.legend(loc="upper right", fontsize=8)

        # Row 1: Accelerometer
        ax = self.axs[1]
        ax.set_title("Accelerometer (X, Y, Z)")
        ax.set_ylabel("Value (g)")
        ax.grid(True)
        ax.set_xlim(0, MAX_POINTS)
        ax.set_ylim(-4.0, 4.0)
        for j, lbl in enumerate(["X", "Y", "Z"]):
            line, = ax.plot([], [], label=lbl, color=colors_xyz[j], linewidth=0.8)
            self.lines_acc.append(line)
        ax.legend(loc="upper right", fontsize=8)

        # Row 2: Gyroscope
        ax = self.axs[2]
        ax.set_title("Gyroscope (X, Y, Z)")
        ax.set_ylabel("Value (deg/s)")
        ax.grid(True)
        ax.set_xlim(0, MAX_POINTS)
        ax.set_ylim(-2000.0, 2000.0)
        for j, lbl in enumerate(["X", "Y", "Z"]):
            line, = ax.plot([], [], label=lbl, color=colors_xyz[j], linewidth=0.8)
            self.lines_gyro.append(line)
        ax.legend(loc="upper right", fontsize=8)

        # Row 3: EMG RMS (4 channels)
        ax = self.axs[3]
        ax.set_title("EMG RMS (4 Channels)")
        ax.set_ylabel("RMS")
        ax.set_xlabel("Sample")
        ax.grid(True)
        ax.set_xlim(0, MAX_POINTS)
        ax.set_ylim(0, 140)
        for ch in range(4):
            line, = ax.plot([], [], label=f"CH{ch}", color=colors_emg[ch], linewidth=0.8)
            self.lines_emg.append(line)
        ax.legend(loc="upper right", fontsize=8)

        plt.tight_layout()

    # ---- Animation ----

    def update_axes(self):
        if not self.time_steps:
            return
        x_min = self.time_steps[0]
        x_max = max(self.time_steps[-1], x_min + 1)
        for ax in self.axs:
            ax.set_xlim(x_min, x_max)

    def update_plot(self, frame):
        if not self.serial_ok:
            return self.lines_quat + self.lines_acc + self.lines_gyro + self.lines_emg

        try:
            samples = self.read_lines()
            for data in samples:
                self.counter += 1
                self.time_steps.append(self.counter)

                if data["type"] == "IMU":
                    for i, key in enumerate(["w", "x", "y", "z"]):
                        self.buf_quat[key].append(data["quat"][i])
                    for i, key in enumerate(["x", "y", "z"]):
                        self.buf_acc[key].append(data["acc"][i])
                        self.buf_gyro[key].append(data["gyro"][i])
                    # EMG RMS: hold previous value if no EMG in this tick
                    for ch in range(4):
                        if self.buf_emg_rms[ch]:
                            self.buf_emg_rms[ch].append(self.buf_emg_rms[ch][-1])
                        else:
                            self.buf_emg_rms[ch].append(0)

                elif data["type"] == "EMG":
                    for ch in range(4):
                        self.buf_emg_rms[ch].append(data["emg_rms"][ch])
                    # IMU: hold previous values
                    for key in self.buf_quat:
                        if self.buf_quat[key]:
                            self.buf_quat[key].append(self.buf_quat[key][-1])
                        else:
                            self.buf_quat[key].append(0)
                    for key in self.buf_acc:
                        if self.buf_acc[key]:
                            self.buf_acc[key].append(self.buf_acc[key][-1])
                        else:
                            self.buf_acc[key].append(0)
                    for key in self.buf_gyro:
                        if self.buf_gyro[key]:
                            self.buf_gyro[key].append(self.buf_gyro[key][-1])
                        else:
                            self.buf_gyro[key].append(0)

                # Trim buffer to MAX_POINTS
                if len(self.time_steps) > MAX_POINTS:
                    self.time_steps.pop(0)
                    for buf in [self.buf_quat, self.buf_acc, self.buf_gyro]:
                        for key in buf:
                            if len(buf[key]) > MAX_POINTS:
                                buf[key].pop(0)
                    for ch in range(4):
                        if len(self.buf_emg_rms[ch]) > MAX_POINTS:
                            self.buf_emg_rms[ch].pop(0)

            # Update line data
            for i, key in enumerate(["w", "x", "y", "z"]):
                self.lines_quat[i].set_data(self.time_steps, self.buf_quat[key])
            for i, key in enumerate(["x", "y", "z"]):
                self.lines_acc[i].set_data(self.time_steps, self.buf_acc[key])
                self.lines_gyro[i].set_data(self.time_steps, self.buf_gyro[key])
            for ch in range(4):
                self.lines_emg[ch].set_data(self.time_steps, self.buf_emg_rms[ch])

            self.update_axes()

        except Exception as e:
            print(f"Error: {e}")

        return self.lines_quat + self.lines_acc + self.lines_gyro + self.lines_emg

    def run(self):
        self.ani = animation.FuncAnimation(
            self.fig,
            self.update_plot,
            interval=PLOT_INTERVAL_MS,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()

        if self.ser and self.ser.is_open:
            self.ser.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        port = sys.argv[1]
    else:
        port = choose_port(default=SERIAL_PORT)

    visualizer = IMUVisualizer(port=port)
    visualizer.run()
