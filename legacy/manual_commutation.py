import time
from dataclasses import dataclass
from datetime import date
from math import pi
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import serial

COLUMNS = ["Time Stamp (ms)", "qw", "qx", "qy", "qz"]


# ---------------- path utilities ----------------
def newest_subdir(path: Path) -> Path:
    subdirs = [p for p in path.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No subfolders in: {path}")
    return max(subdirs, key=lambda p: p.stat().st_mtime)


def resolve_today_csv(base: Path, device_folder: str = "miniscopeDeviceName") -> Path:
    day_dir = base / date.today().strftime("%Y_%m_%d")
    newest_session = newest_subdir(day_dir)
    csv_path = newest_session / device_folder / "headOrientation.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return csv_path


def resolve_newest_csv_any_day(base: Path, device_folder: str = "miniscopeDeviceName") -> Path:
    day_dirs = [p for p in base.iterdir() if p.is_dir()]
    if not day_dirs:
        raise FileNotFoundError(f"No day folders in: {base}")

    newest_day = max(day_dirs, key=lambda p: p.stat().st_mtime)
    newest_session = newest_subdir(newest_day)

    csv_path = newest_session / device_folder / "headOrientation.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return csv_path


# ---------------- yaw + unwrap ----------------
def yaw_from_quat_df(df: pd.DataFrame) -> np.ndarray:
    w = df["qw"].to_numpy(float)
    x = df["qx"].to_numpy(float)
    y = df["qy"].to_numpy(float)
    z = df["qz"].to_numpy(float)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def nearest_wrap(curr: float, last: float) -> float:
    return min((curr, curr + 2.0 * pi, curr - 2.0 * pi), key=lambda v: abs(v - last))


@dataclass
class TurnState:
    last_p: Optional[float] = None

from math import pi

TWOPI = 2.0 * pi

def wrap_to_pi(x: float) -> float:
    # in [-pi, pi)
    return (x + pi) % (2.0 * pi) - pi

def combined_turn_command(df_new, state, deadband_turns=0.0, max_turns_per_chunk=0.25):
    if df_new is None or df_new.empty:
        return None

    yaw = yaw_from_quat_df(df_new)
    if yaw.size == 0:
        return None

    # initialize last yaw
    if state.last_p is None:
        state.last_p = float(yaw[0])
        yaw = yaw[1:]
        if yaw.size == 0:
            return None

    last = float(state.last_p)
    total_turn = 0.0

    for curr in yaw:
        d = wrap_to_pi(float(curr) - last)   # always the smallest signed step
        total_turn += d / TWOPI
        last = last + d                      # keep an unwrapped running last

    state.last_p = last

    # deadband
    if abs(total_turn) < deadband_turns:
        return None

    # safety clamp (prevents "one whole spin" mistakes)
    if abs(total_turn) > max_turns_per_chunk:
        # either clamp or drop; clamping is usually better for commutators
        total_turn = float(np.clip(total_turn, -max_turns_per_chunk, max_turns_per_chunk))

    total_turn = -total_turn  # keep your convention
    return f"{{turn : {total_turn}}}"



# ---------------- CSV tailer (robust to missing/rotated files) ----------------
class CSVTailer:
    """
    Append-only CSV tailer that won't crash on missing files.
    If the file disappears, read_new_rows() raises FileNotFoundError and the caller can rescan/switch.
    """

    def __init__(self, path: Path, has_header: bool = True, encoding: str = "utf-8"):
        self.path = Path(path)
        self.has_header = has_header
        self.encoding = encoding
        self._offset = 0
        self._partial = b""
        self._bootstrapped = False

    def set_path(self, path: Path) -> None:
        """Switch to a new file and reset tail state (start at EOF)."""
        self.path = Path(path)
        self._offset = 0
        self._partial = b""
        self._bootstrapped = False
        self.bootstrap_to_eof()

    def bootstrap_to_eof(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"CSV not found: {self.path}")
        with open(self.path, "rb") as f:
            f.seek(0, 2)
            self._offset = f.tell()
        self._bootstrapped = True

    def read_new_rows(self) -> pd.DataFrame:
        if not self._bootstrapped:
            self.bootstrap_to_eof()

        if not self.path.exists():
            raise FileNotFoundError(f"CSV not found: {self.path}")

        try:
            with open(self.path, "rb") as f:
                f.seek(self._offset)
                data = f.read()
                self._offset = f.tell()
        except FileNotFoundError:
            # race: file got removed between exists() and open()
            raise FileNotFoundError(f"CSV not found: {self.path}")

        if not data:
            return pd.DataFrame(columns=COLUMNS)

        buf = self._partial + data
        lines = buf.split(b"\n")
        self._partial = lines[-1]
        complete = lines[:-1]

        rows = []
        for raw in complete:
            raw = raw.strip()
            if not raw:
                continue
            s = raw.decode(self.encoding, errors="ignore").strip()

            if self.has_header and (s.startswith("Time Stamp") or s.startswith(COLUMNS[0])):
                continue

            parts = s.split(",")
            if len(parts) < 5:
                continue

            try:
                ts, qw, qx, qy, qz = parts[:5]
                rows.append([float(ts), float(qw), float(qx), float(qy), float(qz)])
            except ValueError:
                continue

        return pd.DataFrame(rows, columns=COLUMNS) if rows else pd.DataFrame(columns=COLUMNS)


# ---------------- commutator controller ----------------
class CommutatorController:
    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 1.0):
        self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        time.sleep(0.1)
        self.state = TurnState()

    def close(self) -> None:
        try:
            self.ser.close()
        except Exception:
            pass

    def process_df(self, df_new: pd.DataFrame, deadband_turns: float = 1e-4) -> None:
        cmd = combined_turn_command(df_new, self.state, deadband_turns=deadband_turns)
        if cmd is None:
            return
        self.ser.write((cmd + "\n").encode("ascii"))
        print("SEND:", cmd)


# ---------------- main loop: handles file deletion by rescanning ----------------
def run_polling_with_file_resilience(
    base: Path,
    controller: CommutatorController,
    interval_s: float = 0.05,
    rescan_every_s: float = 0.05,
    device_folder: str = "miniscopeDeviceName",
    any_day: bool = False,
) -> None:
    """
    Continuously tail newest CSV.
    If the current file disappears (deleted/rotated/moved), immediately rescan and switch.
    Also periodically rescans in case a newer session starts.
    """
    def find_csv() -> Path:
        return (resolve_newest_csv_any_day if any_day else resolve_today_csv)(
            base, device_folder=device_folder
        )

    # initial attach (wait until something exists)
    csv_path = None
    while csv_path is None:
        try:
            csv_path = find_csv()
            print("Tailing:", csv_path)
        except FileNotFoundError:
            time.sleep(1.0)

    tailer = CSVTailer(csv_path, has_header=True)
    tailer.bootstrap_to_eof()

    last_rescan = time.monotonic()

    while True:
        # 1) read new rows; if file is gone, rescan until we find a valid one
        try:
            df_new = tailer.read_new_rows()
        except FileNotFoundError:
            csv_path = None
            while csv_path is None:
                try:
                    csv_path = find_csv()
                    print("File gone. Switching to:", csv_path)
                    tailer.set_path(csv_path)
                except FileNotFoundError:
                    time.sleep(0.2)
            time.sleep(interval_s)
            continue

        if not df_new.empty:
            controller.process_df(df_new)

        # 2) periodic rescan for a newer file (new session started)
        now = time.monotonic()
        if (now - last_rescan) >= rescan_every_s:
            last_rescan = now
            try:
                newest_csv = find_csv()
                if newest_csv != tailer.path:
                    print("Newer session detected. Switching to:", newest_csv)
                    tailer.set_path(newest_csv)
            except FileNotFoundError:
                pass

        time.sleep(interval_s)


if __name__ == "__main__":
    base = Path(r"C:\path\to\miniscope\recordings")
    ctl = CommutatorController(port="COM3", baudrate=9600, timeout=1)

    try:
        run_polling_with_file_resilience(
            base=base,
            controller=ctl,
            interval_s=0.5,
            rescan_every_s=0.05,
            device_folder="miniscopeDeviceName",
            any_day=False,  # True if midnight/day-folder mismatch can happen
        )
    finally:
        ctl.close()
