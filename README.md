# OEPS_commutator_script

Manual commutator control driven by head-orientation data from an [Open Ephys](https://open-ephys.org/) / Miniscope recording.

The script tails the `headOrientation.csv` file written by the Miniscope DAQ software, converts incoming quaternions into a yaw angle, accumulates the (signed) yaw change since the last update, and sends a `{turn : <fraction-of-a-turn>}` command over a serial port to a commutator controller — keeping the tether untwisted in real time.

## What it does

- Finds the newest session folder under a configured base path (today's day-folder by default, optionally any day).
- Tails `<session>/<device>/headOrientation.csv` in append-only mode, surviving file deletion / rotation between sessions (it rescans and re-attaches when a new file appears).
- Computes yaw from `qw, qx, qy, qz`, then sums the smallest signed yaw step between consecutive samples (so wrapping at ±π is handled correctly).
- Applies a deadband (small movements ignored) and a per-chunk clamp (so a single bad sample can't command a full spin).
- Writes `{turn : <value>}\n` over serial (default `COM3`, 9600 baud) where `<value>` is the rotation in **turns** (1.0 = 360°). Sign is inverted to match the commutator's convention.

## Requirements

- Python 3.9+
- `numpy`, `pandas`, `pyserial`

```bash
pip install -r requirements.txt
```

## Usage

Minimal:

```bash
python manual_commutation.py --base "C:\path\to\miniscope\recordings" --port COM3
```

The script attaches to the newest `headOrientation.csv` under `--base` and starts commanding the commutator. New rows print as `SEND: {turn : ...}`. `Ctrl+C` to stop — the serial port is closed cleanly on exit.

Test without hardware (no commutator wired up):

```bash
python manual_commutation.py --base <path-to-recordings> --dry-run
```

Commands print as `DRY: {turn : ...}` instead of being sent over serial.

### All flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--base` | required | Recordings folder containing `YYYY_MM_DD` day-folders. |
| `--port` | required unless `--dry-run` | Serial port (`COM3`, `/dev/ttyUSB0`, …). |
| `--device-folder` | auto-detect | Miniscope device subfolder name. By default the script finds the subfolder containing `headOrientation.csv`. |
| `--baudrate` | `9600` | |
| `--interval` | `0.5` | Polling interval in seconds. |
| `--rescan` | `0.05` | Rescan interval for newer sessions, in seconds. |
| `--any-day` | off | Use the newest session under any day-folder, not just today. Useful if recordings cross midnight. |
| `--deadband` | `1e-4` | Ignore yaw changes smaller than this (in turns). |
| `--max-turns-per-chunk` | `0.25` | Clamp commanded rotation per polling chunk (in turns). |
| `--dry-run` | off | Skip the serial port; print commands instead of sending. |

Run `python manual_commutation.py --help` for the same info inline.

## Expected CSV format

`headOrientation.csv` must have the columns:

```
Time Stamp (ms), qw, qx, qy, qz
```

The header line is auto-detected and skipped.

## License

MIT — see [LICENSE](LICENSE).
