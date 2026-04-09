# uart_logger

Reads SEN55 sensor data from a serial/UART connection, buffers it in a pandas
DataFrame, and saves each recording session as a timestamped `.pkl` file.

## Dependencies

```
pip install pyserial pandas
```

## Usage

```bash
python uart_logger.py --port <port> [--baud <rate>] [--out <dir>]
```

| Argument | Default | Description |
|---|---|---|
| `--port` | *(required)* | Serial port — e.g. `/dev/ttyUSB0` (Linux/Mac) or `COM3` (Windows) |
| `--baud` | `115200` | Baud rate — must match `monitor_speed` in `platformio.ini` |
| `--out` | `./data` | Directory where `.pkl` files are written (created if absent) |

### Examples

```bash
# Linux / Mac
python tool/uart_logger.py --port /dev/ttyUSB0

# Windows
python tool/uart_logger.py --port COM3

# Custom output folder
python tool/uart_logger.py --port /dev/ttyUSB0 --out ~/captures
```

Press **Ctrl-C** at any time to stop. Any in-progress recording session is
flushed to disk before the script exits.

## Recording control (firmware sentinels)

The script is **passive** — it does not start recording immediately. Instead,
it waits for control strings sent by the firmware over UART:

| String | Effect |
|---|---|
| `START_RECORDING` | Begin buffering sensor rows |
| `STOP_RECORDING` | Stop buffering and write the session to a `.pkl` file |

Multiple START/STOP pairs in one run each produce a separate file. If
`START_RECORDING` arrives while already recording, the current session is
saved and a new one begins.

### Firmware example

```cpp
Serial.println("START_RECORDING");
// ... sensor loop ...
Serial.println("STOP_RECORDING");
```

## Output format

Each session is saved to:

```
<out>/sen55_YYYY-MM-DD_HH-MM-SS.pkl
```

The file contains a pandas DataFrame with a `DatetimeIndex` (timestamp of each
reading) and the following columns in firmware print order:

| Column | Unit |
|---|---|
| `pm1` | µg/m³ |
| `pm2_5` | µg/m³ |
| `pm4` | µg/m³ |
| `pm10` | µg/m³ |
| `temp` | °C |
| `rh` | % |
| `voc` | index (1–500) |
| `nox` | index (1–500) |

### Loading a file

```python
import pandas as pd

df = pd.read_pickle("data/sen55_2024-03-15_14-30-00.pkl")
print(df.head())
```
