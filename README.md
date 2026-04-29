# EMMS + Reversal Signals — Real-Time MT5 Monitors

Two real-time terminal-table monitors that fetch live OHLCV from MetaTrader 5
for 9 major forex pairs across 7 timeframes and display signal matrices.

| Script | Signals | Table indicators |
|---|---|---|
| `EstimatedManipulationMovementSignal.py` | EMMS manipulation detection | `B` BUY, `S` SELL, `!` anomaly, `·` neutral |
| `ReversalSignals.py` | TD Sequential reversal setups | `B` BUY, `S` SELL, `M` momentum phase, `E` exhaustion phase, `·` neutral |

## Quick-start

```bash
# 1. One-time setup
setup_env.bat                  # Windows
# or:  bash setup_env.sh       # Linux/macOS (without MT5)

# 2. Activate the environment
.venv\Scripts\activate         # Windows
# or:  source .venv/bin/activate

# 3. Run a monitor
python EstimatedManipulationMovementSignal.py --realtime
python ReversalSignals.py --realtime --setup Qualified
```

## Requirements

| Package | Required | Notes |
|---|---|---|
| Python 3.10+ | Yes | `X \| None` type syntax |
| `numpy` | Yes | Signal computation |
| `MetaTrader5` | Yes | Windows-only, MT5 terminal must be running |
| `python-socketio` | Optional | Only with `--socketio` |
| `uvicorn` | Optional | Only with `--socketio` |

Dependencies are split across two requirement files so Socket.IO is never forced:

```
requirements.txt            # numpy, MetaTrader5  (always)
requirements-socketio.txt   # python-socketio, uvicorn  (optional)
```

## Default coverage

**9 pairs:** EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, NZDUSD, EURGBP, EURJPY
**7 timeframes:** 1m, 5m, 15m, 30m, 1H, 4H, 1D
**63 series total** per monitor

## Common CLI flags (both scripts)

| Flag | Default | Description |
|---|---|---|
| `--realtime` | — | Launch the live terminal-table monitor |
| `--interval` | `60` | Refresh interval in seconds |
| `--pairs` | 9 majors | Comma-separated (e.g. `EURUSD,GBPUSD,USDJPY`) |
| `--timeframes` | all 7 | Comma-separated (e.g. `M5,M15,H1`) |
| `--bars` | `1000` | Historical candles to fetch |
| `--workers` | `4` | Thread-pool size for parallel MT5 fetches |
| `--debug` | off | DEBUG-level logging |
| `--socketio` | off | Enable Socket.IO web-server |
| `--socket-port` | `5001`/`5000` | Socket.IO listen port |
| `--mt5` | off | One-shot CSV mode (single symbol/timeframe) |

`ReversalSignals.py` adds:

| Flag | Default | Description |
|---|---|---|
| `--setup` | `Qualified` | `Momentum`, `Exhaustion`, or `Qualified` |

## Examples

```bash
# Real-time terminal table, default 60s refresh
python EstimatedManipulationMovementSignal.py --realtime

# Faster refresh, fewer pairs
python EstimatedManipulationMovementSignal.py --realtime --interval 30 --pairs EURUSD,GBPUSD,AUDUSD

# Momentum-only reversal setups with Socket.IO on port 8080
python ReversalSignals.py --realtime --setup Momentum --socketio --socket-port 8080

# One-shot CSV dump (legacy mode)
python EstimatedManipulationMovementSignal.py --mt5 --symbol EURUSD --timeframe H1
python ReversalSignals.py --mt5 --symbol EURUSD --timeframe H1
```

## Architecture

Both scripts share the same 9‑section layout:

```
1. Constants          — pairs, timeframes, labels
2. Config dataclasses — frozen engine config + mutable monitor config
3. Algorithm engine   — NumPy-based signal computation
4. MT5Connection      — persistent, thread-safe, one-init-per-session
5. SignalMatrix       — grid container for (pair, tf) to snapshot lookup
6. TableRenderer      — Unicode box-drawing terminal table
7. Monitor            — async orchestrator with ThreadPoolExecutor
8. Socket.IO          — optional background ASGI server
9. CLI                — argparse with --realtime, --help, etc.
```

## Socket.IO events (when `--socketio` is on)

| Client emits | Server responds with | Payload |
|---|---|---|
| `get_signals` | `signals` | Flat array: each `{symbol, timeframe, signal, ...}` |
| `get_matrix` | `matrix` | Nested: `pair to timeframe to signal or null` |

Server also emits `status` on connect.
