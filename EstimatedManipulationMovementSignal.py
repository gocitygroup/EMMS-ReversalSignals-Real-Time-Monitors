"""
Estimated Manipulation Movement Signal (EMMS) — Real-Time MT5 Monitor.

Fetches live OHLCV from MetaTrader 5 for 9 major forex pairs across
7 timeframes (1m → D1), computes EMMS manipulation signals, and displays
a refreshable signal matrix in the terminal every 60 seconds.

Usage:
    python EstimatedManipulationMovementSignal.py --realtime
    python EstimatedManipulationMovementSignal.py --realtime --socketio
    python EstimatedManipulationMovementSignal.py --realtime --interval 30
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

# 9 major forex pairs — high liquidity, universally tradeable
DEFAULT_PAIRS: Tuple[str, ...] = (
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
    "USDCAD", "NZDUSD", "EURGBP", "EURJPY",
)

# 7 timeframes from fastest to slowest
DEFAULT_TIMEFRAMES: Tuple[str, ...] = (
    "M1", "M5", "M15", "M30", "H1", "H4", "D1",
)

TF_LABELS: Dict[str, str] = {
    "M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m",
    "H1": "1H", "H4": "4H", "D1": "1D",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class EMMSConfig:
    """Hyperparameters for the EMMS signal computation."""

    atm_lookback: int = 21
    volatility_multiplier: float = 1.9
    volume_multiplier: float = 1.618
    lbm_pivot_lookback: int = 55
    use_eam_confirmation: bool = False
    label_atr_period: int = 30


@dataclass
class MonitorConfig:
    """Top-level configuration for the real-time monitor."""

    pairs: Tuple[str, ...] = DEFAULT_PAIRS
    timeframes: Tuple[str, ...] = DEFAULT_TIMEFRAMES
    bars: int = 1000
    update_interval: float = 60.0
    emms: EMMSConfig = field(default_factory=EMMSConfig)
    fetch_workers: int = 4
    enable_socketio: bool = False
    socketio_host: str = "localhost"
    socketio_port: int = 5001


# ═══════════════════════════════════════════════════════════════════════════════
# NumPy computation engine  (kept from original — solid, DRY, pure functions)
# ═══════════════════════════════════════════════════════════════════════════════


def _validate_arrays(data: Dict[str, np.ndarray]) -> None:
    required = ("open", "high", "low", "close", "volume")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Missing OHLCV arrays: {missing}")
    lengths = {data[k].shape[0] for k in required}
    if len(lengths) != 1:
        raise ValueError("OHLCV arrays must have equal length.")
    if not lengths or next(iter(lengths)) == 0:
        raise ValueError("Input arrays are empty.")


def _shift(arr: np.ndarray) -> np.ndarray:
    """Lag array by one position; first element becomes NaN."""
    out = np.empty_like(arr, dtype=float)
    out[0] = np.nan
    out[1:] = arr[:-1]
    return out


def _sma(arr: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average via cumulative-sum trick (O(n))."""
    out = np.full(arr.shape, np.nan, dtype=float)
    if period <= 0 or period > arr.size:
        return out
    csum = np.cumsum(np.insert(arr.astype(float), 0, 0.0))
    out[period - 1:] = (csum[period:] - csum[:-period]) / period
    return out


def _rma(arr: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothed moving average (RMA / EMA alpha=1/period)."""
    out = np.full(arr.shape, np.nan, dtype=float)
    if period <= 0 or period > arr.size:
        return out
    out[period - 1] = np.mean(arr[:period])
    alpha = 1.0 / period
    for i in range(period, arr.size):
        out[i] = out[i - 1] + alpha * (arr[i] - out[i - 1])
    return out


def _true_range(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    """True Range: max(H-L, |H-prevC|, |L-prevC|)."""
    prev_c = _shift(c)
    return np.nanmax(
        np.stack((h - l, np.abs(h - prev_c), np.abs(l - prev_c)), axis=0), axis=0
    )


def _atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int) -> np.ndarray:
    """Average True Range via Wilder's RMA."""
    return _rma(_true_range(h, l, c), period)


def _ffill(arr: np.ndarray) -> np.ndarray:
    """Forward-fill NaN values (carry last valid forward)."""
    out = arr.copy().astype(float)
    valid = ~np.isnan(out)
    if not np.any(valid):
        return out
    idx = np.where(valid, np.arange(out.size), 0)
    np.maximum.accumulate(idx, out=idx)
    out = out[idx]
    out[: np.argmax(valid)] = np.nan
    return out


def _pivot_high(series: np.ndarray, left: int, right: int) -> np.ndarray:
    """Detect swing-high pivots: highest point in [i-left, i+right]."""
    out = np.full(series.shape, np.nan, dtype=float)
    for i in range(left, series.size - right):
        window = series[i - left : i + right + 1]
        if np.isclose(series[i], np.nanmax(window), rtol=1e-12, atol=1e-12):
            out[i] = series[i]
    return out


def _pivot_low(series: np.ndarray, left: int, right: int) -> np.ndarray:
    """Detect swing-low pivots: lowest point in [i-left, i+right]."""
    out = np.full(series.shape, np.nan, dtype=float)
    for i in range(left, series.size - right):
        window = series[i - left : i + right + 1]
        if np.isclose(series[i], np.nanmin(window), rtol=1e-12, atol=1e-12):
            out[i] = series[i]
    return out


# ── public API ────────────────────────────────────────────────────────────────


def compute_emms(
    data: Dict[str, np.ndarray], cfg: EMMSConfig | None = None
) -> Dict[str, np.ndarray]:
    """
    Compute the full EMMS signal suite from OHLCV arrays.

    Required keys in `data`: ``open``, ``high``, ``low``, ``close``, ``volume``.
    Returns 20+ computed arrays keyed by name (see body).
    """
    cfg = cfg or EMMSConfig()
    _validate_arrays(data)

    o = data["open"].astype(float, copy=False)
    h = data["high"].astype(float, copy=False)
    l = data["low"].astype(float, copy=False)
    c = data["close"].astype(float, copy=False)
    v = data["volume"].astype(float, copy=False)

    # ── ATM: volatility + volume spikes ─────────────────────────────────
    atr_atm = _atr(h, l, c, cfg.atm_lookback)
    avg_vol = _sma(v, cfg.atm_lookback)
    candle_range = h - l

    is_vol_spike = candle_range > (atr_atm * cfg.volatility_multiplier)
    is_vol_spike = is_vol_spike & (candle_range > 0)
    is_vol_spike = is_vol_spike & (v > (avg_vol * cfg.volume_multiplier))
    is_anomaly = is_vol_spike  # alias — both conditions are AND-ed above

    is_vol_spike_only = candle_range > (atr_atm * cfg.volatility_multiplier)
    is_vol_spike_vol = v > (avg_vol * cfg.volume_multiplier)

    # ── LBM: pivot-based liquidity levels ───────────────────────────────
    lb = cfg.lbm_pivot_lookback
    ph = _pivot_high(h, lb, lb)
    pl = _pivot_low(l, lb, lb)
    last_ph = _ffill(ph)
    last_pl = _ffill(pl)

    # ── Absorption detection ────────────────────────────────────────────
    safe_range = candle_range.copy()
    safe_range[safe_range == 0.0] = np.nan
    close_pos = (c - l) / safe_range

    is_red = c < o
    is_green = c > o

    is_bull_abs = is_red & is_vol_spike_vol & (close_pos > 0.5)
    is_bear_abs = is_green & is_vol_spike_vol & (close_pos < 0.5)

    # ── Confirmation gate ───────────────────────────────────────────────
    if cfg.use_eam_confirmation:
        long_gate, short_gate = is_bull_abs, is_bear_abs
    else:
        long_gate = short_gate = np.ones_like(is_anomaly, dtype=bool)

    # ── Final signals ───────────────────────────────────────────────────
    long_sig = is_anomaly & is_red & (l <= last_pl) & (c >= last_pl) & long_gate
    short_sig = is_anomaly & is_green & (h >= last_ph) & (c <= last_ph) & short_gate

    label_atr_val = _atr(h, l, c, cfg.label_atr_period)

    return {
        # passthrough
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": v,
        # ATM
        "atr_atm": atr_atm,
        "avg_volume": avg_vol,
        "is_volatility_spike": is_vol_spike_only,
        "is_volume_spike": is_vol_spike_vol,
        "is_anomaly_candle": is_anomaly,
        # LBM
        "pivot_high_price": ph,
        "pivot_low_price": pl,
        "last_pivot_high": last_ph,
        "last_pivot_low": last_pl,
        # absorption
        "is_bullish_absorption": is_bull_abs,
        "is_bearish_absorption": is_bear_abs,
        # signals
        "long_signal": long_sig,
        "short_signal": short_sig,
        # labels
        "y1": l - label_atr_val,
        "y2": h + label_atr_val,
    }


def latest_signal(result: Dict[str, np.ndarray]) -> Optional[str]:
    """Return ``"BUY"``, ``"SELL"``, or ``None`` for the latest candle."""
    long_sig, short_sig = result["long_signal"], result["short_signal"]
    mask = long_sig | short_sig
    if not np.any(mask):
        return None
    idx = int(np.flatnonzero(mask)[-1])
    return "BUY" if bool(long_sig[idx]) else "SELL"


def latest_anomaly(result: Dict[str, np.ndarray]) -> bool:
    """Return True if the last candle is an anomaly candle."""
    arr = result["is_anomaly_candle"]
    return bool(arr[-1]) if np.any(arr) else False


# ═══════════════════════════════════════════════════════════════════════════════
# MT5 Connection Manager  (persistent, thread-safe)
# ═══════════════════════════════════════════════════════════════════════════════


class MT5ConnectionError(Exception):
    """Raised when MT5 connection operations fail."""


class MT5Connection:
    """
    Persistent, thread-safe connection to a local MetaTrader 5 terminal.

    Usage::

        conn = MT5Connection()
        conn.connect()
        data = conn.fetch("EURUSD", "H1", bars=1000)
        conn.disconnect()
    """

    def __init__(self) -> None:
        self._mt5: Any = None
        self._lock = threading.Lock()
        self._tf_map: Dict[str, int] = {}
        self._connected = False

    # ── lifecycle ──────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Establish connection. Must be called once before any fetch."""
        try:
            import MetaTrader5 as mt5  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "MetaTrader5 package required.  Install:  pip install MetaTrader5"
            ) from exc

        with self._lock:
            self._mt5 = mt5
            if not mt5.initialize():
                raise MT5ConnectionError(
                    "MT5 initialize() failed — ensure MetaTrader 5 is running "
                    "and algorithmic trading is enabled "
                    "(Tools → Options → Expert Advisors → Allow Algo Trading)."
                )
            self._tf_map = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1,
            }
            self._connected = True

    def disconnect(self) -> None:
        """Shut down the MT5 connection gracefully."""
        with self._lock:
            if self._mt5 is not None:
                self._mt5.shutdown()
            self._mt5 = None
            self._connected = False
            self._tf_map.clear()

    @property
    def connected(self) -> bool:
        return self._connected

    # ── data fetching ──────────────────────────────────────────────────────

    def fetch(
        self, symbol: str, timeframe: str, bars: int
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Fetch the *bars* most recent OHLCV candles for one (symbol, timeframe).

        Returns ``None`` when the symbol or timeframe is unavailable in the
        terminal (e.g. not subscribed, wrong broker).
        Thread-safe.
        """
        if not self._connected:
            raise MT5ConnectionError("Not connected.  Call connect() first.")

        tf = self._tf_map.get(timeframe)
        if tf is None:
            return None

        with self._lock:
            rates = self._mt5.copy_rates_from_pos(symbol, tf, 0, bars)

        if rates is None or len(rates) == 0:
            return None

        return {
            "open": rates["open"].astype(float, copy=False),
            "high": rates["high"].astype(float, copy=False),
            "low": rates["low"].astype(float, copy=False),
            "close": rates["close"].astype(float, copy=False),
            "volume": rates["tick_volume"].astype(float, copy=False),
            "timestamp": rates["time"].astype(np.int64, copy=False),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Signal matrix  (grid container — rows = pairs, cols = timeframes)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PairSnapshot:
    """Result for one (symbol, timeframe) after a fetch + compute cycle."""

    symbol: str
    timeframe: str
    signal: Optional[str]  # "BUY", "SELL", or None
    anomaly: bool
    close: Optional[float]
    timestamp: Optional[int]


class SignalMatrix:
    """
    Grid of latest EMMS signals: rows = pairs, cols = timeframes.

    Provides efficient lookup by (symbol, timeframe) and renders via
    :class:`TableRenderer`.
    """

    def __init__(
        self, pairs: Tuple[str, ...], timeframes: Tuple[str, ...]
    ) -> None:
        self.pairs = pairs
        self.timeframes = timeframes
        self._grid: Dict[Tuple[str, str], PairSnapshot] = {}

    def update(self, snapshot: PairSnapshot) -> None:
        self._grid[(snapshot.symbol, snapshot.timeframe)] = snapshot

    def get(self, symbol: str, timeframe: str) -> Optional[PairSnapshot]:
        return self._grid.get((symbol, timeframe))

    def to_rows(self) -> List[List[PairSnapshot]]:
        """Return ``grid[row][col]`` for table rendering (filled with empty
        snapshots where no data exists)."""
        rows: List[List[PairSnapshot]] = []
        for pair in self.pairs:
            row = []
            for tf in self.timeframes:
                snap = self._grid.get((pair, tf))
                row.append(snap if snap is not None else PairSnapshot(pair, tf, None, False, None, None))
            rows.append(row)
        return rows

    @property
    def total_signals(self) -> int:
        return sum(1 for v in self._grid.values() if v.signal is not None)

    @property
    def total_anomalies(self) -> int:
        return sum(1 for v in self._grid.values() if v.anomaly)

    @property
    def populated(self) -> int:
        """Number of cells that have received at least one update."""
        return len(self._grid)


# ═══════════════════════════════════════════════════════════════════════════════
# Terminal table renderer
# ═══════════════════════════════════════════════════════════════════════════════


class TableRenderer:
    """
    Renders the EMMS signal matrix as a Unicode box-drawing table.

    Cell legend::

        B  — BUY signal on latest candle
        S  — SELL signal on latest candle
        !  — anomaly candle (no directional signal)
        ·  — neutral / no anomaly
    """

    PAIR_COL_WIDTH = 10
    CELL_WIDTH = 6

    def __init__(
        self, pairs: Tuple[str, ...], timeframes: Tuple[str, ...]
    ) -> None:
        self.pairs = pairs
        self.timeframes = timeframes
        self.tf_labels = [TF_LABELS.get(tf, tf) for tf in timeframes]

    # ── public render ──────────────────────────────────────────────────────

    def render(self, matrix: SignalMatrix) -> str:
        """Return the full table as a string (no status line)."""
        rows = matrix.to_rows()
        lines: List[str] = []
        lines.append(self._sep("┌", "┬"))
        lines.append(self._header())
        lines.append(self._sep("├", "┼"))
        for pair, row in zip(self.pairs, rows):
            lines.append(self._data_row(pair, row))
        lines.append(self._sep("└", "┴"))
        return "\n".join(lines)

    def render_summary(
        self, matrix: SignalMatrix, elapsed: float, cycle: int
    ) -> str:
        """Full display: table + status line with timing."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        table = self.render(matrix)
        status = (
            f"Update #{cycle} | {now} | "
            f"Fetched {matrix.populated} cells in {elapsed:.1f}s | "
            f"Signals: {matrix.total_signals} | "
            f"Anomalies: {matrix.total_anomalies}"
        )
        return f"{table}\n{status}"

    # ── internal helpers ───────────────────────────────────────────────────

    def _header(self) -> str:
        hdr = f"│ {'Pair':<{self.PAIR_COL_WIDTH - 2}}"
        for label in self.tf_labels:
            hdr += f" │ {label:^{self.CELL_WIDTH - 2}}"
        hdr += " │"
        return hdr

    def _data_row(self, pair: str, cells: List[PairSnapshot]) -> str:
        line = f"│ {pair:<{self.PAIR_COL_WIDTH - 2}}"
        for cell in cells:
            line += f" │ {self._symbol(cell):^{self.CELL_WIDTH - 2}}"
        line += " │"
        return line

    @staticmethod
    def _symbol(cell: PairSnapshot) -> str:
        if cell.signal == "BUY":
            return "B"
        if cell.signal == "SELL":
            return "S"
        if cell.anomaly:
            return "!"
        return "·"  # middle dot

    def _sep(self, left: str, joint: str) -> str:
        if joint == "┬":
            right = "┐"
        elif joint == "┼":
            right = "┤"
        else:
            right = "┘"
        result = left + "─" * self.PAIR_COL_WIDTH
        for _ in self.timeframes:
            result += joint + "─" * self.CELL_WIDTH
        result += right
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# EMMS Real-Time Monitor  (main orchestrator)
# ═══════════════════════════════════════════════════════════════════════════════


class EMMSMonitor:
    """
    Real-time pipeline orchestrator:

        MT5 → [ThreadPool] → compute_emms() → SignalMatrix → terminal table

    Fetches all (pair × timeframe) combinations concurrently via a
    :class:`ThreadPoolExecutor`, computes signals, and refreshes the
    terminal table on a fixed interval (default 60 s).
    """

    def __init__(
        self,
        config: MonitorConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.cfg = config or MonitorConfig()
        self.logger = logger or logging.getLogger("emms")
        self.connection = MT5Connection()
        self.matrix = SignalMatrix(self.cfg.pairs, self.cfg.timeframes)
        self.renderer = TableRenderer(self.cfg.pairs, self.cfg.timeframes)
        self._executor: Optional[ThreadPoolExecutor] = None
        self._cycle = 0
        self._running = False

        # Optional Socket.IO — set up lazily
        self._sio: Any = None
        self._sio_app: Any = None

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Start the monitor.  Blocks until interrupted (Ctrl+C)."""
        self._running = True
        self._executor = ThreadPoolExecutor(max_workers=self.cfg.fetch_workers)
        self.connection.connect()

        total = len(self.cfg.pairs) * len(self.cfg.timeframes)
        self.logger.info(
            "MT5 connected — %d pairs × %d timeframes = %d series, "
            "%d fetch workers",
            len(self.cfg.pairs),
            len(self.cfg.timeframes),
            total,
            self.cfg.fetch_workers,
        )

        try:
            while self._running:
                await self._tick()
                await asyncio.sleep(self.cfg.update_interval)
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            if self._executor is not None:
                self._executor.shutdown(wait=False)
            self.connection.disconnect()
            self.logger.info("Monitor stopped.")

    def stop(self) -> None:
        self._running = False

    # ── fetch cycle ────────────────────────────────────────────────────────

    async def _tick(self) -> None:
        self._cycle += 1
        t0 = time.monotonic()

        snapshots = await self._fetch_all()
        for snap in snapshots:
            self.matrix.update(snap)

        elapsed = time.monotonic() - t0
        self._display(elapsed)

    async def _fetch_all(self) -> List[PairSnapshot]:
        """Submit all (pair, tf) combos to the thread pool, collect results."""
        loop = asyncio.get_running_loop()
        combos: List[Tuple[str, str]] = [
            (p, tf) for p in self.cfg.pairs for tf in self.cfg.timeframes
        ]

        # Wrap each sync call in loop.run_in_executor for true async yield
        tasks = [
            loop.run_in_executor(self._executor, self._fetch_one, symbol, tf)
            for symbol, tf in combos
        ]

        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        results: List[PairSnapshot] = []
        for item in gathered:
            if isinstance(item, Exception):
                self.logger.debug("Fetch failed: %s", item)
            elif item is not None:
                results.append(item)
        return results

    def _fetch_one(
        self, symbol: str, timeframe: str
    ) -> Optional[PairSnapshot]:
        """Fetch + compute for a single (symbol, timeframe). Runs in thread."""
        data = self.connection.fetch(symbol, timeframe, self.cfg.bars)
        if data is None:
            return PairSnapshot(symbol, timeframe, None, False, None, None)

        result = compute_emms(data, self.cfg.emms)
        signal = latest_signal(result)
        anomaly = latest_anomaly(result)
        close = float(result["close"][-1])

        ts_arr = data.get("timestamp")
        ts = int(ts_arr[-1]) if ts_arr is not None and len(ts_arr) > 0 else None

        return PairSnapshot(
            symbol=symbol,
            timeframe=timeframe,
            signal=signal,
            anomaly=anomaly,
            close=close,
            timestamp=ts,
        )

    # ── display ────────────────────────────────────────────────────────────

    def _display(self, elapsed: float) -> None:
        _ensure_utf8_stdout()
        output = self.renderer.render_summary(
            self.matrix, elapsed, self._cycle
        )
        sys.stdout.write("\033[2J\033[H")  # clear screen, cursor home
        sys.stdout.write(output + "\n")
        sys.stdout.flush()

    # ── optional Socket.IO ─────────────────────────────────────────────────

    def setup_socketio(self) -> None:
        """Configure Socket.IO server (fires on ``--socketio`` flag)."""
        try:
            import socketio  # type: ignore
        except ImportError:
            self.logger.warning(
                "python-socketio not installed — skipping Socket.IO server"
            )
            return

        sio = socketio.AsyncServer(cors_allowed_origins="*")
        app = socketio.ASGIApp(sio)

        @sio.event
        async def connect(sid, environ):
            await sio.emit("status", {"message": "Connected to EMMS"}, room=sid)

        @sio.event
        async def get_signals(sid, data):
            rows = self.matrix.to_rows()
            flat = [
                {
                    "symbol": c.symbol,
                    "timeframe": c.timeframe,
                    "signal": c.signal,
                    "anomaly": c.anomaly,
                    "close": c.close,
                }
                for row in rows
                for c in row
            ]
            await sio.emit("signals", flat, room=sid)

        @sio.event
        async def get_matrix(sid, data):
            """Return the full grid suitable for heatmap UIs."""
            payload: Dict[str, Dict[str, Optional[str]]] = {}
            for pair in self.cfg.pairs:
                payload[pair] = {}
                for tf in self.cfg.timeframes:
                    snap = self.matrix.get(pair, tf)
                    payload[pair][tf] = snap.signal if snap else None
            await sio.emit("matrix", payload, room=sid)

        self._sio = sio
        self._sio_app = app
        self.logger.info(
            "Socket.IO ready on %s:%d",
            self.cfg.socketio_host,
            self.cfg.socketio_port,
        )

    async def serve_socketio(self) -> None:
        """Run the Socket.IO ASGI server (blocking asyncio task)."""
        if self._sio_app is None:
            self.logger.warning(
                "Socket.IO not configured. Call setup_socketio() first."
            )
            return
        try:
            import uvicorn  # type: ignore
        except ImportError:
            self.logger.warning(
                "uvicorn not installed — cannot serve Socket.IO"
            )
            return

        config = uvicorn.Config(
            self._sio_app,
            host=self.cfg.socketio_host,
            port=self.cfg.socketio_port,
            log_level="info",
        )
        await uvicorn.Server(config).serve()


# ═══════════════════════════════════════════════════════════════════════════════
# Terminal encoding helper  (Windows → UTF‑8)
# ═══════════════════════════════════════════════════════════════════════════════

_UTF8_OK = False


def _ensure_utf8_stdout() -> None:
    """Reconfigure stdout for UTF‑8 so box‑drawing glyphs render."""
    global _UTF8_OK
    if _UTF8_OK:
        return
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    _UTF8_OK = True


# ═══════════════════════════════════════════════════════════════════════════════
# CSV helpers  (kept minimal — legacy one-shot mode)
# ═══════════════════════════════════════════════════════════════════════════════


def _load_csv(path: Path) -> Dict[str, np.ndarray]:
    arr = np.genfromtxt(
        path, delimiter=",", names=True, dtype=None, encoding="utf-8"
    )
    if arr.size == 0:
        raise ValueError("CSV is empty.")
    if arr.ndim == 0:
        arr = np.array([arr], dtype=arr.dtype)
    cols = {n.lower(): n for n in (arr.dtype.names or ())}
    return {
        k: np.asarray(arr[cols[k]], dtype=float)
        for k in ("open", "high", "low", "close", "volume")
    }


def _save_csv(result: Dict[str, np.ndarray], path: Path) -> None:
    long_sig, short_sig = result["long_signal"], result["short_signal"]
    mask = long_sig | short_sig
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["open", "high", "low", "close", "volume", "long_signal", "short_signal"]
        )
        for i in np.flatnonzero(mask):
            w.writerow(
                [
                    float(result["open"][i]),
                    float(result["high"][i]),
                    float(result["low"][i]),
                    float(result["close"][i]),
                    float(result["volume"][i]),
                    bool(long_sig[i]),
                    bool(short_sig[i]),
                ]
            )
    print(
        f"Saved {int(np.count_nonzero(mask))} signal rows → {path.resolve()}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="EMMS Real-Time Monitor — live MT5 signal matrix.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples
            --------
              %(prog)s --realtime
              %(prog)s --realtime --interval 30
              %(prog)s --realtime --socketio --socket-port 8080
              %(prog)s --realtime --pairs EURUSD,GBPUSD,USDJPY \\
                  --timeframes M5,M15,H1
              %(prog)s --mt5 --symbol EURUSD --timeframe H1   # one-shot CSV
        """),
    )

    # ── real-time mode ──────────────────────────────────────────────────
    p.add_argument(
        "--realtime",
        action="store_true",
        help="Run the live terminal-table monitor (default interval: 60 s).",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="Update interval in seconds (default: 60).",
    )

    # ── pair / timeframe selection ──────────────────────────────────────
    p.add_argument(
        "--pairs",
        type=str,
        default=None,
        help="Comma-separated pairs (default: 9 majors).",
    )
    p.add_argument(
        "--timeframes",
        type=str,
        default=None,
        help="Comma-separated timeframes (default: M1,M5,M15,M30,H1,H4,D1).",
    )
    p.add_argument(
        "--symbol",
        type=str,
        default="EURUSD",
        help="Single symbol (one-shot / CSV mode).",
    )
    p.add_argument(
        "--timeframe",
        type=str,
        default="H1",
        help="Single timeframe (one-shot / CSV mode).",
    )

    # ── data source ─────────────────────────────────────────────────────
    p.add_argument(
        "--mt5",
        action="store_true",
        help="Fetch from MT5 (one-shot mode, no real-time loop).",
    )
    p.add_argument(
        "--bars", type=int, default=1000, help="Candles to fetch (default: 1000)."
    )
    p.add_argument("--input", type=str, default="data.csv")
    p.add_argument("--output", type=str, default="emms_signals.csv")

    # ── Socket.IO ───────────────────────────────────────────────────────
    p.add_argument(
        "--socketio",
        action="store_true",
        help="Enable Socket.IO server alongside the monitor.",
    )
    p.add_argument(
        "--socket-host", type=str, default="localhost"
    )
    p.add_argument(
        "--socket-port", type=int, default=5001
    )

    # ── tuning ──────────────────────────────────────────────────────────
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Thread pool size for parallel MT5 fetches (default: 4).",
    )
    p.add_argument(
        "--debug", action="store_true", help="DEBUG-level logging."
    )

    return p


def _parse_csv_arg(
    value: str | None, fallback: Tuple[str, ...]
) -> Tuple[str, ...]:
    if value is None:
        return fallback
    return tuple(s.strip() for s in value.split(",") if s.strip())


def main() -> None:
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("emms")

    # ═════════════════════════════════════════════════════════════════════
    # Real-time mode
    # ═════════════════════════════════════════════════════════════════════

    if args.realtime:
        pairs = _parse_csv_arg(args.pairs, DEFAULT_PAIRS)
        timeframes = _parse_csv_arg(args.timeframes, DEFAULT_TIMEFRAMES)
        config = MonitorConfig(
            pairs=pairs,
            timeframes=timeframes,
            bars=args.bars,
            update_interval=max(1.0, args.interval),
            fetch_workers=args.workers,
            enable_socketio=args.socketio,
            socketio_host=args.socket_host,
            socketio_port=args.socket_port,
        )

        monitor = EMMSMonitor(config, logger)

        async def _run() -> None:
            if config.enable_socketio:
                monitor.setup_socketio()
                # Run Socket.IO server in background
                asyncio.create_task(monitor.serve_socketio())

            try:
                await monitor.run()
            except KeyboardInterrupt:
                monitor.stop()

        try:
            asyncio.run(_run())
        except KeyboardInterrupt:
            print("\n[INFO] Shut down.")
        return

    # ═════════════════════════════════════════════════════════════════════
    # One-shot / CSV mode  (backward-compatible)
    # ═════════════════════════════════════════════════════════════════════

    input_path = Path(args.input)
    output_path = Path(args.output)
    symbol = args.symbol
    timeframe = args.timeframe

    if args.mt5 or not input_path.exists():
        print(
            f"[INFO] Fetching {symbol} {timeframe} ({args.bars} bars) "
            f"from MT5..."
        )
        conn = MT5Connection()
        try:
            conn.connect()
            data = conn.fetch(symbol, timeframe, args.bars)
        finally:
            conn.disconnect()

        if data is None:
            print(
                f"[ERROR] No data for {symbol} {timeframe}. "
                f"Check symbol name and market watch."
            )
            return

        result = compute_emms(data)
        _save_csv(result, output_path)
        signal = latest_signal(result)
        anomaly = latest_anomaly(result)
        print(
            f"{symbol} {timeframe}:  "
            f"signal={signal or '—'},  "
            f"anomaly={anomaly},  "
            f"close={float(result['close'][-1]):.5f}"
        )
        return

    # Pure CSV mode
    if input_path.exists():
        data = _load_csv(input_path)
        result = compute_emms(data)
        _save_csv(result, output_path)
        signal = latest_signal(result)
        anomaly = latest_anomaly(result)
        print(
            f"[INFO] CSV {symbol} {timeframe}:  "
            f"signal={signal or '—'},  "
            f"anomaly={anomaly},  "
            f"bars={len(data['close'])}"
        )
    else:
        print(f"[ERROR] File not found: {input_path}")


if __name__ == "__main__":
    main()
