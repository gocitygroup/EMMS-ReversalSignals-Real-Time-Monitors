"""
Reversal Signals — TD Sequential-Style Real-Time MT5 Monitor.

Detects momentum-phase and exhaustion-phase reversal setups across
9 major forex pairs and 7 timeframes (1m → D1).  Computes qualified
trade setups with price-flip confirmation and displays a refreshable
signal matrix in the terminal every 60 seconds.

Usage:
    python ReversalSignals.py --realtime
    python ReversalSignals.py --realtime --socketio
    python ReversalSignals.py --realtime --setup Momentum --interval 30
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_PAIRS: Tuple[str, ...] = (
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
    "USDCAD", "NZDUSD", "EURGBP", "EURJPY",
)

DEFAULT_TIMEFRAMES: Tuple[str, ...] = (
    "M1", "M5", "M15", "M30", "H1", "H4", "D1",
)

TF_LABELS: Dict[str, str] = {
    "M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m",
    "H1": "1H", "H4": "4H", "D1": "1D",
}

SETUP_OPTIONS: Tuple[str, ...] = ("Momentum", "Exhaustion", "Qualified")

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ReversalConfig:
    """Tuning parameters for the reversal-signal engine."""

    momentum_max: int = 9
    exhaustion_max: int = 13
    lookback_qualified: int = 50      # bars to scan back for prior completions
    price_flip_lookback: int = 4      # close[i] vs close[i-4] for flip detection
    min_bars: int = 13                # minimum bars required to compute


@dataclass
class MonitorConfig:
    """Top-level configuration for the real-time monitor."""

    pairs: Tuple[str, ...] = DEFAULT_PAIRS
    timeframes: Tuple[str, ...] = DEFAULT_TIMEFRAMES
    bars: int = 1000
    update_interval: float = 60.0
    engine: ReversalConfig = field(default_factory=ReversalConfig)
    trade_setup: str = "Qualified"    # "Momentum" | "Exhaustion" | "Qualified"
    fetch_workers: int = 4
    enable_socketio: bool = False
    socketio_host: str = "localhost"
    socketio_port: int = 5000


# ═══════════════════════════════════════════════════════════════════════════════
# Reversal Engine  — NumPy-based TD Sequential algorithm
# ═══════════════════════════════════════════════════════════════════════════════


class ReversalEngine:
    """
    Stateless calculator for TD Sequential reversal signals.

    Computes three layers from OHLCV arrays:

    1. **Momentum phase** — 9‑count setup (Tom DeMark «Setup»)
    2. **Exhaustion phase** — 13‑count countdown («Countdown»)
    3. **Trade setups** — price‑flip‑confirmed entry signals

    All inputs are raw NumPy float arrays of equal length.
    """

    def __init__(self, config: ReversalConfig | None = None) -> None:
        self.cfg = config or ReversalConfig()

    # ── public entry point ──────────────────────────────────────────────────

    def compute(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> Dict[str, np.ndarray]:
        """Run the full pipeline and return all result arrays."""
        n = len(c)
        if n < self.cfg.min_bars:
            raise ValueError(f"Need at least {self.cfg.min_bars} bars, got {n}")

        momentum = self._momentum_phase(o, h, l, c)
        exhaustion = self._exhaustion_phase(o, h, l, c, momentum)
        trades = self._trade_setups(c, h, l, momentum, exhaustion,
                                    setup_mode="Qualified")

        return {**momentum, **exhaustion, **trades}

    # ── layer 1: momentum phase (TD Setup 9-count) ─────────────────────────

    def _momentum_phase(
        self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray
    ) -> Dict[str, np.ndarray]:
        n = len(c)
        bSC = np.zeros(n, dtype=np.int32)
        sSC = np.zeros(n, dtype=np.int32)
        bSR = np.zeros(n, dtype=float)
        sSS = np.zeros(n, dtype=float)
        bSD = np.zeros(n, dtype=float)
        sSD = np.zeros(n, dtype=float)
        bSH = np.zeros(n, dtype=float)
        bSL = np.zeros(n, dtype=float)
        sSH = np.zeros(n, dtype=float)
        sSL = np.zeros(n, dtype=float)

        _9 = self.cfg.momentum_max  # 9

        for i in range(4, n):
            con = c[i] < c[i - 4]

            if con:
                bSC[i] = 1 if bSC[i - 1] == _9 else bSC[i - 1] + 1
                sSC[i] = 0
            else:
                sSC[i] = 1 if sSC[i - 1] == _9 else sSC[i - 1] + 1
                bSC[i] = 0

            # ── bullish momentum tracking ───────────────────────────────
            if bSC[i] == 1:
                bSL[i] = l[i]

            if bSC[i] > 0:
                bSL[i] = min(bSL[i - 1], l[i]) if bSL[i - 1] > 0 else l[i]
                if l[i] == bSL[i]:
                    bSH[i] = h[i]

                if bSC[i] == _9:
                    bSD[i] = 2.0 * bSL[i] - bSH[i]
                else:
                    bSD[i] = (
                        0.0
                        if c[i] < bSD[i - 1] or sSC[i] == _9
                        else bSD[i - 1]
                    )

                bC8 = bSC[i - 1] == 8 and sSC[i] == 1
                if bSC[i] == _9 or bC8:
                    start = max(0, i - 8)
                    bSR[i] = float(np.max(h[start : i + 1]))
                else:
                    bSR[i] = bSR[i - 1]
            else:
                bSR[i] = 0.0 if c[i] > bSR[i - 1] else bSR[i - 1]

            # ── bearish momentum tracking ───────────────────────────────
            if sSC[i] == 1:
                sSH[i] = h[i]

            if sSC[i] > 0:
                sSH[i] = max(sSH[i - 1], h[i]) if sSH[i - 1] > 0 else h[i]
                if h[i] == sSH[i]:
                    sSL[i] = l[i]

                if sSC[i] == _9:
                    sSD[i] = 2.0 * sSH[i] - sSL[i]
                else:
                    sSD[i] = (
                        0.0
                        if c[i] > sSD[i - 1] or bSC[i] == _9
                        else sSD[i - 1]
                    )

                sC8 = sSC[i - 1] == 8 and bSC[i] == 1
                if sSC[i] == _9 or sC8:
                    start = max(0, i - 8)
                    sSS[i] = float(np.min(l[start : i + 1]))
                else:
                    sSS[i] = sSS[i - 1]
            else:
                sSS[i] = 0.0 if c[i] < sSS[i - 1] else sSS[i - 1]

        return {
            "bSC": bSC, "sSC": sSC,
            "bSR": bSR, "sSS": sSS,
            "bSD": bSD, "sSD": sSD,
            "bSH": bSH, "bSL": bSL,
            "sSH": sSH, "sSL": sSL,
        }

    # ── layer 2: exhaustion phase (TD Countdown 13-count) ──────────────────

    def _exhaustion_phase(
        self,
        o: np.ndarray,
        h: np.ndarray,
        l: np.ndarray,
        c: np.ndarray,
        momentum: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        n = len(c)
        _9 = self.cfg.momentum_max
        _13 = self.cfg.exhaustion_max

        bSC, sSC = momentum["bSC"], momentum["sSC"]
        bSR, sSS = momentum["bSR"], momentum["sSS"]

        bCC = np.zeros(n, dtype=np.int32)
        sCC = np.zeros(n, dtype=np.int32)
        bCL = np.zeros(n, dtype=float)
        bCH = np.zeros(n, dtype=float)
        sCL = np.zeros(n, dtype=float)
        sCH = np.zeros(n, dtype=float)
        bCD = np.zeros(n, dtype=float)
        bCT = np.zeros(n, dtype=float)
        sCD = np.zeros(n, dtype=float)
        sCT = np.zeros(n, dtype=float)
        bC8 = np.zeros(n, dtype=float)
        sC8 = np.zeros(n, dtype=float)

        # pre-compute perfect conditions
        pbS = np.zeros(n, dtype=bool)
        psS = np.zeros(n, dtype=bool)
        for i in range(3, n):
            pbS[i] = (l[i] <= l[i - 3] and l[i] <= l[i - 2]) or \
                     (l[i - 1] <= l[i - 3] and l[i - 1] <= l[i - 2])
            psS[i] = (h[i] >= h[i - 3] and h[i] >= h[i - 2]) or \
                     (h[i - 1] >= h[i - 3] and h[i - 1] >= h[i - 2])

        # tracking state — FIXED: maintained *outside* the loop
        bCLt = bCHt = sCLt = sCHt = 0.0
        sbC = False
        ssC = False

        for i in range(2, n):
            bCC_cond = c[i] <= l[i - 2]
            sCC_cond = c[i] >= h[i - 2]

            # ── update start conditions (stateful) ──────────────────────
            if bSC[i] == _9 and bCC[i - 1] == 0 and (pbS[i] or pbS[i - 1]):
                sbC = True
            elif sSC[i] == _9 or bCC[i - 1] == _13 or c[i] > bSR[i]:
                sbC = False

            if sSC[i] == _9 and sCC[i - 1] == 0 and (psS[i] or psS[i - 1]):
                ssC = True
            elif bSC[i] == _9 or sCC[i - 1] == _13 or c[i] < sSS[i]:
                ssC = False

            # ── bullish exhaustion ──────────────────────────────────────
            if sbC:
                if bCC_cond:
                    bCC[i] = 1 if bCC[i - 1] == 0 else bCC[i - 1] + 1
                else:
                    bCC[i] = 0

                b13 = bCC_cond and l[i] >= bC8[i - 1]
                if bCC[i] == _13 and b13:
                    bCC[i] = 12
            else:
                bCC[i] = 0

            if bCC[i] == 8 and bCC[i] != bCC[i - 1]:
                bC8[i] = c[i]
            else:
                bC8[i] = bC8[i - 1] if i > 0 else 0.0

            if bCC[i] == 1:
                bCLt = l[i]
                bCHt = h[i]
            elif sbC:
                bCHt = max(bCHt, h[i])
                bCLt = min(bCLt, l[i])
                if h[i] == bCHt:
                    bCH[i] = h[i]
                if l[i] == bCLt:
                    bCL[i] = l[i]

            if bCC[i] == _13:
                bCT[i] = 2.0 * bCHt - bCL[i]
                bCD[i] = 2.0 * bCLt - bCH[i]
            else:
                bCT[i] = (
                    0.0
                    if c[i] > bCT[i - 1] or (bCD[i - 1] == 0.0 and sCC[i] == _13)
                    else bCT[i - 1]
                )
                bCD[i] = (
                    0.0
                    if c[i] < bCD[i - 1] or (bCT[i - 1] == 0.0 and sCC[i] == _13)
                    else bCD[i - 1]
                )

            # ── bearish exhaustion ──────────────────────────────────────
            if ssC:
                if sCC_cond:
                    sCC[i] = 1 if sCC[i - 1] == 0 else sCC[i - 1] + 1
                else:
                    sCC[i] = 0

                s13 = sCC_cond and h[i] <= sC8[i - 1]
                if sCC[i] == _13 and s13:
                    sCC[i] = 12
            else:
                sCC[i] = 0

            if sCC[i] == 8 and sCC[i] != sCC[i - 1]:
                sC8[i] = c[i]
            else:
                sC8[i] = sC8[i - 1] if i > 0 else 0.0

            if sCC[i] == 1:
                sCLt = l[i]
                sCHt = h[i]
            elif ssC:
                sCHt = max(sCHt, h[i])
                sCLt = min(sCLt, l[i])
                if h[i] == sCHt:
                    sCH[i] = h[i]
                if l[i] == sCLt:
                    sCL[i] = l[i]

            if sCC[i] == _13:
                sCD[i] = 2.0 * sCHt - sCL[i]
                sCT[i] = 2.0 * sCLt - sCH[i]
            else:
                sCD[i] = (
                    0.0
                    if c[i] > sCD[i - 1] or (sCT[i - 1] == 0.0 and bCC[i] == _13)
                    else sCD[i - 1]
                )
                sCT[i] = (
                    0.0
                    if c[i] < sCT[i - 1] or (sCD[i - 1] == 0.0 and bCC[i] == _13)
                    else sCT[i - 1]
                )

        return {
            "bCC": bCC, "sCC": sCC,
            "bCL": bCL, "bCH": bCH,
            "sCL": sCL, "sCH": sCH,
            "bCD": bCD, "bCT": bCT,
            "sCD": sCD, "sCT": sCT,
            "bC8": bC8, "sC8": sC8,
        }

    # ── layer 3: trade setups ──────────────────────────────────────────────

    def _trade_setups(
        self,
        c: np.ndarray,
        h: np.ndarray,
        l: np.ndarray,
        momentum: Dict[str, np.ndarray],
        exhaustion: Dict[str, np.ndarray],
        *,
        setup_mode: str = "Qualified",
    ) -> Dict[str, np.ndarray]:
        n = len(c)
        _9 = self.cfg.momentum_max
        _13 = self.cfg.exhaustion_max
        pfl = self.cfg.price_flip_lookback           # 4
        qlb = self.cfg.lookback_qualified            # 50

        bSC, sSC = momentum["bSC"], momentum["sSC"]
        bSD, sSD = momentum["bSD"], momentum["sSD"]
        bSR, sSS = momentum["bSR"], momentum["sSS"]
        bCC, sCC = exhaustion["bCC"], exhaustion["sCC"]
        bCT, sCT = exhaustion["bCT"], exhaustion["sCT"]
        bCD, sCD = exhaustion["bCD"], exhaustion["sCD"]

        signal = np.full(n, "", dtype=object)
        setup = np.full(n, "", dtype=object)
        target = np.zeros(n, dtype=float)
        risk = np.zeros(n, dtype=float)
        confidence = np.zeros(n, dtype=float)

        for i in range(pfl, n):
            bull_flip = c[i] > c[i - pfl] and c[i - 1] < c[i - pfl - 1]
            bear_flip = c[i] < c[i - pfl] and c[i - 1] > c[i - pfl - 1]

            bC8_idx = bSC[i - 1] == 8 and sSC[i] == 1
            sC8_idx = sSC[i - 1] == 8 and bSC[i] == 1

            # ── qualified setup pre-check ───────────────────────────────
            bQC = False
            sQC = False
            if setup_mode == "Qualified" and i > _13:
                bBl9 = bBp9 = bB13 = None
                sBl9 = sBp9 = sB13 = None

                for j in range(i - 1, max(0, i - qlb), -1):
                    if bSC[j] == _9:
                        if bBl9 is None:
                            bBl9 = j
                        elif bBp9 is None:
                            bBp9 = j
                            break
                for j in range(i - 1, max(0, i - qlb), -1):
                    if bCC[j] == _13 and bB13 is None:
                        bB13 = j
                        break
                for j in range(i - 1, max(0, i - qlb), -1):
                    if sSC[j] == _9:
                        if sBl9 is None:
                            sBl9 = j
                        elif sBp9 is None:
                            sBp9 = j
                            break
                for j in range(i - 1, max(0, i - qlb), -1):
                    if sCC[j] == _13 and sB13 is None:
                        sB13 = j
                        break

                if bBl9 is not None and bB13 is not None and bBp9 is not None:
                    bQC = (bBl9 > bB13) and (bB13 > bBp9)
                if sBl9 is not None and sB13 is not None and sBp9 is not None:
                    sQC = (sBl9 > sB13) and (sB13 > sBp9)

            # ── Momentum setups ─────────────────────────────────────────
            if setup_mode in ("Momentum", "Qualified"):
                if (bSC[i] == _9 or bC8_idx) and bull_flip:
                    signal[i] = "BUY"
                    setup[i] = "MOMENTUM"
                    target[i] = bSR[i] if bSR[i] > 0 else h[i] * 1.01
                    risk[i] = bSD[i] if bSD[i] > 0 else l[i]
                    confidence[i] = 0.7 if bSC[i] == _9 else 0.5

                if (sSC[i] == _9 or sC8_idx) and bear_flip:
                    signal[i] = "SELL"
                    setup[i] = "MOMENTUM"
                    target[i] = sSS[i] if sSS[i] > 0 else l[i] * 0.99
                    risk[i] = sSD[i] if sSD[i] > 0 else h[i]
                    confidence[i] = 0.7 if sSC[i] == _9 else 0.5

            # ── Exhaustion setups ───────────────────────────────────────
            if setup_mode in ("Exhaustion", "Qualified"):
                if bCC[i] == _13 and bull_flip:
                    signal[i] = "BUY"
                    setup[i] = "EXHAUSTION"
                    target[i] = bCT[i] if bCT[i] > 0 else h[i] * 1.02
                    risk[i] = bCD[i] if bCD[i] > 0 else l[i]
                    confidence[i] = 0.8

                if sCC[i] == _13 and bear_flip:
                    signal[i] = "SELL"
                    setup[i] = "EXHAUSTION"
                    target[i] = sCT[i] if sCT[i] > 0 else l[i] * 0.98
                    risk[i] = sCD[i] if sCD[i] > 0 else h[i]
                    confidence[i] = 0.8

            # ── Qualified setups (override with higher confidence) ──────
            if setup_mode == "Qualified":
                if bSC[i] == _9 and bQC and bull_flip:
                    signal[i] = "BUY"
                    setup[i] = "QUALIFIED"
                    target[i] = bCT[i] if bCT[i] > 0 else (bSR[i] if bSR[i] > 0 else h[i] * 1.02)
                    risk[i] = bCD[i] if bCD[i] > 0 else (bSD[i] if bSD[i] > 0 else l[i])
                    confidence[i] = 0.9

                if sSC[i] == _9 and sQC and bear_flip:
                    signal[i] = "SELL"
                    setup[i] = "QUALIFIED"
                    target[i] = sCT[i] if sCT[i] > 0 else (sSS[i] if sSS[i] > 0 else l[i] * 0.98)
                    risk[i] = sCD[i] if sCD[i] > 0 else (sSD[i] if sSD[i] > 0 else h[i])
                    confidence[i] = 0.9

        return {
            "signal": signal,
            "setup_type": setup,
            "target_level": target,
            "risk_level": risk,
            "confidence": confidence,
        }


# ── snapshot extraction helpers ──────────────────────────────────────────────


def latest_reversal_signal(result: Dict[str, np.ndarray]) -> Optional[str]:
    """Return ``"BUY"``, ``"SELL"``, or ``None`` from last bar."""
    sigs = result["signal"]
    last = sigs[-1]
    return str(last) if last else None


def latest_setup_type(result: Dict[str, np.ndarray]) -> Optional[str]:
    last = result["setup_type"][-1]
    return str(last) if last else None


def latest_phase_info(result: Dict[str, np.ndarray]) -> Dict[str, int]:
    """Return the latest bar's momentum and exhaustion counts."""
    return {
        "momentum_bull": int(result["bSC"][-1]),
        "momentum_bear": int(result["sSC"][-1]),
        "exhaustion_bull": int(result["bCC"][-1]),
        "exhaustion_bear": int(result["sCC"][-1]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MT5 Connection Manager  (identical pattern to EMMS refactor)
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

    def connect(self) -> None:
        """Establish connection.  Must be called once before any fetch."""
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

    def fetch(
        self, symbol: str, timeframe: str, bars: int
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Fetch the *bars* most recent OHLCV candles for one (symbol, timeframe).

        Returns ``None`` when the symbol or timeframe is unavailable.
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
# Signal matrix  (grid container)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PairSnapshot:
    """One (symbol, timeframe) after a fetch + compute cycle."""

    symbol: str
    timeframe: str
    signal: Optional[str]       # "BUY", "SELL", or None
    setup_type: Optional[str]   # "MOMENTUM", "EXHAUSTION", "QUALIFIED", or None
    phase: Dict[str, int] = field(default_factory=dict)
    close: Optional[float] = None
    timestamp: Optional[int] = None


class SignalMatrix:
    """Grid: rows = pairs, cols = timeframes.  Provides efficient lookup."""

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
        rows: List[List[PairSnapshot]] = []
        for pair in self.pairs:
            row: List[PairSnapshot] = []
            for tf in self.timeframes:
                snap = self._grid.get((pair, tf))
                row.append(
                    snap
                    if snap is not None
                    else PairSnapshot(pair, tf, None, None, {})
                )
            rows.append(row)
        return rows

    @property
    def total_signals(self) -> int:
        return sum(1 for v in self._grid.values() if v.signal is not None)

    @property
    def total_phases(self) -> int:
        """Cells with an active phase (momentum or exhaustion count > 0)."""
        count = 0
        for v in self._grid.values():
            p = v.phase
            if p.get("momentum_bull", 0) > 0 or p.get("momentum_bear", 0) > 0:
                count += 1
            elif p.get("exhaustion_bull", 0) > 0 or p.get("exhaustion_bear", 0) > 0:
                count += 1
        return count

    @property
    def populated(self) -> int:
        return len(self._grid)


# ═══════════════════════════════════════════════════════════════════════════════
# Terminal table renderer
# ═══════════════════════════════════════════════════════════════════════════════


class TableRenderer:
    """
    Renders the reversal-signal matrix as a Unicode box‑drawing table.

    Cell legend::

        B  — BUY trade setup (any type) on latest candle
        S  — SELL trade setup (any type) on latest candle
        M  — momentum phase active (count ≥ 1, no trade yet)
        E  — exhaustion phase active (count ≥ 1, no trade yet)
        ·  — neutral
    """

    PAIR_COL_WIDTH = 10
    CELL_WIDTH = 6

    def __init__(
        self, pairs: Tuple[str, ...], timeframes: Tuple[str, ...]
    ) -> None:
        self.pairs = pairs
        self.timeframes = timeframes
        self.tf_labels = [TF_LABELS.get(tf, tf) for tf in timeframes]

    def render(self, matrix: SignalMatrix) -> str:
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
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        table = self.render(matrix)
        status = (
            f"Update #{cycle} | {now} | "
            f"Fetched {matrix.populated} cells in {elapsed:.1f}s | "
            f"Signals: {matrix.total_signals} | "
            f"Active phases: {matrix.total_phases}"
        )
        return f"{table}\n{status}"

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
        # trade signal trumps everything
        if cell.signal == "BUY":
            return "B"
        if cell.signal == "SELL":
            return "S"

        p = cell.phase
        mb, ms = p.get("momentum_bull", 0), p.get("momentum_bear", 0)
        eb, es = p.get("exhaustion_bull", 0), p.get("exhaustion_bear", 0)

        if mb > 0 or ms > 0:
            return "M"
        if eb > 0 or es > 0:
            return "E"
        return "·"

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
# Reversal Monitor  (main orchestrator)
# ═══════════════════════════════════════════════════════════════════════════════


class ReversalMonitor:
    """
    Real-time pipeline:

        MT5 → [ThreadPool] → ReversalEngine → SignalMatrix → terminal table

    Fetches all (pair × timeframe) combos concurrently via a
    :class:`ThreadPoolExecutor`, computes phases & trade setups,
    and refreshes the terminal table on a fixed interval.
    """

    def __init__(
        self,
        config: MonitorConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.cfg = config or MonitorConfig()
        self.logger = logger or logging.getLogger("reversal")
        self.engine = ReversalEngine(self.cfg.engine)
        self.connection = MT5Connection()
        self.matrix = SignalMatrix(self.cfg.pairs, self.cfg.timeframes)
        self.renderer = TableRenderer(self.cfg.pairs, self.cfg.timeframes)
        self._executor: Optional[ThreadPoolExecutor] = None
        self._cycle = 0
        self._running = False

        # optional Socket.IO
        self._sio: Any = None
        self._sio_app: Any = None

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def run(self) -> None:
        self._running = True
        self._executor = ThreadPoolExecutor(max_workers=self.cfg.fetch_workers)
        self.connection.connect()

        total = len(self.cfg.pairs) * len(self.cfg.timeframes)
        self.logger.info(
            "MT5 connected — %d pairs × %d timeframes = %d series, "
            "%d fetch workers, setup mode=%s",
            len(self.cfg.pairs),
            len(self.cfg.timeframes),
            total,
            self.cfg.fetch_workers,
            self.cfg.trade_setup,
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
        loop = asyncio.get_running_loop()
        combos: List[Tuple[str, str]] = [
            (p, tf) for p in self.cfg.pairs for tf in self.cfg.timeframes
        ]

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
        data = self.connection.fetch(symbol, timeframe, self.cfg.bars)
        if data is None:
            return PairSnapshot(symbol, timeframe, None, None, {})

        try:
            result = self.engine.compute(
                data["open"], data["high"], data["low"], data["close"]
            )
        except ValueError:
            return PairSnapshot(symbol, timeframe, None, None, {})

        signal = latest_reversal_signal(result)
        setup = latest_setup_type(result)
        phase = latest_phase_info(result)
        close = float(data["close"][-1])
        ts_arr = data.get("timestamp")
        ts = int(ts_arr[-1]) if ts_arr is not None and len(ts_arr) > 0 else None

        return PairSnapshot(
            symbol=symbol,
            timeframe=timeframe,
            signal=signal,
            setup_type=setup,
            phase=phase,
            close=close,
            timestamp=ts,
        )

    # ── display ────────────────────────────────────────────────────────────

    _UTF8_OK = False

    @classmethod
    def _ensure_utf8(cls) -> None:
        if cls._UTF8_OK:
            return
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
        cls._UTF8_OK = True

    def _display(self, elapsed: float) -> None:
        self._ensure_utf8()
        output = self.renderer.render_summary(
            self.matrix, elapsed, self._cycle
        )
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.write(output + "\n")
        sys.stdout.flush()

    # ── optional Socket.IO ─────────────────────────────────────────────────

    def setup_socketio(self) -> None:
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
            await sio.emit(
                "status", {"message": "Connected to Reversal Signals"}, room=sid
            )

        @sio.event
        async def get_signals(sid, data):
            rows = self.matrix.to_rows()
            flat = [
                {
                    "symbol": c.symbol,
                    "timeframe": c.timeframe,
                    "signal": c.signal,
                    "setup_type": c.setup_type,
                }
                for row in rows
                for c in row
            ]
            await sio.emit("signals", flat, room=sid)

        @sio.event
        async def get_matrix(sid, data):
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
        if self._sio_app is None:
            self.logger.warning(
                "Socket.IO not configured.  Call setup_socketio() first."
            )
            return
        try:
            import uvicorn  # type: ignore
        except ImportError:
            self.logger.warning("uvicorn not installed — cannot serve Socket.IO")
            return

        config = uvicorn.Config(
            self._sio_app,
            host=self.cfg.socketio_host,
            port=self.cfg.socketio_port,
            log_level="info",
        )
        await uvicorn.Server(config).serve()


# ═══════════════════════════════════════════════════════════════════════════════
# CSV helpers  (legacy one‑shot mode)
# ═══════════════════════════════════════════════════════════════════════════════


def _save_csv(result: Dict[str, np.ndarray], path: Path) -> None:
    sigs, setups = result["signal"], result["setup_type"]
    mask = sigs != ""
    long_mask = (sigs == "BUY") & mask
    short_mask = (sigs == "SELL") & mask

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bar", "signal", "setup_type", "confidence",
                     "target", "risk", "momentum_bull", "momentum_bear",
                     "exhaustion_bull", "exhaustion_bear"])
        for i in range(len(sigs)):
            if not sigs[i]:
                continue
            w.writerow([
                i,
                str(sigs[i]),
                str(setups[i]),
                float(result["confidence"][i]),
                float(result["target_level"][i]),
                float(result["risk_level"][i]),
                int(result["bSC"][i]),
                int(result["sSC"][i]),
                int(result["bCC"][i]),
                int(result["sCC"][i]),
            ])
    print(f"Saved {int(np.count_nonzero(mask))} trade signals → {path.resolve()}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reversal Signals — live TD Sequential monitor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples
            --------
              %(prog)s --realtime
              %(prog)s --realtime --setup Momentum --interval 30
              %(prog)s --realtime --socketio --socket-port 8080
              %(prog)s --realtime --pairs EURUSD,GBPUSD,USDJPY \\
                  --timeframes M5,M15,H1
              %(prog)s --mt5 --symbol EURUSD --timeframe H1   # one-shot CSV
        """),
    )

    p.add_argument(
        "--realtime",
        action="store_true",
        help="Run the live terminal-table monitor.",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="Update interval in seconds (default: 60).",
    )

    p.add_argument("--pairs", type=str, default=None,
                   help="Comma-separated pairs (default: 9 majors).")
    p.add_argument("--timeframes", type=str, default=None,
                   help="Comma-separated timeframes (default: M1,M5,M15,M30,H1,H4,D1).")
    p.add_argument("--symbol", type=str, default="EURUSD")
    p.add_argument("--timeframe", type=str, default="H1")

    p.add_argument(
        "--setup",
        type=str,
        default="Qualified",
        choices=("Momentum", "Exhaustion", "Qualified"),
        help="Trade setup filter (default: Qualified).",
    )

    p.add_argument("--mt5", action="store_true")
    p.add_argument("--bars", type=int, default=1000)
    p.add_argument("--input", type=str, default="data.csv")
    p.add_argument("--output", type=str, default="reversal_signals.csv")

    p.add_argument("--socketio", action="store_true")
    p.add_argument("--socket-host", type=str, default="localhost")
    p.add_argument("--socket-port", type=int, default=5000)

    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--debug", action="store_true")

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
    logger = logging.getLogger("reversal")

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
            trade_setup=args.setup,
            fetch_workers=args.workers,
            enable_socketio=args.socketio,
            socketio_host=args.socket_host,
            socketio_port=args.socket_port,
        )

        monitor = ReversalMonitor(config, logger)

        async def _run() -> None:
            if config.enable_socketio:
                monitor.setup_socketio()
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
    # One-shot / CSV mode
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
            print(f"[ERROR] No data for {symbol} {timeframe}.")
            return

        engine = ReversalEngine()
        result = engine.compute(
            data["open"], data["high"], data["low"], data["close"]
        )
        _save_csv(result, output_path)

        sig = latest_reversal_signal(result)
        setup = latest_setup_type(result)
        phase = latest_phase_info(result)
        print(
            f"{symbol} {timeframe}:  "
            f"signal={sig or '—'},  "
            f"setup={setup or '—'},  "
            f"close={float(data['close'][-1]):.5f}"
        )
        print(
            f"  momentum: B={phase['momentum_bull']} S={phase['momentum_bear']}  "
            f"|  exhaustion: B={phase['exhaustion_bull']} S={phase['exhaustion_bear']}"
        )
        return

    if input_path.exists():
        print(f"[INFO] CSV mode not fully supported — use --mt5 instead.")
    else:
        print(f"[ERROR] File not found: {input_path}")


if __name__ == "__main__":
    main()
