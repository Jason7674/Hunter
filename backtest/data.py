"""Load Databento futures data into a clean front-month 1-minute OHLCV frame.

Accepts any mix of:
  - Databento batch-download .zip files (DBN or CSV inside)
  - folders containing those files
  - .dbn / .dbn.zst / .csv / .csv.zst files

Raw outright data (e.g. ES.FUT parent symbology) holds every contract month and spread.
We drop spreads and keep, for each trading day (18:00 NY roll), the single contract with
the most volume. Continuous symbology (ES.c.0) already arrives as one series and passes through.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

NY = "America/New_York"
_DATA_SUFFIXES = (".dbn", ".dbn.zst", ".csv", ".csv.zst")


def _is_data_name(name: str) -> bool:
    n = name.lower()
    return n.endswith(_DATA_SUFFIXES) and "metadata" not in n and "condition" not in n


def _read_dbn_bytes(raw: bytes) -> pd.DataFrame:
    import databento as db

    store = db.DBNStore.from_bytes(io.BytesIO(raw))
    df = store.to_df(price_type="float", pretty_ts=True, map_symbols=True)
    return df.reset_index()


def _read_csv_bytes(raw: bytes, name: str) -> pd.DataFrame:
    if name.lower().endswith(".zst"):
        import zstandard

        raw = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(raw)).read()
    return pd.read_csv(io.BytesIO(raw))


def _read_one(raw: bytes, name: str) -> pd.DataFrame:
    if ".dbn" in name.lower():
        return _read_dbn_bytes(raw)
    return _read_csv_bytes(raw, name)


def _iter_sources(paths: list[str | Path]):
    for p in map(Path, paths):
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file() and (child.suffix.lower() == ".zip" or _is_data_name(child.name)):
                    yield from _iter_sources([child])
        elif p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as zf:
                for info in sorted(zf.infolist(), key=lambda i: i.filename):
                    if _is_data_name(info.filename):
                        yield info.filename, zf.read(info)
        elif _is_data_name(p.name):
            yield p.name, p.read_bytes()


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"ts_event": "ts", "ts_recv": "ts_recv"})
    if "ts" not in df.columns:
        raise ValueError(f"No ts_event column; got {list(df.columns)}")
    ts = df["ts"]
    if pd.api.types.is_integer_dtype(ts):
        ts = pd.to_datetime(ts, unit="ns", utc=True)
    else:
        ts = pd.to_datetime(ts, utc=True)
    df["ts"] = ts

    # Trades schema -> 1m bars
    if "open" not in df.columns and {"price", "size"} <= set(df.columns):
        key = "symbol" if "symbol" in df.columns else "instrument_id"
        df = (
            df.set_index("ts")
            .groupby(key)
            .resample("1min")
            .agg(open=("price", "first"), high=("price", "max"), low=("price", "min"),
                 close=("price", "last"), volume=("size", "sum"))
            .dropna(subset=["open"])
            .reset_index()
        )

    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    # CSV without pretty_px stores fixed-point prices (1e-9 units)
    if df["close"].median() > 1e6:
        for c in ("open", "high", "low", "close"):
            df[c] = df[c] / 1e9
    df["volume"] = df["volume"].astype(float)
    if "symbol" not in df.columns:
        df["symbol"] = df["instrument_id"].astype(str) if "instrument_id" in df.columns else "SERIES"
    return df[["ts", "symbol", "open", "high", "low", "close", "volume"]]


def front_month(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one outright contract per trading day: the one with the most volume."""
    df = df[~df["symbol"].astype(str).str.contains(r"[-:]", regex=True)].copy()
    local = df["ts"].dt.tz_convert(NY)
    # Trading day rolls at 18:00 NY
    df["tday"] = (local + pd.Timedelta(hours=6)).dt.date
    vol = df.groupby(["tday", "symbol"])["volume"].sum().reset_index()
    best = vol.loc[vol.groupby("tday")["volume"].idxmax(), ["tday", "symbol"]]
    df = df.merge(best, on=["tday", "symbol"])
    return df.drop(columns="tday")


def load(paths: list[str | Path], timeframe: str = "1min") -> pd.DataFrame:
    """Return a NY-time indexed OHLCV frame (index = bar open time) for the front month."""
    frames = [_normalize(_read_one(raw, name)) for name, raw in _iter_sources(paths)]
    if not frames:
        raise FileNotFoundError(f"No Databento data files found in {paths}")
    df = pd.concat(frames, ignore_index=True)
    df = front_month(df)
    df = df.sort_values("ts").drop_duplicates("ts", keep="last")
    df = df.set_index(df["ts"].dt.tz_convert(NY)).drop(columns="ts")
    df.index.name = "time"
    if timeframe != "1min":
        df = df.resample(timeframe, label="left", closed="left").agg(
            {"symbol": "last", "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna(subset=["open"])
    return df
