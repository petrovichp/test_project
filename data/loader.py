"""
Data loading with CSV → Parquet caching.

First call: reads CSV, splits into meta and OB parquets, saves to cache/.
Subsequent calls: loads parquet directly (~10x faster than CSV).

Cache is keyed by source filename — new CSV triggers automatic regeneration.
"""

import pandas as pd
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "cache"
DATA_DIR = Path("/Users/petrpogoraev/Documents/Projects/options_trading/DATA/last_source_data")

_DROP_COLS = {"Unnamed: 0", "datetime"}
_OB_PREFIXES = ("spot_bids_amount_", "spot_asks_amount_", "perp_bids_amount_", "perp_asks_amount_")


def _split_cols(all_cols: list[str]) -> tuple[list[str], list[str]]:
    ob = [c for c in all_cols if c.startswith(_OB_PREFIXES)]
    meta = [c for c in all_cols if c not in ob and c not in _DROP_COLS]
    return meta, ob


def _latest_csv(ticker: str) -> Path:
    files = sorted(DATA_DIR.glob(f"okx_{ticker}usdt_spotpepr_*.csv"), key=lambda f: f.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No CSV found for ticker '{ticker}' in {DATA_DIR}")
    return files[-1]


def _cache_paths(csv_path: Path) -> tuple[Path, Path]:
    stem = csv_path.stem
    return CACHE_DIR / f"{stem}_meta.parquet", CACHE_DIR / f"{stem}_ob.parquet"


def _build_cache(csv_path: Path, meta_path: Path, ob_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Building cache from {csv_path.name} ...")
    df = pd.read_csv(csv_path)

    meta_cols, ob_cols = _split_cols(df.columns.tolist())
    meta = df[meta_cols].copy()
    ob = df[ob_cols].copy()
    ob.insert(0, "timestamp", meta["timestamp"].values)

    CACHE_DIR.mkdir(exist_ok=True)
    meta.to_parquet(meta_path, index=False)
    ob.to_parquet(ob_path, index=False)

    print(f"  meta → {meta_path.name}  ({meta_path.stat().st_size // 1024:,} KB)")
    print(f"  ob   → {ob_path.name}  ({ob_path.stat().st_size // 1024:,} KB)")
    return meta, ob


def load(ticker: str, include_ob: bool = True) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Load data for a ticker. Returns (meta_df, ob_df).

    meta_df  — 30 market/microstructure columns + timestamp
    ob_df    — 800 orderbook amount columns + timestamp (None if include_ob=False)

    Cache is keyed by source filename; a new CSV auto-invalidates the cache.
    """
    csv_path = _latest_csv(ticker)
    meta_path, ob_path = _cache_paths(csv_path)

    if meta_path.exists() and ob_path.exists():
        print(f"Loading from cache: {csv_path.stem}")
        meta = pd.read_parquet(meta_path)
        ob = pd.read_parquet(ob_path) if include_ob else None
    else:
        meta, ob = _build_cache(csv_path, meta_path, ob_path)
        if not include_ob:
            ob = None

    return meta, ob


def load_meta(ticker: str) -> pd.DataFrame:
    """Load only the 30 metadata columns. Fast — skips the 800 OB columns."""
    meta, _ = load(ticker, include_ob=False)
    return meta
