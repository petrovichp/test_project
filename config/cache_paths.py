"""
Central cache-path constants and helpers.

All cache I/O should go through this module so that any future reorganization
touches one file rather than 60+ scripts. Import the subfolder constants and
join with filenames, e.g.

    from config.cache_paths import STATE, POLICIES, PREDS
    sp = np.load(STATE / f"{ticker}_dqn_state_train_v8_s11s13.npz")
    net.load_state_dict(torch.load(POLICIES / f"{ticker}_dqn_policy_{tag}.pt"))

Helper functions are provided for the most common patterns
(see `policy_path`, `state_path`, `train_history_path`, ...).
"""
from pathlib import Path

# ── roots ────────────────────────────────────────────────────────────────────
ROOT  = Path(__file__).parent.parent
CACHE = ROOT / "cache"

# ── subfolders by data type ──────────────────────────────────────────────────
RAW       = CACHE / "raw"        # okx_*usdt_spotpepr_*.parquet
FEATURES  = CACHE / "features"   # *_features_*.parquet
PREDS     = CACHE / "preds"      # vol, direction, regime, pnl predictors
STATE     = CACHE / "state"      # DQN state arrays + scaler stats
POLICIES  = CACHE / "policies"   # trained .pt + train_history.json
DISTILL   = CACHE / "distill"    # C2 teacher labels
RESULTS   = CACHE / "results"    # eval / audit / walkforward result JSONs
PLOTS     = CACHE / "plots"      # generated PNG plots + summary JSONs
LOOKUP    = CACHE / "lookup"     # thresholds, calibration, audit trades

ALL_SUBFOLDERS = (RAW, FEATURES, PREDS, STATE, POLICIES, DISTILL,
                   RESULTS, PLOTS, LOOKUP)


# ── canonical filename helpers ───────────────────────────────────────────────

def policy_path(tag: str, seed: int | None = None, ticker: str = "btc") -> Path:
    """cache/policies/{ticker}_dqn_policy_{tag}_seed{seed}.pt (omit _seed if None)"""
    suffix = f"_seed{seed}" if seed is not None else ""
    return POLICIES / f"{ticker}_dqn_policy_{tag}{suffix}.pt"


def train_history_path(tag: str, seed: int | None = None, ticker: str = "btc") -> Path:
    """cache/policies/{ticker}_dqn_train_history_{tag}_seed{seed}.json"""
    suffix = f"_seed{seed}" if seed is not None else ""
    return POLICIES / f"{ticker}_dqn_train_history_{tag}{suffix}.json"


def state_path(split: str, version: str = "v5", ticker: str = "btc") -> Path:
    """cache/state/{ticker}_dqn_state_{split}[_{version}].npz

    version ∈ {'v5', 'v6', 'v7_pa', 'v7_basis', 'v8_s11s13', 'v9_basis_s11s13'}.
    v5 is the historical default and omits the version suffix.
    """
    suffix = "" if version == "v5" else f"_{version}"
    return STATE / f"{ticker}_dqn_state_{split}{suffix}.npz"


def distill_targets_path(split: str, version: str = "v8",
                          ticker: str = "btc") -> Path:
    """cache/distill/{ticker}_distill_targets_{split}_{version}.npz"""
    return DISTILL / f"{ticker}_distill_targets_{split}_{version}.npz"


def pred_vol_path(ticker: str = "btc", v4: bool = True) -> Path:
    """cache/preds/{ticker}_pred_vol[_v4].npz"""
    suffix = "_v4" if v4 else ""
    return PREDS / f"{ticker}_pred_vol{suffix}.npz"


def pred_dir_path(direction: str, horizon: int, ticker: str = "btc",
                   v4: bool = True) -> Path:
    """cache/preds/{ticker}_pred_dir_{up|down}_{60|100}[_v4].npz"""
    suffix = "_v4" if v4 else ""
    return PREDS / f"{ticker}_pred_dir_{direction}_{horizon}{suffix}.npz"


def regime_path(ticker: str = "btc", v4: bool = True) -> Path:
    """cache/preds/{ticker}_regime_cusum[_v4].parquet"""
    suffix = "_v4" if v4 else ""
    return PREDS / f"{ticker}_regime_cusum{suffix}.parquet"


def features_assembled_path(ticker: str = "btc") -> Path:
    """cache/features/{ticker}_features_assembled.parquet"""
    return FEATURES / f"{ticker}_features_assembled.parquet"


def raw_meta_path(ticker: str, date_stamp: str = "20260425") -> Path:
    """cache/raw/okx_{ticker}usdt_spotpepr_{stamp}_meta.parquet"""
    return RAW / f"okx_{ticker}usdt_spotpepr_{date_stamp}_meta.parquet"


def raw_ob_path(ticker: str, date_stamp: str = "20260425") -> Path:
    """cache/raw/okx_{ticker}usdt_spotpepr_{stamp}_ob.parquet"""
    return RAW / f"okx_{ticker}usdt_spotpepr_{date_stamp}_ob.parquet"


def ensure_subfolders() -> None:
    """Create all subfolders if missing. Safe to call at startup."""
    for d in ALL_SUBFOLDERS:
        d.mkdir(parents=True, exist_ok=True)
