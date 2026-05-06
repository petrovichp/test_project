# System Architecture

Last updated: 2026-05-05 (post-regime-cleanup, Phase 2 starting from scratch)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RAW DATA (Parquet cache)                          │
│  384k rows × 1-min bars  |  BTC spot+perp  |  Jul 2025 → Apr 2026     │
│                                                                          │
│  meta parquet:  29 cols — prices, OI, funding, spreads, taker flow     │
│  ob parquet:   800 cols — 200 price-bin amounts × 4 sides              │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      FEATURE ENGINEERING  (features/)                   │
│                                                                          │
│  orderbook.py  →  32 feat   OB imbalance, OFI, depth-bands, velocity   │
│  price.py      →  51 feat   returns, SMA/EMA, RSI, MACD, BB, VWAP      │
│  volume.py     →  17 feat   taker imbalance/net, OBV, vol z-score      │
│  market.py     →  30 feat   OI, funding rate, spread, calendar          │
│                                                                          │
│  assembly.py:  191 features  |  gap-masked (9.45% gaps removed)         │
│  StandardScaler fit on train only  |  cached as parquet                 │
│                                                                          │
│  Split ──────────────────────────────────────────────────────────────   │
│    Train  70,902 bars   Jul 2025 → Oct 2025   (50%)                    │
│    Val    35,451 bars   Oct 2025 → Dec 2025   (25%)   ← tune params    │
│    Test   35,451 bars   Dec 2025 → Apr 2026   (25%)   ← locked         │
└──────────┬───────────────────────────────────┬──────────────────────────┘
           │                                   │
           ▼                                   ▼
┌──────────────────┐                ┌────────────────────────┐
│  VOLATILITY      │                │  DIRECTION MODELS       │
│  MODEL           │                │  (two-stage pipeline)   │
│                  │                │                         │
│  LightGBM        │                │  Step 1: ATR-30 rank    │
│  btc_lgbm_atr_30 │                │  appended to X          │
│                  │                │                         │
│  Input: 191 feat │                │  Step 2: CNN-LSTM        │
│  Target: ATR-30  │                │  SEQ_LEN=30 bars        │
│                  │                │  34 seq features+ATR    │
│  Spearman=0.80   │                │                         │
│  (test, atr_30)  │                │  4 models cached:       │
│                  │                │  cnn2s_dir_up_60        │
│  Output:         │                │  cnn2s_dir_down_60      │
│  atr_pred ($)    │                │  cnn2s_dir_up_100       │
│  vol_pred (rank) │                │  cnn2s_dir_down_100     │
│                  │                │                         │
│  Used for:       │                │  AUC 0.75 (up_60)       │
│  • vol gate      │                │       0.71 (down_60)    │
│  • ATR-dynamic   │                │                         │
│    TP/SL scaling │                │  Output:                │
│  • atr_rank feat │                │  p_up_60, p_dn_60       │
│    for CNN-LSTM  │                │  p_up_100, p_dn_100     │
│                  │                │                         │
│  Predictions     │                │  Threshold ≥0.75 (up)   │
│  cached to .npz  │                │             ≥0.70 (dn)  │
│                  │                │  Cached to .npz         │
└──────┬───────────┘                └───────────┬────────────┘
       │                                        │
       └────────────────┬───────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     SIGNAL DATAFRAME  (per bar)                         │
│                                                                          │
│  price  atr_pred  vol_pred  p_up_60  p_dn_60  p_up_100  p_dn_100       │
│  bb_pct_b  bb_width  rsi_6  rsi_14  macd_hist  ofi_perp_10_r15         │
│  taker_net_15/30/60  fund_rate  fund_mom_480  vwap_dev_240/1440         │
│  oi_price_div_15  spot/perp_imbalance  vol_z_spot_60  diff_price        │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                SIGNAL LAYER  (strategy/agent.py)                         │
│  Output: direction signal (+1 long / -1 short / 0 flat) per bar         │
│                                                                          │
│  S1_VolDir     vol>0.60 & p_up_60>0.75 → LONG  (momentum)             │
│                vol>0.60 & p_dn_60>0.70 → SHORT                         │
│  S2_Funding    fund_z>+2σ → SHORT  (funding mean-reversion)            │
│  S3_BBRevert   bb_pct_b<0.05 & OFI+ → LONG  (BB mean-reversion)      │
│  S4_MACDTrend  sma stack + MACD↑ + p_up>0.70 → LONG  (momentum)      │
│  S5_OFIScalp   OFI z>2σ → LONG/SHORT  (structural failure, drop)      │
│  S6_TwoSignal  p_up>0.70 & MACD & OFI → LONG  (consensus)            │
│  S7_OIDiverg   oi_price_div_15 z>1.5σ → LONG/SHORT                   │
│  S8_TakerFlow  taker_net_60 z>1σ & OFI → LONG/SHORT                  │
│  S9_LargeOrd   spot_large_bid z>1.5σ → LONG/SHORT  (failure, drop)   │
│  S10_Squeeze   bb_width<0.65×mean & MACD cross → LONG/SHORT           │
│  S11_Basis     diff_price z>1.5σ & fund_mom > 0 → LONG/SHORT  (drop) │
│  S12_VWAPVol   vwap_dev_240<-0.8% & vol_z>1σ & taker turn → LONG    │
│  S13_OBDiv     spot_imb>0.10 & perp_imb<-0.10 → LONG/SHORT  (drop)   │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               EXECUTION LAYER  (execution/)                              │
│                                                                          │
│  ┌─ ENTRY  (entry.py) ───────────────────────────────────────────────┐  │
│  │  MarketEntry   → enter at next bar close                          │  │
│  │  ConfirmEntry  → k consecutive bars same direction                │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌─ EXIT  (exit.py) ─────────────────────────────────────────────────┐  │
│  │  ATR-dynamic TP/SL (ComboExit):                                   │  │
│  │    TP = base_tp × (atr_pred / atr_median)  clipped [0.2×, 5×]   │  │
│  │    SL = base_sl × (atr_pred / atr_median)                        │  │
│  │                                                                   │  │
│  │  Momentum  (S1,S4,S6,S8): tp=2–2.5%, sl=0.7–0.8%               │  │
│  │                            trail-after-breakeven ON               │  │
│  │  Mean-rev  (S2,S3,S10):   tp=0.8–1.5%, sl=0.4–0.5%             │  │
│  │                            time_stop=30–120 bars                  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌─ SIZING  (sizing.py) ─────────────────────────────────────────────┐  │
│  │  VolScaledSizer:  size = target_risk(1%) / sl_pct                │  │
│  │    SL=0.5% → 20% pos  |  SL=1.5% → 6.7% pos  (const $ risk)    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │  tp_arr, sl_arr, trail_arr, tab_arr,
                               │  be_arr, ts_arr, size_arr per bar
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   BACKTEST ENGINE  (backtest/engine.py)                  │
│                                                                          │
│  • 1-bar execution lag  (signal at T → fill at T+1 close)              │
│  • OKX taker fee 0.08% per side  (0.16% round-trip)                   │
│  • Max 1 position at a time                                             │
│                                                                          │
│  Exit priority per bar:                                                 │
│    1. force_exit   → FORCE   (external close signal — generic)          │
│    2. time_stop    → TIME    (max hold bars exceeded)                  │
│    3. breakeven    → BE      (SL moved to entry once in profit)        │
│    4. trail-after-BE → TSL  (trailing activates post-breakeven)       │
│    5. TP hit       → TP                                                │
│    6. SL hit       → SL                                                │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         RESULTS                                          │
│                                                                          │
│  Backtest results (no regime gate, free):                               │
│                                                                          │
│  Strategy       Val Sharpe  Val Tr  Val Win%   Test Sharpe  Test Tr     │
│  S1_VolDir       +7.02 ✓      98     46%         -0.81       140        │
│  S8_TakerFlow    +2.42 ✓      99     48%         -6.12       147        │
│  S4_MACDTrend    +0.19        56     39%         +0.70 ✓      93        │
│  All others      negative                         negative               │
│                                                                          │
│  Core problem: val/test gap. S1 +7 → -1, S8 +2.4 → -6.                │
│  Phase 2 needs to identify when to trust each strategy.                 │
└─────────────────────────────────────────────────────────────────────────┘
```

## Status

**Done:**
- Phase 1: features, vol model, direction model, execution layer, backtest engine
- Prediction caching (vol + direction) → backtest 9.5s

**Phase 2 — TODO (fresh start):**
- Regime/context awareness layer
- Approach to be redesigned (previous HMM + extended-models attempt scrapped)

## Performance (runtime)

| Operation | Time | Notes |
|---|---|---|
| Feature assembly | ~2 min | Cached after first run |
| Vol model training | ~30s | LightGBM, 191 features |
| Direction CNN-LSTM training | ~20 min | 4 models × 4 horizons |
| Backtest (13 strategies, 2 splits) | **9.5s** | Predictions cached in .npz |

## Cached files (cache/)

| File | Description |
|---|---|
| `btc_features_assembled.parquet` | 191 features, all splits |
| `btc_lgbm_atr_30.txt` | Volatility LightGBM model |
| `btc_cnn2s_dir_{dir}_{H}.keras` | CNN-LSTM direction models (4) |
| `btc_pred_vol.npz` | Vol + rank predictions, all splits (cached) |
| `btc_pred_dir_{col}.npz` | Direction predictions per model (cached, 4 files) |
| `btc_backtest_results.parquet` | Per-strategy backtest results |

## Key files

| File | Role |
|---|---|
| `features/assembly.py` | Assemble + split + scale 191 features |
| `models/volatility.py` | Train LightGBM ATR volatility model |
| `models/direction_dl.py` | Train CNN-LSTM direction models |
| `models/ensemble.py` | LightGBM + CNN-LSTM weighted ensemble |
| `strategy/agent.py` | Signal functions S1–S13 + DEFAULT_PARAMS |
| `execution/exit.py` | FixedExit, ATRDynamicExit, ComboExit |
| `execution/sizing.py` | FixedFraction, VolScaledSizer |
| `execution/entry.py` | MarketEntry, ConfirmEntry, SpreadEntry |
| `execution/config.py` | EXECUTION_CONFIG per strategy |
| `backtest/engine.py` | Bar-by-bar simulator |
| `backtest/run.py` | Strategy backtest runner |
| `backtest/preds.py` | Cached vol + direction predictions |
| `backtest/costs.py` | OKX fee model |

## Known structural problems

**1. Val/test gap is the core unsolved problem (Phase 2 target)**
- S1 val Sharpe +7.02 → test -0.81. S8 +2.42 → -6.12.
- No mechanism currently to identify when each strategy can be trusted

**2. S2_Funding signal broken**
- 90–95% of trades exit via TIME stop (funding reversion takes >60 bars)
- Fix: widen time_stop from 60 → 240 bars, tighten signal threshold

**3. Drop structural failures**
- S5_OFIScalp: 330+ trades, 12% win rate
- S9_LargeOrd: 447 trades, 21% win rate
- S11_Basis: 337 trades, 22% win rate, -22 Sharpe
- S13_OBDiv: 79 trades, 14% win rate, -27 Sharpe
