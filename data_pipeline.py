"""
Підготовка даних та антивитік (Zero Lookahead Leakage)
======================================================
Чесний конвеєр обробки 1-хвилинних даних BTC/USDT (2021-12-01 .. 2026-09-30)
виключно на чистому NumPy та Pandas без жодних сторонніх ML-бібліотек.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_RAW_PATH = r"C:\Users\Public\роботи\диплом\датасети\btc_1m_full_2019_2026_V2.parquet"
PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / "cache"
FALLBACK_SLICE_PATH = CACHE_DIR / "btc_1m_slice_2021_2026.parquet"


class NumPyStandardScaler:
    """
    Власна реалізація Z-score нормалізації виключно на чистому NumPy:
      1) Медіанне заповнення пропусків (NaN / Inf) медіанами Train-вибірки.
      2) Масштабування Z = (X - mean) / (std + 1e-8).
      3) Обрізання екстремальних викидів np.clip(Z, -clip_val, clip_val).
    Навчається (fit) ВИКЛЮЧНО на Train-вибірці для суворого запобігання витоку з майбутнього.
    """

    def __init__(self, clip_val: float = 5.0, eps: float = 1e-8) -> None:
        self.clip_val = float(clip_val)
        self.eps = float(eps)
        self.medians_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False

    def fit(self, X: np.ndarray) -> "NumPyStandardScaler":
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]

        # Заміна +/-inf на NaN перед обчисленням медіан
        X_clean = np.where(np.isfinite(X_arr), X_arr, np.nan)
        with np.errstate(all="ignore"):
            medians = np.nanmedian(X_clean, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        self.medians_ = medians

        # Заповнення пропусків медіанами Train-вибірки
        nan_mask = ~np.isfinite(X_clean)
        if np.any(nan_mask):
            X_filled = X_clean.copy()
            col_idx = np.where(nan_mask)[1]
            X_filled[nan_mask] = self.medians_[col_idx]
        else:
            X_filled = X_clean

        self.mean_ = np.mean(X_filled, axis=0)
        self.std_ = np.std(X_filled, axis=0)
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_ or self.medians_ is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("NumPyStandardScaler має бути навчений через .fit(X_train) перед .transform()!")

        was_1d = ( np.asarray(X).ndim == 1 )
        X_arr = np.asarray(X, dtype=np.float64)
        if was_1d:
            X_arr = X_arr[:, None]

        X_filled = np.where(np.isfinite(X_arr), X_arr, np.nan)
        nan_mask = ~np.isfinite(X_filled)
        if np.any(nan_mask):
            col_idx = np.where(nan_mask)[1]
            X_filled[nan_mask] = self.medians_[col_idx]

        Z = (X_filled - self.mean_) / (self.std_ + self.eps)
        Z_clipped = np.clip(Z, -self.clip_val, self.clip_val)
        return Z_clipped[:, 0] if was_1d else Z_clipped

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def compute_oi_price_quadrant_and_product(
    open_interest: pd.Series,
    perp_close: pd.Series,
    diff_periods: int = 5,
) -> Tuple[pd.Series, pd.Series]:
    """
    Чесний перерахунок delta_oi_x_delta_price та oi_price_quadrant (1..4)
    ПІСЛЯ застосування лагу публікації до open_interest:
      Квадрант 1: d_OI > 0, d_Price > 0 (Long Build-up)
      Квадрант 2: d_OI > 0, d_Price < 0 (Short Build-up)
      Квадрант 3: d_OI < 0, d_Price > 0 (Short Covering)
      Квадрант 4: d_OI < 0, d_Price < 0 (Long Liquidation)
      Квадрант 0: якщо приріст нульовий або NaN
    """
    d_oi = open_interest.pct_change(diff_periods, fill_method=None)
    d_p = perp_close.pct_change(diff_periods, fill_method=None)
    product = (d_oi * d_p).astype(np.float32)

    quad = np.zeros(len(open_interest), dtype=np.int8)
    doi_vals = d_oi.to_numpy(dtype=np.float64)
    dp_vals = d_p.to_numpy(dtype=np.float64)

    valid = np.isfinite(doi_vals) & np.isfinite(dp_vals)
    quad[valid & (doi_vals > 0) & (dp_vals > 0)] = 1
    quad[valid & (doi_vals > 0) & (dp_vals < 0)] = 2
    quad[valid & (doi_vals < 0) & (dp_vals > 0)] = 3
    quad[valid & (doi_vals < 0) & (dp_vals < 0)] = 4

    return product, pd.Series(quad, index=open_interest.index, dtype=np.int8)


def apply_publication_lags(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Застосовує чесні лаги публікації до 1-хвилинних даних перед ресемплінгом:
      - dvol_close: +60 хвилин
      - vix, dxy: +1440 хвилин (1 доба)
      - fng_index: 0 хвилин
      - open_interest, open_interest_usd, lsr_top_traders, taker_buy_sell_ratio: +5 хвилин
      - Нульові артефакти open_interest <= 0 замінюються на NaN + ffill
      - Перераховуються oi_price_quadrant та delta_oi_x_delta_price ПІСЛЯ зсуву open_interest на +5 хв.
    """
    df = df_1m.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    # 1. Очищення артефактів open_interest <= 0 до зсуву
    if "open_interest" in df.columns:
        oi = df["open_interest"].astype(np.float64)
        oi = oi.where(oi > 0.0, np.nan).ffill()
        # Лаг публікації +5 хвилин
        df["open_interest"] = oi.shift(5).astype(np.float32)

    if "open_interest_usd" in df.columns:
        if "open_interest" in df.columns and "perp_close" in df.columns:
            # Перераховуємо USD-еквівалент на основі зсунутого open_interest (без витоку)
            df["open_interest_usd"] = (
                df["open_interest"].astype(np.float64) * df["perp_close"].astype(np.float64)
            ).astype(np.float32)
        else:
            oi_usd = df["open_interest_usd"].astype(np.float64)
            oi_usd = oi_usd.where(oi_usd > 0.0, np.nan).ffill()
            df["open_interest_usd"] = oi_usd.shift(5).astype(np.float32)

    # 2. Лаг +5 хвилин для деривативних метрик топ-трейдерів та тейкерів
    for col_5m in ("lsr_top_traders", "taker_buy_sell_ratio"):
        if col_5m in df.columns:
            df[col_5m] = df[col_5m].ffill().shift(5).astype(np.float32)

    # 3. Лаг +60 хвилин для DVOL
    if "dvol_close" in df.columns:
        df["dvol_close"] = df["dvol_close"].ffill().shift(60).astype(np.float32)

    # 4. Лаг +1440 хвилин (1 доба) для макроіндексів VIX та DXY
    for col_1d in ("vix", "dxy"):
        if col_1d in df.columns:
            df[col_1d] = df[col_1d].ffill().shift(1440).astype(np.float32)

    # 5. fng_index: зсув 0 хвилин (вже містить вчорашнє значення)
    if "fng_index" in df.columns:
        df["fng_index"] = df["fng_index"].ffill().shift(0).astype(np.float32)

    # 6. Чесний перерахунок delta_oi_x_delta_price та oi_price_quadrant ПІСЛЯ зсуву OI на +5 хв
    if "open_interest" in df.columns and "perp_close" in df.columns:
        prod, quad = compute_oi_price_quadrant_and_product(
            df["open_interest"], df["perp_close"], diff_periods=5
        )
        df["delta_oi_x_delta_price"] = prod
        df["oi_price_quadrant"] = quad

    return df


def resample_bars(
    df_1m_lagged: pd.DataFrame,
    timeframe: str = "1h",
    start_date: str = "2021-12-01",
    end_date: str = "2026-09-30 23:59:59",
) -> pd.DataFrame:
    """
    Агрегує 1-хвилинні дані (з уже застосованими лагами публікації) у 1h (або 4h) бари.
    Кожна свічка [T, T + timeframe) має мітку часу відкриття T (label='left', closed='left').
    """
    df = df_1m_lagged.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    df = df.sort_index()

    agg_spec = {
        "first": ["perp_open", "spot_open"],
        "max": ["perp_high", "spot_high"],
        "min": ["perp_low", "spot_low"],
        "sum": ["perp_volume", "spot_volume", "volume_delta", "trades_count", "taker_buy_volume"],
        "last": [
            "perp_close", "spot_close", "cvd", "funding_rate", "open_interest", "open_interest_usd",
            "dvol_close", "fng_index", "dxy", "vix", "premium_index", "lsr_top_traders",
            "taker_buy_sell_ratio", "delta_oi_x_delta_price", "oi_price_quadrant",
        ],
        "mean": ["aggressor_imbalance"],
    }
    agg_dict = {c: how for how, cols in agg_spec.items() for c in cols if c in df.columns}

    bars = df.resample(timeframe, label="left", closed="left").agg(agg_dict)
    bars = bars.dropna(subset=["perp_open", "perp_high", "perp_low", "perp_close"])

    # Оновлюємо aggressor_imbalance на рівні бару (volume_delta / perp_volume)
    if "volume_delta" in bars.columns and "perp_volume" in bars.columns:
        bars["aggressor_imbalance"] = (
            bars["volume_delta"].astype(np.float64)
            / (bars["perp_volume"].astype(np.float64) + 1e-8)
        ).astype(np.float32)

    # Оновлюємо 1h/4h delta_oi_x_delta_price та oi_price_quadrant на основі зсунутого OI
    if "open_interest" in bars.columns and "perp_close" in bars.columns:
        bars["open_interest"] = bars["open_interest"].ffill()
        prod_bar, quad_bar = compute_oi_price_quadrant_and_product(
            bars["open_interest"], bars["perp_close"], diff_periods=1
        )
        bars["delta_oi_x_delta_price"] = prod_bar
        bars["oi_price_quadrant"] = quad_bar

    # Обчислюємо realized_vol_24h до обрізання по start_date, щоб не мати NaN на старті
    bars_per_24h = 24 if timeframe.lower() == "1h" else max(1, int(pd.Timedelta("24h") / pd.Timedelta(timeframe)))
    ret_1 = bars["perp_close"].pct_change(1, fill_method=None)
    bars["realized_vol_24h"] = ret_1.rolling(bars_per_24h, min_periods=2).std().astype(np.float32)

    bars = bars.reset_index()
    if start_date is not None:
        bars = bars[bars["timestamp"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        bars = bars[bars["timestamp"] <= pd.Timestamp(end_date)]

    bars = bars.reset_index(drop=True)
    return bars


def compute_weekend_gap_pct(df_bars: pd.DataFrame) -> np.ndarray:
    """
    Каузальний розрахунок неперекритого цінового розриву вихідних (CME Weekend Gap Magnet Effect):
      - Щоп'ятниці о 21:00 UTC (час закриття CME) фіксується рівень friday_close = perp_close.
      - Під час вихідних (п'ятниця >21:00, субота вся, неділя до 21:00 UTC) weekend_gap_pct = 0.0.
      - Починаючи з відкриття щотижневої сесії (неділя 22:00 UTC) до п'ятниці 20:00 UTC:
        якщо геп виник за вихідні і ще не закрився (свічка ще не торкнулася friday_close),
        weekend_gap_pct = (friday_close - perp_close) / perp_close.
      - Як тільки свічка торкнулася friday_close (perp_low <= friday_close <= perp_high),
        геп закривається, і weekend_gap_pct = 0.0 до наступних вихідних.
    """
    n = len(df_bars)
    weekend_gap_pct = np.zeros(n, dtype=np.float64)
    if "timestamp" not in df_bars.columns or "perp_close" not in df_bars.columns:
        return weekend_gap_pct

    timestamps = pd.to_datetime(df_bars["timestamp"])
    perp_close = df_bars["perp_close"].to_numpy(dtype=np.float64)
    perp_high = df_bars["perp_high"].to_numpy(dtype=np.float64) if "perp_high" in df_bars.columns else perp_close
    perp_low = df_bars["perp_low"].to_numpy(dtype=np.float64) if "perp_low" in df_bars.columns else perp_close

    dows = timestamps.dt.dayofweek.to_numpy()
    hours = timestamps.dt.hour.to_numpy()

    friday_close: Optional[float] = None
    gap_active = False
    for t in range(n):
        dow, hr = dows[t], hours[t]
        if dow == 4 and hr == 21:  # П'ятниця 21:00 UTC -> фіксація закриття CME
            friday_close, gap_active = perp_close[t], False
            continue
        if (dow == 4 and hr > 21) or dow == 5 or (dow == 6 and hr <= 21):  # Вихідні
            gap_active = False
            continue
        if dow == 6 and hr == 22:  # Відкриття нового тижня CME: геп активний, якщо є friday_close
            gap_active = friday_close is not None
        if gap_active:
            if perp_low[t] <= friday_close <= perp_high[t]:  # Геп закрито
                gap_active = False
            else:
                weekend_gap_pct[t] = (friday_close - perp_close[t]) / (perp_close[t] + 1e-12)

    return weekend_gap_pct


def build_feature_matrix(df_bars: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Формує матрицю ознак X виключно з рідних колонок датасету без складних технічних індикаторів
    (рівно 34 ознаки: 27 ринкових/деривативних + weekend_gap_pct + 4 швидких sin/cos + month + is_month_turn).
    Повертає (X_df, feature_names).
    """
    feats: Dict[str, pd.Series] = {}

    # 1. Нестаціонарні колонки -> стаціонарні відсоткові прирости .pct_change(1) (11 колонок)
    pct_candidates = [
        "perp_close",
        "perp_high",
        "perp_low",
        "perp_volume",
        "spot_close",
        "spot_volume",
        "open_interest",
        "open_interest_usd",
        "trades_count",
        "taker_buy_volume",
        "dxy",
    ]
    for col in pct_candidates:
        if col in df_bars.columns:
            s = df_bars[col].astype(np.float64)
            if col in ("perp_volume", "spot_volume", "trades_count", "taker_buy_volume"):
                s = s.where(s > 0.0, np.nan).ffill()
            pct = s.pct_change(1, fill_method=None)
            pct = pct.replace([np.inf, -np.inf], np.nan)
            feats[f"{col}_pct"] = pct

    # Для знакозмінних потоків обсягу (volume_delta та cvd) (2 колонки)
    if "volume_delta" in df_bars.columns and "perp_volume" in df_bars.columns:
        vd = df_bars["volume_delta"].astype(np.float64)
        vol_ma = df_bars["perp_volume"].astype(np.float64).rolling(24, min_periods=1).mean() + 1e-8
        feats["volume_delta_norm"] = vd / vol_ma
    if "cvd" in df_bars.columns and "perp_volume" in df_bars.columns:
        cvd_diff = df_bars["cvd"].astype(np.float64).diff(24)
        vol_ma24 = (df_bars["perp_volume"].astype(np.float64).rolling(24, min_periods=1).sum()) + 1e-8
        feats["cvd_change_24h_norm"] = cvd_diff / vol_ma24

    # 2. Відношення тіла та хвостів поточної свічки (2 колонки)
    if "perp_open" in df_bars.columns and "perp_close" in df_bars.columns:
        p_open = df_bars["perp_open"].astype(np.float64) + 1e-8
        p_close = df_bars["perp_close"].astype(np.float64)
        feats["candle_body_to_open"] = (p_close - p_open) / p_open

    if "perp_open" in df_bars.columns and "perp_high" in df_bars.columns and "perp_low" in df_bars.columns:
        p_open = df_bars["perp_open"].astype(np.float64) + 1e-8
        p_high = df_bars["perp_high"].astype(np.float64)
        p_low = df_bars["perp_low"].astype(np.float64)
        feats["candle_range_to_open"] = (p_high - p_low) / p_open

    # 3. Рідні стаціонарні колонки напряму (10 колонок)
    direct_stationary = [
        "funding_rate",
        "premium_index",
        "lsr_top_traders",
        "taker_buy_sell_ratio",
        "aggressor_imbalance",
        "delta_oi_x_delta_price",
        "dvol_close",
        "fng_index",
        "vix",
        "realized_vol_24h",
    ]
    for col in direct_stationary:
        if col in df_bars.columns:
            feats[col] = df_bars[col].astype(np.float64)
        elif col == "realized_vol_24h" and "perp_close" in df_bars.columns:
            feats["realized_vol_24h"] = (
                df_bars["perp_close"].astype(np.float64).pct_change(1, fill_method=None).rolling(24, min_periods=2).std()
            )

    # 4. Квадранти OI-Price (5 колонок)
    if "oi_price_quadrant" in df_bars.columns:
        q_series = df_bars["oi_price_quadrant"].fillna(0).astype(np.int8)
        feats["oi_price_quadrant"] = q_series.astype(np.float64)
        for q_val in (1, 2, 3, 4):
            feats[f"oi_quadrant_{q_val}"] = (q_series == q_val).astype(np.float64)

    # 5. Фундаментальна мікроструктурна ознака неперекритого гепу вихідного дня (1 колонка)
    feats["weekend_gap_pct"] = pd.Series(compute_weekend_gap_pct(df_bars), index=df_bars.index)

    # 6. Компактний інституційний часовий блок (6 ознак: 4 швидких sin/cos + month + is_month_turn)
    if "timestamp" in df_bars.columns:
        ts_s = pd.to_datetime(df_bars["timestamp"])
        hour = ts_s.dt.hour.to_numpy(dtype=np.float64)
        day_of_week = ts_s.dt.dayofweek.to_numpy(dtype=np.float64)
        month = ts_s.dt.month.to_numpy(dtype=np.float64)
        day = ts_s.dt.day.to_numpy(dtype=np.int32)
        days_in_month = ts_s.dt.days_in_month.to_numpy(dtype=np.int32)

        # Швидкі тригонометричні цикли (2 пари = 4 колонки)
        feats["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        feats["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        feats["dow_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
        feats["dow_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)

        # Повільна календарна ознака (1 колонка)
        feats["month"] = month

        # Інституційна метрика межі місяця (Turn-of-the-Month Effect) (1 колонка)
        # 1.0 для перших 3 днів (day <= 3) або останніх 3 днів місяця (day >= days_in_month - 2)
        is_turn = ((day <= 3) | (day >= days_in_month - 2)).astype(np.float64)
        feats["is_month_turn"] = is_turn

    X_df = pd.DataFrame(feats, index=df_bars.index)
    feature_names = list(X_df.columns)
    return X_df, feature_names


def load_raw_1m_dataset(
    raw_path: str = DEFAULT_RAW_PATH,
    warmup_start: str = "2021-11-25",
    end_limit: str = "2026-10-31",
) -> pd.DataFrame:
    """
    Завантажує сирий 1-хвилинний датасет із основного шляху або локального кешу воркспейсу.
    """
    p = Path(raw_path)
    try:
        if p.exists():
            df = pd.read_parquet(p)
        elif FALLBACK_SLICE_PATH.exists():
            df = pd.read_parquet(FALLBACK_SLICE_PATH)
        else:
            raise FileNotFoundError(f"Не знайдено датасет ні за шляхом {p}, ні у {FALLBACK_SLICE_PATH}")
    except PermissionError:
        if FALLBACK_SLICE_PATH.exists():
            df = pd.read_parquet(FALLBACK_SLICE_PATH)
        else:
            raise

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if warmup_start is not None:
        df = df[df["timestamp"] >= pd.Timestamp(warmup_start)]
    if end_limit is not None:
        df = df[df["timestamp"] <= pd.Timestamp(end_limit)]
    return df.sort_values("timestamp").reset_index(drop=True)


def prepare_dataset(
    raw_path: str = DEFAULT_RAW_PATH,
    timeframe: str = "1h",
    start_date: str = "2021-12-01",
    end_date: str = "2026-09-30 23:59:59",
    use_cache: bool = True,
    return_1m_for_barriers: bool = True,
) -> Dict[str, object]:
    """
    Повний конвеєр КРОКУ 1:
      1) Завантаження 1m-даних із запасом на лаги (2021-11-25 .. 2026-01-05).
      2) Застосування чесних лагів публікації та перерахунок квадрантів OI.
      3) Ресемплінг у 1h (або 4h) бари та кешування у coursework/cache/.
      4) Побудова матриці ознак X.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"bars_{timeframe}_{start_date[:10]}_{end_date[:10]}.parquet"

    df_1m_lagged: Optional[pd.DataFrame] = None
    if use_cache and cache_file.exists() and not return_1m_for_barriers:
        df_bars = pd.read_parquet(cache_file)
    else:
        df_1m_raw = load_raw_1m_dataset(raw_path=raw_path)
        df_1m_lagged = apply_publication_lags(df_1m_raw)
        # Зберігаємо додаткові 24 години після end_date для чесної розмітки бар'єрів останніх барів,
        # але сам df_bars обмежуємо періодом [start_date, end_date]
        df_bars = resample_bars(
            df_1m_lagged,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )
        df_bars.to_parquet(cache_file, index=False)

    X_df, feature_names = build_feature_matrix(df_bars)
    return {
        "bars": df_bars,
        "X_df": X_df,
        "feature_names": feature_names,
        "df_1m_lagged": df_1m_lagged,
        "cache_file": str(cache_file),
    }

