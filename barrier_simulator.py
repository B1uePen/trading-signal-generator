"""
Векторний симулятор ф'ючерсних угод з фіксованим співвідношенням SL:TP та портфельний бектестер
==============================================================================================
Чесний симулятор угод (базово SL = 1.5%, TP = 4.5%, H = 36 год) та послідовний портфельний бектестер
без накладання угод (Zero Overlapping Trades) з механізмом Margin Call (ліквідація при Equity <= 0)
на чистому NumPy.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROUND_TRIP_COST_DEFAULT = 0.0010  # 0.04% вхід + 0.04% вихід + 0.01%*2 проковзування = 0.10%


def _build_1m_lookup(df_1m: pd.DataFrame) -> Dict[np.datetime64, Tuple[np.ndarray, np.ndarray]]:
    """Індексує 1-хвилинні (high, low) за годинами для уточнення спірних 1h-свічок."""
    df_sorted = df_1m.sort_values("timestamp")
    m1_ts = pd.to_datetime(df_sorted["timestamp"]).to_numpy()
    m1_high = df_sorted["perp_high"].to_numpy(dtype=np.float64)
    m1_low = df_sorted["perp_low"].to_numpy(dtype=np.float64)
    hour_floor = m1_ts.astype("datetime64[h]").astype(m1_ts.dtype)
    unique_hours, starts = np.unique(hour_floor, return_index=True)
    ends = np.append(starts[1:], len(m1_ts))
    return {h: (m1_high[s:e], m1_low[s:e]) for h, s, e in zip(unique_hours, starts, ends)}


def simulate_barriers_1to3(
    df_bars: pd.DataFrame,
    sl_pct: float = 0.01,
    tp_pct: float = 0.03,
    horizon_hours: int = 24,
    round_trip_cost: float = ROUND_TRIP_COST_DEFAULT,
    df_1m: Optional[pd.DataFrame] = None,
    entry_on_next_bar: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Векторизований симулятор ф'ючерсних угод для кожного бару t:
      - entry_on_next_bar=True (чесний режим без витоку з майбутнього): вхід за ціною відкриття бару t+1,
        вікно утримання охоплює бари [t+1 .. t+H]; інакше вхід за open[t], вікно [t .. t+H-1].
      - Песимістичне правило: якщо в одній свічці пробито і TP, і SL — зараховується Stop-Loss
        (за наявності 1-хвилинних даних спірна свічка розв'язується по хвилинах).
      - Враховується комісія + проковзування (round_trip_cost) та funding_rate о 00:00, 08:00, 16:00 UTC.

    Повертає вектори довжини N: pnl_{long,short}, exit_idx_{long,short}, outcome_{long,short}
    (+1 TP, -1 SL, 0 Horizon) та entry_price.
    """
    n_bars = len(df_bars)
    H = int(horizon_hours)
    offset = 1 if entry_on_next_bar else 0

    p_open = df_bars["perp_open"].to_numpy(dtype=np.float64)
    p_high = df_bars["perp_high"].to_numpy(dtype=np.float64)
    p_low = df_bars["perp_low"].to_numpy(dtype=np.float64)
    p_close = df_bars["perp_close"].to_numpy(dtype=np.float64)

    funding = (
        np.nan_to_num(df_bars["funding_rate"].to_numpy(dtype=np.float64), nan=0.0)
        if "funding_rate" in df_bars.columns else np.zeros(n_bars)
    )
    if "timestamp" in df_bars.columns:
        ts_pd = pd.to_datetime(df_bars["timestamp"])
        ts_hours, ts_arr = ts_pd.dt.hour.to_numpy(dtype=np.int32), ts_pd.to_numpy()
    else:
        ts_hours, ts_arr = np.arange(n_bars, dtype=np.int32) % 24, None

    # Префіксні суми фандінгу (00:00, 08:00, 16:00 UTC) для O(1)-підсумовування на інтервалі утримання
    effective_funding = np.where(np.isin(ts_hours, (0, 8, 16)), funding, 0.0)
    funding_cumsum = np.concatenate([[0.0], np.cumsum(effective_funding)])

    m1_lookup = _build_1m_lookup(df_1m) if (df_1m is not None and "timestamp" in df_1m.columns and ts_arr is not None) else None

    entry_prices = np.copy(p_open)
    if entry_on_next_bar and n_bars > 1:
        entry_prices[:-1] = p_open[1:]
        entry_prices[-1] = p_close[-1]

    result = {
        "pnl_long": np.zeros(n_bars), "exit_idx_long": np.arange(n_bars, dtype=np.int64),
        "outcome_long": np.zeros(n_bars, dtype=np.int8),
        "pnl_short": np.zeros(n_bars), "exit_idx_short": np.arange(n_bars, dtype=np.int64),
        "outcome_short": np.zeros(n_bars, dtype=np.int8),
        "entry_price": entry_prices,
    }
    valid_n = n_bars - offset
    if valid_n <= 0:
        return result

    # 2D-вікна (valid_n, H) для векторизованої перевірки бар'єрів
    steps = np.arange(valid_n)[:, None] + offset + np.arange(H)[None, :]
    win_high = np.pad(p_high, (0, H + 1), mode="edge")[steps]
    win_low = np.pad(p_low, (0, H + 1), mode="edge")[steps]
    valid_step = steps < n_bars

    entries = entry_prices[:valid_n, None]
    idx_range = np.arange(valid_n)
    start_bar = idx_range + offset
    horizon_end_idx = np.minimum(start_bar + H - 1, n_bars - 1)

    for side, sign in (("long", 1.0), ("short", -1.0)):
        tp_lvl = entries * (1.0 + sign * tp_pct)
        sl_lvl = entries * (1.0 - sign * sl_pct)
        hit_tp = ((win_high >= tp_lvl) if sign > 0 else (win_low <= tp_lvl)) & valid_step
        hit_sl = ((win_low <= sl_lvl) if sign > 0 else (win_high >= sl_lvl)) & valid_step
        hit_any = hit_tp | hit_sl
        has_hit = np.any(hit_any, axis=1)
        first_step = np.argmax(hit_any, axis=1)
        first_bar = np.where(has_hit, start_bar + first_step, horizon_end_idx)

        tp_first = hit_tp[idx_range, first_step]
        sl_first = hit_sl[idx_range, first_step]
        outcome = np.zeros(valid_n, dtype=np.int8)
        outcome[has_hit & tp_first & ~sl_first] = 1
        outcome[has_hit & sl_first] = -1  # Песимістичний SL при одночасному пробитті

        if m1_lookup is not None:
            for i_bar in np.where(has_hit & tp_first & sl_first)[0]:
                h_ts = ts_arr[int(first_bar[i_bar])]
                if h_ts not in m1_lookup:
                    continue
                m_hi, m_lo = m1_lookup[h_ts]
                m_tp = (m_hi >= tp_lvl[i_bar, 0]) if sign > 0 else (m_lo <= tp_lvl[i_bar, 0])
                m_sl = (m_lo <= sl_lvl[i_bar, 0]) if sign > 0 else (m_hi >= sl_lvl[i_bar, 0])
                m_any = m_tp | m_sl
                if np.any(m_any):
                    outcome[i_bar] = -1 if m_sl[int(np.argmax(m_any))] else 1  # SL, якщо в ту ж хвилину обидва

        e0 = entry_prices[:valid_n]
        horizon_ret = sign * (p_close[first_bar] - e0) / (e0 + 1e-12)
        raw_ret = np.where(outcome == 1, tp_pct, np.where(outcome == -1, -sl_pct, horizon_ret))
        fund_sum = funding_cumsum[first_bar + 1] - funding_cumsum[start_bar]
        # Long сплачує додатний фандінг, Short його отримує
        result[f"pnl_{side}"][:valid_n] = raw_ret - round_trip_cost - sign * fund_sum
        result[f"exit_idx_{side}"][:valid_n] = first_bar
        result[f"outcome_{side}"][:valid_n] = outcome

    if entry_on_next_bar:
        result["exit_idx_long"][-1] = n_bars - 1
        result["exit_idx_short"][-1] = n_bars - 1
    return result


def _resolve_position_multipliers(
    signals: np.ndarray,
    conviction: Optional[np.ndarray],
    high_quantile: float = 0.75,
    conviction_threshold_2x: Optional[float] = None,
) -> np.ndarray:
    """
    Перетворює впевненість моделі на дискретні множники позиції {0, 1, 2}:
    5 станів портфеля: 0 (Кеш), +1x / -1x (Базовий), +2x / -2x (Посилений топ-квантиль).
    """
    active = signals != 0
    if not np.any(active):
        return np.zeros(len(signals), dtype=np.float64)
    if conviction is None:
        return active.astype(np.float64)

    conv = np.abs(np.asarray(conviction, dtype=np.float64))
    if np.all(np.isin(np.unique(conv[active]), [0.0, 1.0, 2.0])):  # Вже дискретні множники {1, 2}
        return np.where(active, np.where(conv >= 1.5, 2.0, 1.0), 0.0)

    if conviction_threshold_2x is None:
        scores = conv[active]
        conviction_threshold_2x = float(np.quantile(scores, high_quantile)) if len(scores) > 4 else float(np.max(scores) + 1.0)
    return np.where(active, np.where(conv >= conviction_threshold_2x, 2.0, 1.0), 0.0)


def compute_portfolio_metrics(
    bar_returns: np.ndarray,
    trade_pnls: np.ndarray,
    trade_dirs: np.ndarray,
    trade_mults: np.ndarray,
    bars_per_year: float = 8760.0,
) -> Dict[str, float]:
    """
    Повний набір фінансових метрик. Margin Call: якщо капітал впав до нуля (Equity <= 0),
    фіксуються Net PnL = -100%, MaxDD = 100%, CAGR = -100%.
    """
    n_bars, n_trades = len(bar_returns), len(trade_pnls)
    net_pnl = float(np.sum(bar_returns))
    equity_curve = 1.0 + np.cumsum(bar_returns)
    peaks = np.maximum.accumulate(np.maximum(equity_curve, 1e-8))
    max_dd = float(np.abs(np.min((equity_curve - peaks) / peaks))) if n_bars > 0 else 0.0

    years = max(n_bars / bars_per_year, 1e-6)
    final_eq = max(float(equity_curve[-1]) if n_bars > 0 else 1.0, 1e-4)
    cagr = float(final_eq ** (1.0 / years) - 1.0)

    if n_bars > 0 and np.min(equity_curve) <= 1e-8:
        net_pnl, max_dd, cagr = -1.0, 1.0, -1.0

    mean_r = float(np.mean(bar_returns)) if n_bars > 0 else 0.0
    std_r = float(np.std(bar_returns)) if n_bars > 0 else 0.0
    sharpe = float(mean_r / (std_r + 1e-12) * np.sqrt(bars_per_year)) if std_r > 1e-12 else 0.0
    downside_std = float(np.sqrt(np.mean(np.minimum(0.0, bar_returns) ** 2))) if n_bars > 0 else 0.0
    sortino = float(mean_r / (downside_std + 1e-12) * np.sqrt(bars_per_year)) if downside_std > 1e-12 else 0.0

    win_rate = profit_factor = expectancy = long_pnl = short_pnl = ls_ratio = 0.0
    n_long = n_short = n_2x = 0
    if n_trades > 0:
        wins, losses = trade_pnls > 0.0, trade_pnls < 0.0
        win_rate = float(np.mean(wins))
        gross_profit = float(np.sum(trade_pnls[wins])) if np.any(wins) else 0.0
        gross_loss = float(np.abs(np.sum(trade_pnls[losses]))) if np.any(losses) else 0.0
        profit_factor = float(gross_profit / (gross_loss + 1e-12)) if gross_loss > 0 else (10.0 if gross_profit > 0 else 0.0)
        expectancy = float(np.mean(trade_pnls))

        long_mask, short_mask = trade_dirs == 1, trade_dirs == -1
        n_long, n_short = int(np.sum(long_mask)), int(np.sum(short_mask))
        ls_ratio = float(n_long / max(n_short, 1))
        long_pnl = float(np.sum(trade_pnls[long_mask])) if n_long > 0 else 0.0
        short_pnl = float(np.sum(trade_pnls[short_mask])) if n_short > 0 else 0.0
        n_2x = int(np.sum(trade_mults >= 1.5))

    return {
        "net_pnl": net_pnl, "cagr": cagr, "sharpe": sharpe, "sortino": sortino, "max_dd": max_dd,
        "win_rate": win_rate, "profit_factor": profit_factor, "expectancy": expectancy,
        "n_trades": float(n_trades), "n_long": float(n_long), "n_short": float(n_short),
        "long_short_ratio": ls_ratio, "long_pnl": long_pnl, "short_pnl": short_pnl, "n_2x_trades": float(n_2x),
    }


def simulate_portfolio(
    signals: np.ndarray,
    conviction: Optional[np.ndarray],
    pnl_long: np.ndarray,
    pnl_short: np.ndarray,
    exit_idx_long: np.ndarray,
    exit_idx_short: np.ndarray,
    outcome_long: Optional[np.ndarray] = None,
    outcome_short: Optional[np.ndarray] = None,
    conviction_threshold_2x: Optional[float] = None,
    bars_per_year: float = 8760.0,
) -> Dict[str, object]:
    """
    Послідовний портфельний бектестер:
      - Не більше однієї відкритої угоди одночасно: поки попередня не закрилась на exit_idx, нові сигнали ігноруються.
      - 5 станів позиції: 0 (Кеш), +-1x (Базовий), +-2x (Посилений для топ-квантилю впевненості).
      - Margin Call: якщо кумулятивний капітал досягає нуля (100% втрата), угоду ліквідовано на цьому барі,
        PnL клампується до -100%, торгівля остаточно припиняється.
    """
    sig_arr = np.asarray(signals, dtype=np.int8)
    n_bars = len(sig_arr)
    mults = _resolve_position_multipliers(
        sig_arr, conviction, high_quantile=0.75, conviction_threshold_2x=conviction_threshold_2x
    )

    bar_returns = np.zeros(n_bars, dtype=np.float64)
    position_states = np.zeros(n_bars, dtype=np.int8)
    trades: List[Tuple[int, int, int, float, float, int]] = []  # (entry, exit, dir, mult, pnl, outcome)

    locked_until, cum_pnl, liquidated = -1, 0.0, False
    for t in np.flatnonzero((sig_arr != 0) & (mults > 0.0)):
        if t < locked_until:
            continue
        s, m = int(sig_arr[t]), float(mults[t])
        long_side = s == 1
        ex_idx = max(t, min(int((exit_idx_long if long_side else exit_idx_short)[t]), n_bars - 1))
        outcome_arr = outcome_long if long_side else outcome_short
        out_c = int(outcome_arr[t]) if outcome_arr is not None else 0
        realized = m * float((pnl_long if long_side else pnl_short)[t])

        # PnL рівномірно розподіляється по барах утримання [t+1 .. ex_idx] (або на бар t, якщо ex_idx == t)
        lo, hi = (t + 1, ex_idx + 1) if ex_idx > t else (t, t + 1)
        step = realized / max(1, ex_idx - t)
        path = np.cumsum(np.concatenate([[cum_pnl], np.full(hi - lo, step)]))[1:]

        liq = np.flatnonzero(path <= -1.0)
        if len(liq) > 0:
            j = int(liq[0])
            bar_returns[lo : lo + j] += step
            bar_returns[lo + j] = -1.0 - (path[j - 1] if j > 0 else cum_pnl)
            position_states[lo : lo + j + 1] = int(s * round(m))
            trades.append((t, lo + j, s, m, float(bar_returns[lo + j]), -1))
            liquidated = True
            break

        bar_returns[lo:hi] += step
        position_states[lo:hi] = int(s * round(m))
        cum_pnl = float(path[-1])
        trades.append((t, ex_idx, s, m, realized, out_c))
        locked_until = max(ex_idx, t + 1)  # Блокуємо нові угоди до закриття поточної

    cols = list(zip(*trades)) if trades else [[]] * 6
    trade_dirs_arr = np.asarray(cols[2], dtype=np.int8)
    trade_mults_arr = np.asarray(cols[3], dtype=np.float64)
    trade_pnls_arr = np.asarray(cols[4], dtype=np.float64)

    metrics = compute_portfolio_metrics(bar_returns, trade_pnls_arr, trade_dirs_arr, trade_mults_arr, bars_per_year)

    equity_curve = np.maximum(0.0, 1.0 + np.cumsum(bar_returns))
    peaks = np.maximum.accumulate(np.maximum(equity_curve, 1e-8))
    drawdown_curve = (equity_curve - peaks) / peaks
    if liquidated:
        drawdown_curve[np.isclose(equity_curve, 0.0)] = -1.0

    return {
        "bar_returns": bar_returns,
        "equity_curve": equity_curve,
        "drawdown_curve": drawdown_curve,
        "position_states": position_states,
        "trade_entries": np.asarray(cols[0], dtype=np.int64),
        "trade_exits": np.asarray(cols[1], dtype=np.int64),
        "trade_dirs": trade_dirs_arr,
        "trade_mults": trade_mults_arr,
        "trade_pnls": trade_pnls_arr,
        "trade_outcomes": np.asarray(cols[5], dtype=np.int8),
        "metrics": metrics,
    }
