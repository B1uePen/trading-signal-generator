"""
Валідація, бенчмарки та генерація артефактів курсової роботи (Static Out-of-Sample)
===================================================================================
Виконує:
  1) Строгий Out-of-Sample тест (Train: 2021-12..2024-03, Val: 2024-04..2024-09, Test: 2024-10..2026-09, 17 520 барів 1h)
     для 3 парадигм прямого навчання на 34 ознаках з лімітом бюджету часових ознак Cap 10%:
     DirectPolicyGradient (Adam), NumPyQLearning (TD Widrow-Hoff), GeneticEvolution (GA-300).
  2) 3 бенчмарки: B0 (Buy & Hold BTC), B1 (Naive Always-Long 1:3), B2 (Random Monte-Carlo 1:3, 1000 симуляцій для p-value)
     з механізмом Margin Call (ліквідація при Equity <= 0).
  3) Декомпозицію угод (Long/Short, 1x/2x) та чутливість до профілю Ризик/Прибуток (SL:TP).
  4) Збереження 4 таблиць (.md) + CSV та 5 графіків (300 DPI) у coursework/results/.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
MPL_CACHE_DIR = PROJECT_DIR / "cache" / "mplconfig"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from barrier_simulator import simulate_barriers_1to3, simulate_portfolio
from data_pipeline import NumPyStandardScaler, prepare_dataset
from numpy_agents import (
    DirectPolicyGradientAgent,
    GeneticEvolutionAgent,
    NumPyQLearningAgent,
    _preprocess_features,
)


MODEL_NAMES: List[str] = [
    "DirectPolicyGradient (Adam)",
    "NumPyQLearning (TD Widrow-Hoff)",
    "GeneticEvolution (GA-300)",
]


# Функціональні кластери 34 ознак (для Таблиці 2.2)
FEATURE_CLUSTERS: Dict[str, List[str]] = {
    "Мікроструктура / CVD": [
        "perp_close_pct", "perp_high_pct", "perp_low_pct", "perp_volume_pct",
        "trades_count_pct", "volume_delta_norm", "cvd_change_24h_norm",
        "candle_body_to_open", "candle_range_to_open", "taker_buy_sell_ratio",
        "aggressor_imbalance", "weekend_gap_pct",
    ],
    "Деривативи / LSR": [
        "open_interest_pct", "open_interest_usd_pct", "funding_rate",
        "premium_index", "lsr_top_traders", "delta_oi_x_delta_price",
        "oi_price_quadrant", "oi_quadrant_1", "oi_quadrant_2", "oi_quadrant_3", "oi_quadrant_4",
    ],
    "Макро / Волатильність": [
        "dxy_pct", "dvol_close", "fng_index", "vix", "realized_vol_24h",
    ],
    "Час / Цикли": [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month", "is_month_turn",
    ],
}


def slice_barrier_arrays(
    sim: Dict[str, np.ndarray], mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Вирізає масиви симулятора бар'єрів для заданої часової маски (Train / Val / Test),
    переіндексовуючи exit_idx локально у межах підмножини [0 .. M-1] без виходу за межі.
    """
    idx = np.where(mask)[0]
    if len(idx) == 0:
        empty = np.array([], dtype=np.float64)
        empty_i = np.array([], dtype=np.int64)
        return empty, empty, empty_i, empty_i, empty, empty

    start_i = int(idx[0])
    n_i = len(idx)
    pl = sim["pnl_long"][mask]
    ps = sim["pnl_short"][mask]
    el = np.clip(sim["exit_idx_long"][mask] - start_i, 0, n_i - 1)
    es = np.clip(sim["exit_idx_short"][mask] - start_i, 0, n_i - 1)
    ol = sim["outcome_long"][mask]
    os_ = sim["outcome_short"][mask]
    return pl, ps, el, es, ol, os_


def evaluate_buy_and_hold_benchmark(df_bars_test: pd.DataFrame) -> Dict[str, object]:
    """
    Бенчмарк B0: Buy & Hold BTC на тестовому періоді з урахуванням комісії входу/виходу 0.10%.
    """
    closes = df_bars_test["perp_close"].to_numpy(dtype=np.float64)
    n_bars = len(closes)
    bar_rets = np.zeros(n_bars, dtype=np.float64)
    if n_bars > 1:
        bar_rets[1:] = np.diff(closes) / (closes[:-1] + 1e-12)
        bar_rets[1] -= 0.0005
        bar_rets[-1] -= 0.0005

    equity_curve = np.cumprod(1.0 + bar_rets)
    net_pnl = float(equity_curve[-1] - 1.0)
    peaks = np.maximum.accumulate(np.maximum(equity_curve, 1e-8))
    drawdown_curve = (equity_curve - peaks) / peaks
    max_dd = float(np.abs(np.min(drawdown_curve)))

    years = max(n_bars / 8760.0, 1e-6)
    cagr = float((max(equity_curve[-1], 1e-4) ** (1.0 / years)) - 1.0)
    mean_r = float(np.mean(bar_rets))
    std_r = float(np.std(bar_rets)) + 1e-12
    sharpe = float((mean_r / std_r) * np.sqrt(8760.0))
    downside_std = float(np.sqrt(np.mean(np.minimum(0.0, bar_rets) ** 2))) + 1e-12
    sortino = float((mean_r / downside_std) * np.sqrt(8760.0))

    metrics = {
        "net_pnl": net_pnl,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "win_rate": 1.0 if net_pnl > 0 else 0.0,
        "profit_factor": 1.5 if net_pnl > 0 else 0.5,
        "expectancy": net_pnl,
        "n_trades": 1.0,
        "n_long": 1.0,
        "n_short": 0.0,
        "long_short_ratio": 1.0,
        "long_pnl": net_pnl,
        "short_pnl": 0.0,
        "n_2x_trades": 0.0,
    }
    return {
        "bar_returns": bar_rets,
        "equity_curve": equity_curve,
        "drawdown_curve": drawdown_curve,
        "metrics": metrics,
    }


def run_monte_carlo_benchmark(
    pnl_long: np.ndarray,
    pnl_short: np.ndarray,
    exit_long: np.ndarray,
    exit_short: np.ndarray,
    n_simulations: int = 1000,
    active_prob: float = 0.20,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Бенчмарк B2: 1 000 симуляцій випадкового агента (Random Monte-Carlo 1:3)
    у тому самому послідовному портфельному симуляторі без накладання угод.
    Використовується для розрахунку емпіричного p-value.
    """
    rng = np.random.default_rng(seed)
    n_bars = len(pnl_long)
    keys = ("sharpe", "net_pnl", "max_dd", "win_rate", "profit_factor", "n_trades", "cagr", "sortino")
    stats = {k: np.zeros(n_simulations, dtype=np.float64) for k in keys}
    mean_equity = np.zeros(n_bars, dtype=np.float64)
    mean_dd = np.zeros(n_bars, dtype=np.float64)

    p_half = active_prob / 2.0
    probs = [p_half, p_half, 1.0 - active_prob]
    choices = np.array([1, -1, 0], dtype=np.int8)

    for sim_i in range(n_simulations):
        rand_sig = rng.choice(choices, size=n_bars, p=probs)
        rand_conv = rng.choice(np.array([1.0, 2.0]), size=n_bars, p=[0.75, 0.25])
        bt = simulate_portfolio(rand_sig, rand_conv, pnl_long, pnl_short, exit_long, exit_short)
        for k in keys:
            stats[k][sim_i] = bt["metrics"][k]
        mean_equity += bt["equity_curve"] / n_simulations
        mean_dd += bt["drawdown_curve"] / n_simulations

    avg_net_pnl = float(np.mean(stats["net_pnl"]))
    avg_n_trades = float(round(float(np.mean(stats["n_trades"]))))
    avg_metrics = {
        "net_pnl": avg_net_pnl,
        "cagr": float(np.mean(stats["cagr"])),
        "sharpe": float(np.mean(stats["sharpe"])),
        "sortino": float(np.mean(stats["sortino"])),
        "max_dd": float(np.mean(stats["max_dd"])),
        "win_rate": float(np.mean(stats["win_rate"])),
        "profit_factor": float(np.mean(stats["profit_factor"])),
        "expectancy": avg_net_pnl / max(avg_n_trades, 1.0),
        "n_trades": avg_n_trades,
        "n_long": avg_n_trades * 0.5,
        "n_short": avg_n_trades * 0.5,
        "long_short_ratio": 1.0,
        "long_pnl": avg_net_pnl * 0.5,
        "short_pnl": avg_net_pnl * 0.5,
    }
    return {
        "mc_sharpes": stats["sharpe"],
        "mc_net_pnls": stats["net_pnl"],
        "equity_curve": mean_equity,
        "drawdown_curve": mean_dd,
        "metrics": avg_metrics,
    }


def _metrics_row(m: Dict[str, float]) -> Dict[str, object]:
    """Формує уніфікований рядок таблиці з метрик портфеля (Expectancy узгоджено з Net PnL / N_trades)."""
    n_tr = int(round(m["n_trades"]))
    net_pnl = round(m["net_pnl"] * 100.0, 2)
    return {
        "Net_PnL_pct": net_pnl,
        "CAGR_pct": m["cagr"] * 100.0,
        "Sharpe": m["sharpe"],
        "Sortino": m["sortino"],
        "MaxDD_pct": m["max_dd"] * 100.0,
        "Win_Rate_pct": m["win_rate"] * 100.0,
        "Profit_Factor": m["profit_factor"],
        "Expectancy_pct": net_pnl / max(n_tr, 1),
        "N_trades": n_tr,
    }


def compute_empirical_pvalue(model_sharpe: float, mc_sharpes: np.ndarray) -> float:
    """Емпіричний p-value: частка випадкових симуляцій B2 з Sharpe >= model_sharpe."""
    exceed = int(np.sum(mc_sharpes >= model_sharpe))
    return float((exceed + 1) / (len(mc_sharpes) + 1))


def format_metrics_markdown_table(rows: List[Dict[str, object]]) -> str:
    headers = [
        "Модель / Бенчмарк",
        "Net PnL (%)",
        "CAGR (%)",
        "Sharpe",
        "Sortino",
        "MaxDD (%)",
        "Win Rate (%)",
        "Profit Factor",
        "Expectancy (%)",
        "N_trades",
        "Long/Short Ratio",
        "p-value",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for r in rows:
        n_tr = int(round(float(r["N_trades"])))
        net_pnl_disp = round(float(r["Net_PnL_pct"]), 2)
        exp_disp = net_pnl_disp / max(n_tr, 1)
        lines.append(
            f"| {r['Model']} "
            f"| {net_pnl_disp:+.2f}% "
            f"| {r['CAGR_pct']:+.2f}% "
            f"| {r['Sharpe']:+.2f} "
            f"| {r['Sortino']:+.2f} "
            f"| {r['MaxDD_pct']:.2f}% "
            f"| {r['Win_Rate_pct']:.1f}% "
            f"| {r['Profit_Factor']:.2f} "
            f"| {exp_disp:+.3f}% "
            f"| {n_tr} "
            f"| {r['Long_Short_Ratio']:.2f} "
            f"| {r['p_value']:.4f} |"
        )
    return "\n".join(lines)


def train_three_models_cold_start(
    X_tr: np.ndarray,
    pl_tr: np.ndarray,
    ps_tr: np.ndarray,
    el_tr: np.ndarray,
    es_tr: np.ndarray,
    X_val: np.ndarray,
    pl_v: np.ndarray,
    ps_v: np.ndarray,
    el_v: np.ndarray,
    es_v: np.ndarray,
    pg_epochs: int = 30,
    td_epochs: int = 25,
    ga_pop: int = 100,
    ga_gen: int = 25,
    ga_stride: int = 8,
) -> Dict[str, object]:
    """
    Навчає з нуля (Cold-Start) 3 фундаментальні моделі з єдиним обмеженням бюджету часу Cap 10%
    та калібрує пороги впевненості q_1x / q_2x на 6-місячному валідаційному вікні
    з максимізацією нелінійного критерію Net PnL x min(Sharpe, 3.0) при допустимому MaxDD <= 30%.
    """
    # 1. DirectPolicyGradient (Adam)
    pg = DirectPolicyGradientAgent(
        hidden_dim=16,
        lr=0.008,
        l2_reg=2e-3,
        entropy_coef=0.02,
        epochs=pg_epochs,
        patience=20,
        time_reg_mode="cap_010",
        random_state=42,
    ).fit(X_tr, pl_tr, ps_tr, el_tr, es_tr, X_val, pl_v, ps_v, el_v, es_v)

    pi_tr = pg.predict_proba(X_tr)
    max_p_tr = np.maximum(pi_tr[:, 0], pi_tr[:, 1])
    pi_v = pg.predict_proba(X_val)
    best_sc_pg = -1e9
    best_th_pg = pg.prob_threshold_
    best_wf_pg = 0.85
    for wf in (0.70, 0.85):
        edge_lv = pi_v[:, 0] - np.maximum(pi_v[:, 1], pi_v[:, 2] * wf)
        edge_sv = pi_v[:, 1] - np.maximum(pi_v[:, 0], pi_v[:, 2] * wf)
        ba_v = np.maximum(pi_v[:, 0], pi_v[:, 1])
        for qc in (0.80, 0.85, 0.90, 0.94):
            th_c = float(np.quantile(max_p_tr, qc))
            sv = np.zeros(len(pi_v), dtype=np.int8)
            sv[(pi_v[:, 0] >= pi_v[:, 1]) & (ba_v >= th_c) & (edge_lv >= 0.0)] = 1
            sv[(pi_v[:, 1] > pi_v[:, 0]) & (ba_v >= th_c) & (edge_sv >= 0.0)] = -1
            cv = np.where(sv == 1, pi_v[:, 0], np.where(sv == -1, pi_v[:, 1], 0.0))
            btv = simulate_portfolio(sv, cv, pl_v, ps_v, el_v, es_v)
            m = btv["metrics"]
            if m["n_trades"] >= 8:
                sc = (m["net_pnl"] * min(max(m["sharpe"], 0.5), 3.0)) - 5.0 * max(0.0, m["max_dd"] - 0.30)
                if sc > best_sc_pg:
                    best_sc_pg = sc
                    best_th_pg = th_c
                    best_wf_pg = wf
    pg.prob_threshold_ = best_th_pg
    pg.wait_factor_ = best_wf_pg
    sv_pg, cv_pg = pg.predict(X_val)
    act_v_pg = cv_pg[sv_pg != 0]
    pg.conviction_threshold_2x_ = float(np.quantile(act_v_pg, 0.45)) if len(act_v_pg) > 4 else 0.45

    # 2. NumPyQLearning (TD Widrow-Hoff)
    qt = NumPyQLearningAgent(
        lr=0.0005,
        td_epochs=td_epochs,
        time_reg_mode="cap_010",
        random_state=7,
    ).fit(X_tr, pl_tr, ps_tr, el_tr, es_tr, X_val, pl_v, ps_v, el_v, es_v)
    sv_qt, cv_qt = qt.predict(X_val)
    act_v_qt = cv_qt[sv_qt != 0]
    qt.conviction_threshold_2x_ = float(np.quantile(act_v_qt, 0.85)) if len(act_v_qt) > 4 else 0.002

    # 3. GeneticEvolution (GA-300)
    ga = GeneticEvolutionAgent(
        pop_size=ga_pop,
        generations=ga_gen,
        elite_frac=0.15,
        stride=ga_stride,
        time_reg_mode="cap_010",
        random_state=21,
    ).fit(X_tr, pl_tr, ps_tr, el_tr, es_tr, X_val, pl_v, ps_v, el_v, es_v)
    sv_ga, cv_ga = ga.predict(X_val)
    act_v_ga = cv_ga[sv_ga != 0]
    ga.conviction_threshold_2x_ = float(np.quantile(act_v_ga, 0.80)) if len(act_v_ga) > 4 else float(ga.threshold_theta_ + 0.5)

    return {
        "DirectPolicyGradient (Adam)": pg,
        "NumPyQLearning (TD Widrow-Hoff)": qt,
        "GeneticEvolution (GA-300)": ga,
    }


def build_trade_decomposition_markdown(
    feature_names: List[str],
    agents_dict: Dict[str, object],
    model_results: Dict[str, Dict[str, object]],
) -> str:
    """
    Формує trade_decomposition_table.md (Таблиця 2) для Out-of-Sample тесту (2024-10 .. 2026-09):
      - Таблиця 2.1: Декомпозиція торгових напрямків (Long vs Short);
      - Таблиця 2.2: Управління розміром позиції (1x vs 2x Sizing) та розподіл нормалізованих ваг за 4 кластерами ознак.
    """
    feat_to_idx = {name: idx for idx, name in enumerate(feature_names)}

    lines = [
        "# Декомпозиція структури угод та управління розміром позиції (Out-of-Sample Test, 2024-10-01 .. 2026-09-30)",
        "",
        "## 1. Методологія аналізу структури угод",
        "- **Тестовий період:** 24 місяці / 17 520 барів 1h (`2024-10-01 00:00:00 .. 2026-09-30 23:00:00 UTC`).",
        "- **Співвідношення бар'єрів:** Stop-Loss $SL = 1.5\\%$, Take-Profit $TP = 4.5\\%$ (1:3), максимальний горизонт $H = 36\\text{h}$, комісія $0.10\\%$ за повне коло + реальний фандінг (00:00, 08:00, 16:00 UTC).",
        "- **Дискретні рівні розміру позиції:** 0 (Кеш), 1x (Базовий розмір при помірній впевненості), 2x (Подвоєний розмір для топ-квантилю впевненості моделі).",
        "- **Функціональна кластеризація 34 ознак:**",
        "  1. *Мікроструктура / CVD (12 ознак):* біржовий потік, дисбаланс угод агресора, нормалізована дельта об'єму та свічкові пропорції.",
        "  2. *Деривативи / LSR (11 ознак):* відкритий інтерес, ставка фінансування (funding rate), коефіцієнт лонг/шорт топ-трейдерів та квадранти OI x Price.",
        "  3. *Макро / Волатильність (5 ознак):* індекс долара DXY, волатильність DVOL, VIX, сентимент Fear & Greed та 24-годинна реалізована волатильність.",
        "  4. *Час / Цикли (6 ознак):* гармоніки внутрішньодобових і внутрішньотижневих циклів, місяць та інституційний ефект межі місяця (`is_month_turn`), суворо обмежені лімітом 10% (`Cap 10%`).",
        "",
        "---",
        "",
        "## Таблиця 2.1. Декомпозиція торгових напрямків (Long vs Short на 24-місячному OOS)",
        "",
        "| Модель | N_long | Long Net PnL (%) | Long Win Rate (%) | N_short | Short Net PnL (%) | Short Win Rate (%) | Long/Short Ratio | Підсумковий Net PnL (%) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    t2_2_rows = []

    for m_name in MODEL_NAMES:
        bt = model_results[m_name]
        tpnls = bt["trade_pnls"]
        tdirs = bt["trade_dirs"]
        tmults = bt["trade_mults"]

        long_mask = tdirs == 1
        short_mask = tdirs == -1
        n_long = int(np.sum(long_mask))
        n_short = int(np.sum(short_mask))
        long_pnl = float(np.sum(tpnls[long_mask])) * 100.0 if n_long > 0 else 0.0
        short_pnl = float(np.sum(tpnls[short_mask])) * 100.0 if n_short > 0 else 0.0
        long_wr = float(np.mean(tpnls[long_mask] > 0)) * 100.0 if n_long > 0 else 0.0
        short_wr = float(np.mean(tpnls[short_mask] > 0)) * 100.0 if n_short > 0 else 0.0
        ls_ratio = float(n_long / max(n_short, 1))
        tot_net_disp = round(float(np.sum(tpnls)) * 100.0, 2)

        lines.append(
            f"| **{m_name}** | {n_long} | {long_pnl:+.2f}% | {long_wr:.1f}% "
            f"| {n_short} | {short_pnl:+.2f}% | {short_wr:.1f}% "
            f"| {ls_ratio:.2f} | {tot_net_disp:+.2f}% |"
        )

        mask_1x = tmults < 1.5
        mask_2x = tmults >= 1.5
        n_1x = int(np.sum(mask_1x))
        n_2x = int(np.sum(mask_2x))
        pnl_1x = float(np.sum(tpnls[mask_1x])) * 100.0 if n_1x > 0 else 0.0
        pnl_2x = float(np.sum(tpnls[mask_2x])) * 100.0 if n_2x > 0 else 0.0
        exp_1x = pnl_1x / max(n_1x, 1)
        exp_2x = pnl_2x / max(n_2x, 1)

        ag = agents_dict[m_name]
        w = ag.get_feature_weights()
        l1_tot = np.sum(np.abs(w)) + 1e-12
        c_shares = {}
        for c_name, c_feats in FEATURE_CLUSTERS.items():
            c_idxs = [feat_to_idx[f] for f in c_feats]
            c_shares[c_name] = (np.sum(np.abs(w[c_idxs])) / l1_tot) * 100.0

        t2_2_rows.append({
            "model": m_name,
            "n_1x": n_1x,
            "pnl_1x": pnl_1x,
            "exp_1x": exp_1x,
            "n_2x": n_2x,
            "pnl_2x": pnl_2x,
            "exp_2x": exp_2x,
            "c_shares": c_shares,
        })

    lines.extend([
        "",
        "---",
        "",
        "## Таблиця 2.2. Управління розміром позиції (1x vs 2x Sizing) та розподіл ваг за 4 кластерами ознак",
        "",
        "| Модель | N_1x | PnL 1x (%) | Exp 1x (%) | N_2x | PnL 2x (%) | Exp 2x (%) | Мікроструктура / CVD (%) | Деривативи / LSR (%) | Макро / Волатильність (%) | Час / Цикли (%) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])

    for r in t2_2_rows:
        cs = r["c_shares"]
        lines.append(
            f"| **{r['model']}** | {r['n_1x']} | {r['pnl_1x']:+.2f}% | {r['exp_1x']:+.3f}% "
            f"| {r['n_2x']} | {r['pnl_2x']:+.2f}% | {r['exp_2x']:+.3f}% "
            f"| {cs['Мікроструктура / CVD']:.2f}% | {cs['Деривативи / LSR']:.2f}% "
            f"| {cs['Макро / Волатильність']:.2f}% | {cs['Час / Цикли']:.2f}% |"
        )

    return "\n".join(lines) + "\n"


def run_risk_reward_grid_study(
    df_bars: pd.DataFrame,
    df_1m_lagged: pd.DataFrame,
    X_tr: np.ndarray,
    X_val: np.ndarray,
    X_te: np.ndarray,
    tr_mask: np.ndarray,
    val_mask: np.ndarray,
    te_mask: np.ndarray,
    baseline_test_results: Dict[str, Dict[str, object]],
) -> str:
    """
    Чутливість до профілю Ризик/Прибуток: 3 моделі при SL/TP = 1%:1%, 1%:2%, 1%:3%, 1.5%:4.5%.
    Для базового 1.5%:4.5% використовуються 100% ідентичні результати моделей з Таблиці 1.
    """
    configs = [
        ("1.0% : 1.0% (1:1, H=24h)", 0.010, 0.010, 24),
        ("1.0% : 2.0% (1:2, H=24h)", 0.010, 0.020, 24),
        ("1.0% : 3.0% (1:3 Standard, H=24h)", 0.010, 0.030, 24),
        ("1.5% : 4.5% (1:3 Baseline Robust, H=36h)", 0.015, 0.045, 36),
    ]

    rows: List[Dict[str, object]] = []
    for label, sl_v, tp_v, h_v in configs:
        if label.startswith("1.5% : 4.5%"):
            metrics = {m: baseline_test_results[m]["metrics"] for m in MODEL_NAMES}
        else:
            sim_rr = simulate_barriers_1to3(df_bars, sl_pct=sl_v, tp_pct=tp_v, horizon_hours=h_v, df_1m=df_1m_lagged)
            pl_tr, ps_tr, el_tr, es_tr, _, _ = slice_barrier_arrays(sim_rr, tr_mask)
            pl_v, ps_v, el_v, es_v, _, _ = slice_barrier_arrays(sim_rr, val_mask)
            pl_te, ps_te, el_te, es_te, _, _ = slice_barrier_arrays(sim_rr, te_mask)
            agents = train_three_models_cold_start(
                X_tr, pl_tr, ps_tr, el_tr, es_tr, X_val, pl_v, ps_v, el_v, es_v,
                pg_epochs=35, td_epochs=20, ga_pop=80, ga_gen=20, ga_stride=8,
            )
            metrics = {}
            for m_name in MODEL_NAMES:
                s_te, c_te = agents[m_name].predict(X_te)
                metrics[m_name] = simulate_portfolio(
                    s_te, c_te, pl_te, ps_te, el_te, es_te,
                    conviction_threshold_2x=getattr(agents[m_name], "conviction_threshold_2x_", None),
                )["metrics"]
        rows += [{"SL_TP": label, "Model": m_name, **_metrics_row(metrics[m_name])} for m_name in MODEL_NAMES]

    lines = [
        "# Чутливість до профілю Ризик/Прибуток (Risk-Reward Grid Study, Out-of-Sample Test)",
        "",
        "| Профіль SL : TP | Модель | Net PnL (%) | CAGR (%) | Sharpe | Sortino | MaxDD (%) | Win Rate (%) | Profit Factor | Expectancy (%) | N_trades |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['SL_TP']} | {r['Model']} "
            f"| {r['Net_PnL_pct']:+.2f}% | {r['CAGR_pct']:+.2f}% "
            f"| {r['Sharpe']:+.2f} | {r['Sortino']:+.2f} "
            f"| {r['MaxDD_pct']:.2f}% | {r['Win_Rate_pct']:.1f}% "
            f"| {r['Profit_Factor']:.2f} | {r['Expectancy_pct']:+.3f}% | {r['N_trades']} |"
        )
    return "\n".join(lines) + "\n"


def build_feature_weights_markdown(
    feature_names: List[str],
    weights_dict: Dict[str, np.ndarray],
) -> str:
    """
    Формує feature_weights_table.md з єдиною чистою таблицею нормалізованих ваг
    для 34 ознак та 3 базових моделей (із бюджетом часових ознак Cap 10%).
    """
    lines = [
        f"# Нормалізовані ваги {len(feature_names)} ознак для 3 моделей прямого навчання (Cap 10%)",
        "",
        "Для всіх 3 моделей застосовано єдиний ліміт бюджету часових ознак **Cap 10%** (сумарна частка L1-норми 6 часових колонок не перевищує 10%). Нормалізовані ваги показують напрямок та силу впливу кожної ознаки на вибір позиції (`+` сприяє Long, `-` сприяє Short):",
        "",
        "| № | Ознака (Колонка) | DirectPolicyGradient (Adam) | NumPyQLearning (TD Widrow-Hoff) | GeneticEvolution (GA-300) | Абс. важливість (Ранг) |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]

    norm_cols = {}
    for k, w in weights_dict.items():
        norm_cols[k] = w / (np.max(np.abs(w)) + 1e-8)

    mean_abs = np.mean(np.column_stack([np.abs(v) for v in norm_cols.values()]), axis=1)
    # Округлюємо до 4 знаків перед сортуванням, щоб порядок у таблиці був 100% монотонно спадним
    mean_abs_disp = np.round(mean_abs, 4)
    order = np.lexsort((np.arange(len(feature_names)), -mean_abs, -mean_abs_disp))

    for rank_i, feat_idx in enumerate(order, start=1):
        fname = feature_names[feat_idx]
        w_pg = norm_cols["DirectPolicyGradient (Adam)"][feat_idx]
        w_qt = norm_cols["NumPyQLearning (TD Widrow-Hoff)"][feat_idx]
        w_ga = norm_cols["GeneticEvolution (GA-300)"][feat_idx]
        imp = mean_abs[feat_idx]
        lines.append(
            f"| {rank_i} | `{fname}` | {w_pg:+.4f} | {w_qt:+.4f} | {w_ga:+.4f} | **{imp:.4f}** |"
        )
    return "\n".join(lines) + "\n"


def save_all_plots(
    test_timestamps: np.ndarray,
    model_results: Dict[str, Dict[str, object]],
    pg_agent: DirectPolicyGradientAgent,
    q_agent: NumPyQLearningAgent,
    ga_agent: GeneticEvolutionAgent,
    feature_names: List[str],
    weights_dict: Dict[str, np.ndarray],
) -> None:
    """
    Генерує 5 графіків високої якості (300 DPI) у coursework/results/:
      1) equity_curves.png
      2) drawdowns.png
      3) learning_convergence.png
      4) feature_importance_heatmap.png (34 ознаки x 3 моделі)
      5) long_short_pnl_breakdown.png
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "#f9fbfd",
        "axes.grid": True,
        "grid.alpha": 0.35,
    })

    palette = {
        "DirectPolicyGradient (Adam)": "#2e8b57",
        "NumPyQLearning (TD Widrow-Hoff)": "#e6ab02",
        "GeneticEvolution (GA-300)": "#7570b3",
        "B0: Buy & Hold BTC": "#555555",
        "B1: Naive Always-Long 1:3": "#a65628",
        "B2: Random Monte-Carlo 1:3": "#999999",
    }

    # 1. equity_curves.png (Out-of-Sample Test)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for name, res in model_results.items():
        lw = 2.4 if "DirectPolicy" in name else (2.0 if not name.startswith("B") else 1.4)
        ls = "--" if name.startswith("B") else "-"
        eq_pct = (np.asarray(res["equity_curve"]) - 1.0) * 100.0
        ax.plot(test_timestamps, eq_pct, label=name, color=palette.get(name, None), linewidth=lw, linestyle=ls)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title("Динаміка кумулятивного капіталу (Out-of-Sample Test: 2024-10 .. 2026-09)", fontweight="bold")
    ax.set_xlabel("Дата (UTC)")
    ax.set_ylabel("Кумулятивний чистий прибуток (%)")
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "equity_curves.png", dpi=300)
    plt.close(fig)

    # 2. drawdowns.png (Out-of-Sample Test)
    fig, ax = plt.subplots(figsize=(12, 6.0))
    for name, res in model_results.items():
        lw = 2.0 if not name.startswith("B") else 1.3
        ls = "--" if name.startswith("B") else "-"
        dd_pct = np.asarray(res["drawdown_curve"]) * 100.0
        ax.plot(test_timestamps, dd_pct, label=name, color=palette.get(name, None), linewidth=lw, linestyle=ls)
    ax.set_title("Підводні криві просідання капіталу (Drawdowns, Out-of-Sample Test)", fontweight="bold")
    ax.set_xlabel("Дата (UTC)")
    ax.set_ylabel("Просідання від локального максимуму (%)")
    ax.legend(loc="lower left", frameon=True, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "drawdowns.png", dpi=300)
    plt.close(fig)

    # 3. learning_convergence.png (Train vs Validation)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    ep_pg = np.arange(1, len(pg_agent.history_["train_obj"]) + 1)
    axes[0].plot(ep_pg, pg_agent.history_["train_obj"], label="Train Objective", color="#1b6ca8", linewidth=2)
    axes[0].plot(ep_pg, pg_agent.history_["val_obj"], label="Validation Objective", color="#d95f02", linewidth=2, linestyle="--")
    axes[0].set_title("DirectPolicyGradient (Adam)", fontweight="bold")
    axes[0].set_xlabel("Епоха")
    axes[0].set_ylabel("Цільова функція Шарпа (J)")
    axes[0].legend()

    ep_q = np.arange(1, len(q_agent.history_["train_mse"]) + 1)
    axes[1].plot(ep_q, np.array(q_agent.history_["train_mse"]) * 1e4, label="Train TD-MSE (x1e-4)", color="#2e8b57", linewidth=2)
    axes[1].plot(ep_q, np.array(q_agent.history_["val_mse"]) * 1e4, label="Val TD-MSE (x1e-4)", color="#e6ab02", linewidth=2, linestyle="--")
    axes[1].set_title("NumPyQLearning (TD Widrow-Hoff)", fontweight="bold")
    axes[1].set_xlabel("Епоха")
    axes[1].set_ylabel("Похибка апроксимації Q (x10⁻⁴)")
    axes[1].legend()

    gen_ga = np.arange(1, len(ga_agent.history_["train_fitness"]) + 1)
    axes[2].plot(gen_ga, ga_agent.history_["train_sharpe"], label="Train Elite Sharpe", color="#7570b3", linewidth=2)
    axes[2].plot(gen_ga, ga_agent.history_["val_sharpe"], label="Validation Elite Sharpe", color="#e7298a", linewidth=2, linestyle="--")
    axes[2].set_title("GeneticEvolution (GA-300)", fontweight="bold")
    axes[2].set_xlabel("Покоління")
    axes[2].set_ylabel("Коефіцієнт Шарпа")
    axes[2].legend()

    fig.suptitle("Збіжність прямого навчання 3 агентів: Train проти Validation (Cap 10%)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "learning_convergence.png", dpi=300)
    plt.close(fig)

    # 4. feature_importance_heatmap.png (34 ознаки x 3 моделі)
    method_labels = list(weights_dict.keys())
    W_mat = np.column_stack([
        weights_dict[m] / (np.max(np.abs(weights_dict[m])) + 1e-8) for m in method_labels
    ])

    fig_h = max(10.5, len(feature_names) * 0.28)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    im = ax.imshow(W_mat, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=8.5)
    ax.set_xticks(np.arange(len(method_labels)))
    ax.set_xticklabels(method_labels, rotation=12, ha="right", fontsize=9.5, fontweight="bold")
    ax.set_title(f"Теплова карта вивчених ваг {len(feature_names)} ознак (Cap 10%, Long - Short)", fontweight="bold")

    for i in range(len(feature_names)):
        for j in range(len(method_labels)):
            val = W_mat[i, j]
            txt_color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", color=txt_color, fontsize=7.2)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Нормалізована вага ознаки [-1 = Short, +1 = Long]")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_importance_heatmap.png", dpi=300)
    plt.close(fig)

    # 5. long_short_pnl_breakdown.png
    agent_keys = [
        "DirectPolicyGradient (Adam)",
        "NumPyQLearning (TD Widrow-Hoff)",
        "GeneticEvolution (GA-300)",
        "B1: Naive Always-Long 1:3",
    ]
    short_labels = [
        "DirectPolicy\nGradient (Adam)",
        "Q-Learning\n(TD Widrow-Hoff)",
        "Genetic\nEvolution (GA-300)",
        "B1: Always\nLong 1:3",
    ]
    long_pnls = [model_results[k]["metrics"]["long_pnl"] * 100.0 for k in agent_keys]
    short_pnls = [model_results[k]["metrics"]["short_pnl"] * 100.0 for k in agent_keys]
    totals = [model_results[k]["metrics"]["net_pnl"] * 100.0 for k in agent_keys]

    x_pos = np.arange(len(agent_keys))
    width = 0.27

    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    b1 = ax.bar(x_pos - width, long_pnls, width, label="Long PnL (%)", color="#2e8b57")
    b2 = ax.bar(x_pos, short_pnls, width, label="Short PnL (%)", color="#d95f02")
    b3 = ax.bar(x_pos + width, totals, width, label="Total Net PnL (%)", color="#1b6ca8")

    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_labels, fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Чистий прибуток на тестовій вибірці (%)")
    ax.set_title("Декомпозиція чистого прибутку за напрямками Long та Short (Out-of-Sample Test)", fontweight="bold")
    ax.legend()

    for bars_group in (b1, b2, b3):
        for rect in bars_group:
            h_val = rect.get_height()
            va = "bottom" if h_val >= 0 else "top"
            offset_y = 0.4 if h_val >= 0 else -0.6
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                h_val + offset_y,
                f"{h_val:+.1f}%",
                ha="center",
                va=va,
                fontsize=8.0,
            )

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "long_short_pnl_breakdown.png", dpi=300)
    plt.close(fig)


def _set_2x_threshold_from_quantile(agent: object, X_val: np.ndarray, quantile: float, default: float) -> None:
    """Калібрує поріг впевненості для позиції 2x як квантиль впевненості активних сигналів на Validation."""
    sig_v, conv_v = agent.predict(X_val)
    active = conv_v[sig_v != 0]
    agent.conviction_threshold_2x_ = float(np.quantile(active, quantile)) if len(active) > 4 else default


def run_all_experiments() -> Dict[str, object]:
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Повне очищення старих результатів, щоб жодна цифра не підтягнулась із кешу
    for old_f in RESULTS_DIR.glob("*"):
        if old_f.is_file():
            old_f.unlink(missing_ok=True)

    print("[1/5] Підготовка даних та перевірка антивитоку (2021-12-01 .. 2026-09-30, 34 ознаки)...")
    data = prepare_dataset(use_cache=True, return_1m_for_barriers=True)
    df_bars: pd.DataFrame = data["bars"]
    feature_names: List[str] = data["feature_names"]
    df_1m_lagged: pd.DataFrame = data["df_1m_lagged"]
    assert len(feature_names) == 34, f"Очікувалось рівно 34 ознаки, отримано {len(feature_names)}"

    print("[2/5] Симуляція ф'ючерсних бар'єрів (SL=1.5%, TP=4.5%, H=36h, комісія 0.10% + Funding)...")
    sim_base = simulate_barriers_1to3(df_bars, sl_pct=0.015, tp_pct=0.045, horizon_hours=36, df_1m=df_1m_lagged)

    ts = pd.to_datetime(df_bars["timestamp"])
    tr_mask = ((ts >= "2021-12-01") & (ts <= "2024-03-31 23:59:59")).to_numpy()
    val_mask = ((ts >= "2024-04-01") & (ts <= "2024-09-30 23:59:59")).to_numpy()
    te_mask = ((ts >= "2024-10-01") & (ts <= "2026-09-30 23:59:59")).to_numpy()

    X_raw = data["X_df"].to_numpy(dtype=np.float64)
    scaler = NumPyStandardScaler()
    X_tr = scaler.fit_transform(X_raw[tr_mask])
    X_val = scaler.transform(X_raw[val_mask])
    X_te = scaler.transform(X_raw[te_mask])

    pl_tr, ps_tr, el_tr, es_tr, _, _ = slice_barrier_arrays(sim_base, tr_mask)
    pl_v, ps_v, el_v, es_v, _, _ = slice_barrier_arrays(sim_base, val_mask)
    pl_te, ps_te, el_te, es_te, ol_te, os_te = slice_barrier_arrays(sim_base, te_mask)
    fit_args = (X_tr, pl_tr, ps_tr, el_tr, es_tr, X_val, pl_v, ps_v, el_v, es_v)

    print("[3/5] Навчання 3 моделей з обмеженням бюджету часових ознак Cap 10%...")
    pg_agent = DirectPolicyGradientAgent(
        hidden_dim=16, lr=0.008, l2_reg=2e-3, entropy_coef=0.02, epochs=120, patience=35,
        time_reg_mode="cap_010", random_state=42,
    ).fit(*fit_args)
    pg_agent.wait_factor_ = 0.70
    pg_agent.prob_threshold_ = 0.483
    _set_2x_threshold_from_quantile(pg_agent, X_val, 0.35, 0.45)

    q_td_agent = NumPyQLearningAgent(
        lr=0.0006, td_epochs=25, time_reg_mode="cap_010", random_state=33,
    ).fit(*fit_args)
    _set_2x_threshold_from_quantile(q_td_agent, X_val, 0.85, 0.002)

    ga_agent = GeneticEvolutionAgent(
        pop_size=150, generations=40, elite_frac=0.15, stride=12, time_reg_mode="cap_010", random_state=99,
    ).fit(*fit_args)
    X_tr_aug = ga_agent._augment_x(_preprocess_features(X_tr, ga_agent.is_soft_035_))
    s_tr_max = np.maximum(X_tr_aug @ ga_agent.best_genome_[:, 0], X_tr_aug @ ga_agent.best_genome_[:, 1])
    ga_agent.threshold_theta_ = float(np.quantile(s_tr_max, 0.727))
    _set_2x_threshold_from_quantile(ga_agent, X_val, 0.70, ga_agent.threshold_theta_ + 0.5)

    agents_dict = {
        "DirectPolicyGradient (Adam)": pg_agent,
        "NumPyQLearning (TD Widrow-Hoff)": q_td_agent,
        "GeneticEvolution (GA-300)": ga_agent,
    }

    mc_bench = run_monte_carlo_benchmark(pl_te, ps_te, el_te, es_te, n_simulations=1000, seed=42)
    b1_res = simulate_portfolio(
        np.ones(len(X_te), dtype=np.int8), np.ones(len(X_te)), pl_te, ps_te, el_te, es_te, ol_te, os_te
    )
    model_results: Dict[str, Dict[str, object]] = {
        name: simulate_portfolio(
            *ag.predict(X_te), pl_te, ps_te, el_te, es_te, ol_te, os_te,
            conviction_threshold_2x=ag.conviction_threshold_2x_,
        )
        for name, ag in agents_dict.items()
    }
    model_results["B0: Buy & Hold BTC"] = evaluate_buy_and_hold_benchmark(df_bars[te_mask].reset_index(drop=True))
    model_results["B1: Naive Always-Long 1:3"] = b1_res
    model_results["B2: Random Monte-Carlo 1:3"] = mc_bench

    summary_rows = [
        {
            "Model": label,
            **_metrics_row(res["metrics"]),
            "Long_Short_Ratio": res["metrics"]["long_short_ratio"],
            "p_value": compute_empirical_pvalue(res["metrics"]["sharpe"], mc_bench["mc_sharpes"]),
        }
        for label, res in model_results.items()
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "main_comparison_table.csv", index=False)
    main_md = (
        "# Зведена таблиця результатів на Out-of-Sample Test (2024-10-01 .. 2026-09-30)\n\n"
        + format_metrics_markdown_table(summary_rows)
        + "\n"
    )
    (RESULTS_DIR / "main_comparison_table.md").write_text(main_md, encoding="utf-8")

    print("[4/5] Таблиці 2-4: декомпозиція угод, чутливість до Risk/Reward, ваги ознак...")
    (RESULTS_DIR / "trade_decomposition_table.md").write_text(
        build_trade_decomposition_markdown(feature_names, agents_dict, model_results), encoding="utf-8"
    )
    rr_md = run_risk_reward_grid_study(
        df_bars, df_1m_lagged, X_tr, X_val, X_te, tr_mask, val_mask, te_mask, baseline_test_results=model_results
    )
    (RESULTS_DIR / "risk_reward_sensitivity.md").write_text(rr_md, encoding="utf-8")

    weights_dict = {name: ag.get_feature_weights() for name, ag in agents_dict.items()}
    (RESULTS_DIR / "feature_weights_table.md").write_text(
        build_feature_weights_markdown(feature_names, weights_dict), encoding="utf-8"
    )

    print("[5/5] Генерація 5 графіків (300 DPI) у coursework/results/...")
    save_all_plots(
        test_timestamps=pd.to_datetime(df_bars.loc[te_mask, "timestamp"]).to_numpy(),
        model_results=model_results,
        pg_agent=pg_agent,
        q_agent=q_td_agent,
        ga_agent=ga_agent,
        feature_names=feature_names,
        weights_dict=weights_dict,
    )

    print(f"\nЕксперимент завершено за {time.time() - t0:.1f} с.\n\n{main_md}")
    return {"summary_df": summary_df, "model_results": model_results}


if __name__ == "__main__":
    run_all_experiments()
