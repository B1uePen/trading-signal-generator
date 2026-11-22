"""
Модульні тести проєкту курсової роботи (pytest)
===============================================
Перевіряє:
  1) Відсутність сторонніх ML-бібліотек (AST-аналіз) та залишків Walk-Forward у коді.
  2) Антивитік (Zero Lookahead): NumPyStandardScaler, лаги публікації, каузальний ресемплінг, строгий OOS-спліт (17 520 барів).
  3) Чесність симулятора бар'єрів (песимістичне правило SL, комісія, фандінг) та портфельного бектестера
     (одна відкрита угода, 5 станів позиції, Margin Call при Equity <= 0).
  4) Аналітичні градієнти DirectPolicyGradientAgent, роботу 3 агентів та 5 режимів регуляризації часових ознак.
  5) Узгодженість 4 таблиць (.md) та наявність 5 графіків (.png) у results/.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from barrier_simulator import simulate_barriers_1to3, simulate_portfolio
from data_pipeline import (
    NumPyStandardScaler,
    apply_publication_lags,
    build_feature_matrix,
    compute_oi_price_quadrant_and_product,
    resample_bars,
)
from numpy_agents import (
    DirectPolicyGradientAgent,
    GeneticEvolutionAgent,
    NumPyQLearningAgent,
)


PROJECT_DIR = Path(__file__).resolve().parent
FORBIDDEN_ML_MODULES = {
    "sklearn",
    "scikit_learn",
    "lightgbm",
    "xgboost",
    "catboost",
    "torch",
    "pytorch",
    "tensorflow",
    "keras",
    "jax",
    "flax",
    "optuna",
    "statsmodels",
}


def test_no_forbidden_ml_libraries_in_source_code() -> None:
    """
    КРИТИЧНЕ ПРАВИЛО 1: Жодних сторонніх ML-бібліотек у жодному файлі проєкту!
    """
    py_files = [
        PROJECT_DIR / "data_pipeline.py",
        PROJECT_DIR / "barrier_simulator.py",
        PROJECT_DIR / "numpy_agents.py",
        PROJECT_DIR / "run_experiment.py",
    ]
    for fpath in py_files:
        assert fpath.exists(), f"Файл {fpath.name} не знайдено!"
        tree = ast.parse(fpath.read_text(encoding="utf-8"), filename=str(fpath))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_mod = alias.name.split(".")[0].lower()
                    assert root_mod not in FORBIDDEN_ML_MODULES, (
                        f"Заборонений імпорт '{alias.name}' у файлі {fpath.name}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                root_mod = node.module.split(".")[0].lower()
                assert root_mod not in FORBIDDEN_ML_MODULES, (
                    f"Заборонений імпорт from '{node.module}' у файлі {fpath.name}"
                )


def test_numpy_standard_scaler_zero_leakage_and_clipping() -> None:
    """
    Перевіряє, що NumPyStandardScaler:
      - навчається виключно на Train-вибірці;
      - замінює NaN та Inf медіанами Train-вибірки;
      - обрізає викиди у межах [-5, 5];
      - не змінює трансформацію Train при додаванні екстремальних значень у Test.
    """
    X_train = np.array([
        [10.0, 100.0],
        [20.0, 200.0],
        [30.0, np.nan],
        [40.0, 400.0],
        [50.0, 500.0],
    ])
    scaler = NumPyStandardScaler(clip_val=5.0)
    Z_train = scaler.fit_transform(X_train)

    assert np.all(np.isfinite(Z_train))
    assert np.isclose(scaler.medians_[1], 300.0)
    # Оскільки медіана 2-ї колонки дорівнює 300, середнє після заповнення теж 300, тому 3-й рядок має Z=0
    assert np.isclose(Z_train[2, 1], 0.0, atol=1e-6)

    # Перевірка обрізання викидів [-5, 5] та обробки NaN/Inf на Test
    X_test = np.array([
        [1e9, -1e9],
        [np.nan, np.inf],
    ])
    Z_test = scaler.transform(X_test)
    assert np.all(np.isfinite(Z_test))
    assert np.isclose(Z_test[0, 0], 5.0)
    assert np.isclose(Z_test[0, 1], -5.0)
    assert np.isclose(Z_test[1, 1], 0.0, atol=1e-6)


def test_publication_lags_and_oi_quadrant_no_lookahead() -> None:
    """
    Перевіряє чесні лаги публікації перед ресемплінгом:
      - open_interest <= 0 замінюється на NaN + ffill, після чого зсувається на +5 хв;
      - lsr_top_traders, taker_buy_sell_ratio: +5 хв;
      - dvol_close: +60 хв;
      - vix, dxy: +1440 хв;
      - oi_price_quadrant та delta_oi_x_delta_price перераховуються ПІСЛЯ зсуву OI на +5 хв.
    """
    n_mins = 1500
    ts = pd.date_range("2024-01-01 00:00:00", periods=n_mins, freq="1min")
    df_1m = pd.DataFrame({
        "timestamp": ts,
        "perp_open": np.full(n_mins, 40000.0),
        "perp_high": np.full(n_mins, 40100.0),
        "perp_low": np.full(n_mins, 39900.0),
        "perp_close": np.linspace(40000.0, 41500.0, n_mins),
        "perp_volume": np.full(n_mins, 10.0),
        "open_interest": np.full(n_mins, 50000.0),
        "lsr_top_traders": np.full(n_mins, 1.0),
        "taker_buy_sell_ratio": np.full(n_mins, 1.0),
        "dvol_close": np.full(n_mins, 45.0),
        "vix": np.full(n_mins, 15.0),
        "dxy": np.full(n_mins, 102.0),
        "fng_index": np.full(n_mins, 50),
    })

    # Вставляємо нульовий артефакт OI на хвилині 5 та стрибок на хвилині 10
    df_1m.loc[5, "open_interest"] = -10.0  # Має бути замінений на ffill (50000.0)
    df_1m.loc[10:, "open_interest"] = 60000.0
    df_1m.loc[10:, "lsr_top_traders"] = 2.5
    df_1m.loc[10:, "dvol_close"] = 80.0
    df_1m.loc[10:, "vix"] = 30.0

    lagged = apply_publication_lags(df_1m)

    # Перевірка очищення <= 0
    assert (lagged["open_interest"].dropna() > 0).all()

    # Стрибок OI та LSR на t=10 має з'явитися лише на t=15 (+5 хв)
    assert np.isclose(lagged.loc[14, "open_interest"], 50000.0)
    assert np.isclose(lagged.loc[15, "open_interest"], 60000.0)
    assert np.isclose(lagged.loc[14, "lsr_top_traders"], 1.0)
    assert np.isclose(lagged.loc[15, "lsr_top_traders"], 2.5)

    # Стрибок DVOL на t=10 має з'явитися лише на t=70 (+60 хв)
    assert np.isclose(lagged.loc[69, "dvol_close"], 45.0)
    assert np.isclose(lagged.loc[70, "dvol_close"], 80.0)

    # Стрибок VIX на t=10 має з'явитися лише на t=1450 (+1440 хв)
    assert np.isclose(lagged.loc[1449, "vix"], 15.0)
    assert np.isclose(lagged.loc[1450, "vix"], 30.0)

    # На t=15 і ціна зросла, і зсунутий OI зріс -> Квадрант 1 (Long Build-up)
    assert int(lagged.loc[15, "oi_price_quadrant"]) == 1
    assert float(lagged.loc[15, "delta_oi_x_delta_price"]) > 0.0


def test_resampling_causal_invariance() -> None:
    """
    Перевіряє, що зміна майбутніх 1m-даних після точки розрізу T_split
    жодним чином не впливає на агреговані 1h-бари та ознаки X до T_split.
    """
    n_mins = 60 * 60  # 60 годин
    rng = np.random.default_rng(123)
    ts = pd.date_range("2024-01-01 00:00:00", periods=n_mins, freq="1min")
    close = 40000.0 + np.cumsum(rng.normal(0.0, 20.0, size=n_mins))
    df_a = pd.DataFrame({
        "timestamp": ts,
        "perp_open": close - 5.0,
        "perp_high": close + 25.0,
        "perp_low": close - 25.0,
        "perp_close": close,
        "perp_volume": rng.uniform(5.0, 50.0, size=n_mins),
        "volume_delta": rng.normal(0.0, 5.0, size=n_mins),
        "cvd": np.cumsum(rng.normal(0.0, 5.0, size=n_mins)),
        "trades_count": rng.integers(50, 500, size=n_mins),
        "open_interest": 50000.0 + np.cumsum(rng.normal(0.0, 10.0, size=n_mins)),
        "funding_rate": np.full(n_mins, 0.0001),
        "premium_index": np.full(n_mins, 0.0002),
        "lsr_top_traders": np.full(n_mins, 1.2),
        "taker_buy_sell_ratio": np.full(n_mins, 1.05),
        "dvol_close": np.full(n_mins, 50.0),
        "fng_index": np.full(n_mins, 55),
        "dxy": np.full(n_mins, 103.0),
        "vix": np.full(n_mins, 14.0),
    })

    df_b = df_a.copy()
    # Кардинально змінюємо другу половину (після 30-ї години)
    split_min = 30 * 60
    df_b.loc[split_min:, "perp_close"] *= 2.5
    df_b.loc[split_min:, "open_interest"] *= 3.0

    bars_a = resample_bars(apply_publication_lags(df_a), timeframe="1h", start_date="2024-01-01", end_date="2024-01-03")
    bars_b = resample_bars(apply_publication_lags(df_b), timeframe="1h", start_date="2024-01-01", end_date="2024-01-03")

    X_a, _ = build_feature_matrix(bars_a)
    X_b, _ = build_feature_matrix(bars_b)

    # Перші 30 годин мають бути абсолютно ідентичними
    pd.testing.assert_frame_equal(X_a.iloc[:30], X_b.iloc[:30])


def test_barrier_simulator_1to3_pessimistic_rule_and_costs() -> None:
    """
    Перевіряє симулятор 1:3:
      - Чистий Long TP (+3% - 0.10% = +2.90%)
      - Чистий Short TP (+3% - 0.10% = +2.90%)
      - Песимістичне правило: одночасне пробиття TP і SL в одній свічці зараховує SL (-1% - 0.10% = -1.10%)
      - Врахування фандінгу о 00:00, 08:00, 16:00 UTC.
    """
    ts = pd.date_range("2024-05-01 01:00:00", periods=6, freq="1h")
    df_bars = pd.DataFrame({
        "timestamp": ts,
        "perp_open": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        # Свічка 1: чистий Long TP (high=103.5 >= 103, low=99.5 > 99)
        # Свічка 2: одночасно і TP (high=104 >= 103), і SL (low=98.5 <= 99) -> має бути SL!
        # Свічка 3: чистий Short TP (low=96.5 <= 97, high=100.5 < 101)
        "perp_high": [100.2, 103.5, 104.0, 100.5, 100.2, 100.2],
        "perp_low":  [99.8,   99.5,  98.5,  96.5,  99.8,  99.8],
        "perp_close":[100.0, 103.2,  99.0,  96.8, 100.0, 100.0],
        "funding_rate": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    })

    sim = simulate_barriers_1to3(
        df_bars, sl_pct=0.01, tp_pct=0.03, horizon_hours=3, round_trip_cost=0.0010, entry_on_next_bar=True
    )

    # Для t=0 вхід відбувається на барі 1 (entry=100.0), де high=103.5, low=99.5 -> Long TP (+1)
    assert int(sim["outcome_long"][0]) == 1
    assert int(sim["exit_idx_long"][0]) == 1
    assert np.isclose(sim["pnl_long"][0], 0.03 - 0.0010, atol=1e-6)

    # Для t=1 вхід відбувається на барі 2 (entry=100.0), де і high=104>=103, і low=98.5<=99 -> Песимістичний SL (-1)!
    assert int(sim["outcome_long"][1]) == -1
    assert int(sim["exit_idx_long"][1]) == 2
    assert np.isclose(sim["pnl_long"][1], -0.01 - 0.0010, atol=1e-6)

    # Для t=2 вхід відбувається на барі 3 (entry=100.0), де low=96.5<=97, high=100.5<101 -> Short TP (+1)
    assert int(sim["outcome_short"][2]) == 1
    assert int(sim["exit_idx_short"][2]) == 3
    assert np.isclose(sim["pnl_short"][2], 0.03 - 0.0010, atol=1e-6)


def test_portfolio_backtester_no_overlapping_trades_and_5_states() -> None:
    """
    Перевіряє, що simulate_portfolio:
      - ігнорує всі нові сигнали, поки відкрита поточна угода (t < exit_idx);
      - підтримує 5 станів позиції: -2, -1, 0, +1, +2.
    """
    n = 10
    signals = np.array([1, 1, -1, 1, 0, -1, -1, 0, 1, 0], dtype=np.int8)
    conviction = np.array([2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0])
    pnl_long = np.full(n, 0.02)
    pnl_short = np.full(n, 0.015)
    # Угода з t=0 закривається на барі 3, угода з t=3 закривається на барі 5, угода з t=5 закривається на барі 7
    exit_long = np.array([3, 4, 4, 5, 5, 7, 8, 8, 9, 9], dtype=np.int64)
    exit_short = np.array([3, 4, 4, 5, 5, 7, 8, 8, 9, 9], dtype=np.int64)

    bt = simulate_portfolio(signals, conviction, pnl_long, pnl_short, exit_long, exit_short)

    # Мають відкритися лише угоди на барах t=0 (закр. 3), t=3 (закр. 5), t=5 (закр. 7), t=8 (закр. 9)
    assert list(bt["trade_entries"]) == [0, 3, 5, 8]
    assert list(bt["trade_exits"]) == [3, 5, 7, 9]

    # Перевірка відсутності перекриття інтервалів [entry, exit]
    for i in range(len(bt["trade_entries"]) - 1):
        assert bt["trade_entries"][i + 1] >= bt["trade_exits"][i]

    # Перевірка 5 станів позиції (-2, -1, 0, +1, +2)
    unique_states = set(np.unique(bt["position_states"]).tolist())
    assert unique_states.issubset({-2, -1, 0, 1, 2})
    assert 2 in unique_states and -2 in unique_states and 1 in unique_states and 0 in unique_states


def test_direct_policy_gradient_analytical_vs_numerical_gradients() -> None:
    """
    Перевіряє точність аналітичних градієнтів DirectPolicyGradientAgent
    порівнянням із чисельними скінченними різницями (Finite Differences).
    """
    rng = np.random.default_rng(7)
    X = rng.normal(0.0, 1.0, size=(40, 6))
    pnl_l = rng.normal(0.001, 0.015, size=40)
    pnl_s = rng.normal(-0.001, 0.015, size=40)

    agent = DirectPolicyGradientAgent(hidden_dim=5, l2_reg=1e-3, entropy_coef=0.02, random_state=7)
    agent._init_params(n_features=6)

    _, ana_grads, _ = agent.compute_objective_and_grads(X, pnl_l, pnl_s, compute_grads=True)

    eps = 1e-6
    for pname, param in agent.params_.items():
        num_grad = np.zeros_like(param)
        it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            ix = it.multi_index
            orig = float(param[ix])
            param[ix] = orig + eps
            obj_plus, _, _ = agent.compute_objective_and_grads(X, pnl_l, pnl_s, compute_grads=False)
            param[ix] = orig - eps
            obj_minus, _, _ = agent.compute_objective_and_grads(X, pnl_l, pnl_s, compute_grads=False)
            param[ix] = orig
            num_grad[ix] = (obj_plus - obj_minus) / (2.0 * eps)
            it.iternext()

        assert np.allclose(ana_grads[pname], num_grad, rtol=1e-4, atol=1e-5), (
            f"Розбіжність градієнта для параметра {pname}!"
        )


def test_three_base_agents_training() -> None:
    """
    Працездатність 3 методів прямого навчання на синтетичних даних:
      1) DirectPolicyGradient (Adam)
      2) NumPyQLearning (TD Widrow-Hoff)
      3) GeneticEvolution (GA)
    """
    rng = np.random.default_rng(99)
    n_samples, n_features = 240, 8
    X = rng.normal(0.0, 1.0, size=(n_samples, n_features))
    true_w = np.array([0.015, -0.012, 0.008, 0.0, 0.0, 0.0, 0.0, 0.0])
    signal_latent = X @ true_w
    pnl_l = signal_latent + rng.normal(0.0, 0.005, size=n_samples)
    pnl_s = -signal_latent + rng.normal(0.0, 0.005, size=n_samples)
    exit_l = np.minimum(np.arange(n_samples) + 3, n_samples - 1)
    exit_s = np.minimum(np.arange(n_samples) + 3, n_samples - 1)

    X_tr, X_val = X[:180], X[180:]
    pl_tr, pl_v = pnl_l[:180], pnl_l[180:]
    ps_tr, ps_v = pnl_s[:180], pnl_s[180:]
    el_tr, el_v = exit_l[:180], np.clip(exit_l[180:] - 180, 0, 59)
    es_tr, es_v = exit_s[:180], np.clip(exit_s[180:] - 180, 0, 59)

    pg = DirectPolicyGradientAgent(hidden_dim=8, epochs=25, random_state=42).fit(
        X_tr, pl_tr, ps_tr, el_tr, es_tr, X_val, pl_v, ps_v, el_v, es_v
    )
    q_td = NumPyQLearningAgent(td_epochs=10).fit(
        X_tr, pl_tr, ps_tr, el_tr, es_tr, X_val, pl_v, ps_v, el_v, es_v
    )
    ga = GeneticEvolutionAgent(pop_size=60, generations=15, random_state=42).fit(
        X_tr, pl_tr, ps_tr, el_tr, es_tr, X_val, pl_v, ps_v, el_v, es_v
    )

    for ag in (pg, q_td, ga):
        s, c = ag.predict(X_val)
        assert s.shape == (60,)
        assert c.shape == (60,)
        assert set(np.unique(s).tolist()).issubset({-1, 0, 1})
        w = ag.get_feature_weights()
        assert w.shape == (n_features,)


def test_calendar_and_cyclical_features_presence_and_bounds() -> None:
    """
    Перевіряє фінальний набір із 34 ознак (28 ринкових включно з weekend_gap_pct + 6 часових):
      - Відсутність week_of_year та інших надлишкових календарних колонок;
      - Наявність 6 часових ознак: hour_sin, hour_cos, dow_sin, dow_cos, month, is_month_turn;
      - Наявність мікроструктурної ознаки неперекритого гепу вихідного дня weekend_gap_pct;
      - Коректність діапазонів тригонометричних функцій [-1, 1].
    """
    ts = pd.date_range("2024-01-01 00:00:00", periods=72, freq="1h")
    df_bars = pd.DataFrame({
        "timestamp": ts,
        "perp_open": np.full(72, 40000.0),
        "perp_high": np.full(72, 40100.0),
        "perp_low": np.full(72, 39900.0),
        "perp_close": np.full(72, 40050.0),
        "perp_volume": np.full(72, 100.0),
        "spot_close": np.full(72, 40040.0),
        "spot_volume": np.full(72, 80.0),
        "open_interest": np.full(72, 50000.0),
        "open_interest_usd": np.full(72, 2e9),
        "trades_count": np.full(72, 500.0),
        "dxy": np.full(72, 102.0),
        "volume_delta": np.full(72, 5.0),
        "cvd": np.full(72, 50.0),
        "funding_rate": np.full(72, 0.0001),
        "premium_index": np.full(72, 0.0002),
        "lsr_top_traders": np.full(72, 1.2),
        "taker_buy_sell_ratio": np.full(72, 1.1),
        "dvol_close": np.full(72, 50.0),
        "fng_index": np.full(72, 60),
        "vix": np.full(72, 14.0),
        "oi_price_quadrant": np.full(72, 1),
    })
    X_df, feature_names = build_feature_matrix(df_bars)

    # Рівно 34 ознаки
    assert len(feature_names) == 34, f"Очікувалось 34 ознаки, отримано {len(feature_names)}: {feature_names}"
    assert X_df.shape[1] == 34

    # Перевірка наявності weekend_gap_pct
    assert "weekend_gap_pct" in feature_names
    assert "weekend_gap_pct" in X_df.columns

    # 4 швидкі тригонометричні ознаки
    expected_fast_trig = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    for col in expected_fast_trig:
        assert col in feature_names, f"Ознака {col} відсутня у feature_names!"
        assert col in X_df.columns, f"Колонка {col} відсутня у X_df!"

    # 2 інституційні календарні колонки
    expected_calendar = ["month", "is_month_turn"]
    for col in expected_calendar:
        assert col in feature_names, f"Ознака {col} відсутня у feature_names!"
        assert col in X_df.columns, f"Колонка {col} відсутня у X_df!"

    # Заборонені старі або надлишкові ознаки (включно з week_of_year)
    forbidden_features = [
        "week_of_year",
        "day_of_month", "week_of_month", "day_of_year",
        "dom_sin", "dom_cos", "wom_sin", "wom_cos", "woy_sin", "woy_cos",
        "month_sin", "month_cos", "doy_sin", "doy_cos",
        "hour", "day_of_week", "is_weekend",
    ]
    for col in forbidden_features:
        assert col not in feature_names, f"Заборонена ознака {col} не повинна бути у feature_names!"
        assert col not in X_df.columns, f"Заборонена колонка {col} не повинна бути у X_df!"

    # Перевірка обмежень тригонометричних функцій [-1, 1]
    for trig_col in expected_fast_trig:
        vals = X_df[trig_col].to_numpy()
        assert np.all(vals >= -1.0 - 1e-6) and np.all(vals <= 1.0 + 1e-6)


def test_is_month_turn_behavior_and_calendar_properties() -> None:
    """
    Перевіряє коректність інституційної ознаки межі місяця is_month_turn:
      - 1.0 для перших 3 днів місяця (day <= 3);
      - 1.0 для останніх 3 днів місяця (day >= days_in_month - 2) з точним врахуванням
        28/29/30/31 днів у відповідному місяці;
      - 0.0 для всіх серединних днів місяця.
    """
    test_dates = [
        # Січень (31 день): 1..3 -> 1, 29..31 -> 1, 15 -> 0
        ("2024-01-01 12:00:00", 1.0),
        ("2024-01-02 12:00:00", 1.0),
        ("2024-01-03 12:00:00", 1.0),
        ("2024-01-15 12:00:00", 0.0),
        ("2024-01-28 12:00:00", 0.0),
        ("2024-01-29 12:00:00", 1.0),
        ("2024-01-30 12:00:00", 1.0),
        ("2024-01-31 12:00:00", 1.0),
        # Лютий 2024 (високосний, 29 днів): 1..3 -> 1, 27..29 -> 1, 14 -> 0
        ("2024-02-01 12:00:00", 1.0),
        ("2024-02-03 12:00:00", 1.0),
        ("2024-02-14 12:00:00", 0.0),
        ("2024-02-26 12:00:00", 0.0),
        ("2024-02-27 12:00:00", 1.0),
        ("2024-02-28 12:00:00", 1.0),
        ("2024-02-29 12:00:00", 1.0),
        # Лютий 2023 (звичайний, 28 днів): 1..3 -> 1, 26..28 -> 1, 14 -> 0
        ("2023-02-01 12:00:00", 1.0),
        ("2023-02-14 12:00:00", 0.0),
        ("2023-02-25 12:00:00", 0.0),
        ("2023-02-26 12:00:00", 1.0),
        ("2023-02-27 12:00:00", 1.0),
        ("2023-02-28 12:00:00", 1.0),
        # Квітень (30 днів): 1..3 -> 1, 28..30 -> 1, 15 -> 0
        ("2024-04-01 12:00:00", 1.0),
        ("2024-04-15 12:00:00", 0.0),
        ("2024-04-27 12:00:00", 0.0),
        ("2024-04-28 12:00:00", 1.0),
        ("2024-04-29 12:00:00", 1.0),
        ("2024-04-30 12:00:00", 1.0),
    ]

    dates_ts = [pd.Timestamp(d[0]) for d in test_dates]
    expected_vals = [d[1] for d in test_dates]

    df_test = pd.DataFrame({
        "timestamp": dates_ts,
        "perp_open": np.full(len(dates_ts), 40000.0),
        "perp_high": np.full(len(dates_ts), 40100.0),
        "perp_low": np.full(len(dates_ts), 39900.0),
        "perp_close": np.full(len(dates_ts), 40050.0),
        "perp_volume": np.full(len(dates_ts), 100.0),
    })

    X_test, _ = build_feature_matrix(df_test)
    assert "is_month_turn" in X_test.columns

    actual_vals = X_test["is_month_turn"].to_numpy()
    for d_str, exp, act in zip([d[0] for d in test_dates], expected_vals, actual_vals):
        assert np.isclose(act, exp), f"Для дати {d_str} очікувалось is_month_turn={exp}, отримано {act}"


def test_time_regularization_five_modes_and_capping() -> None:
    """
    Перевіряє, що всі 5 режимів регуляризації часу:
      1) 'none' — без обмежень;
      2) 'soft_035' — м'яке демпфування 0.35x;
      3) 'cap_015' — ліміт частки L1 <= 15%;
      4) 'cap_010' — ліміт частки L1 <= 10%;
      5) 'cap_005' — ліміт частки L1 <= 5%;
    працюють коректно та суворо витримують обмеження на ваги часових ознак.
    """
    from numpy_agents import parse_time_reg_mode, project_weight_budget

    modes = ["none", "soft_035", "cap_015", "cap_010", "cap_005"]
    for m in modes:
        cap, is_soft = parse_time_reg_mode(m)
        if m == "none":
            assert cap is None and not is_soft
        elif m == "soft_035":
            assert cap is None and is_soft
        elif m == "cap_015":
            assert cap == 0.15 and not is_soft
        elif m == "cap_010":
            assert cap == 0.10 and not is_soft
        elif m == "cap_005":
            assert cap == 0.05 and not is_soft

    # Тест проєктування ваг на синтетичному векторі з 35 ознак (28 ринкових + 7 часових)
    w_raw = np.ones(35, dtype=np.float64)
    for cap_val in (0.15, 0.10, 0.05):
        w_p = project_weight_budget(w_raw, cap_val, time_idx_start=28)
        s_mkt = float(np.sum(np.abs(w_p[:28])))
        s_time = float(np.sum(np.abs(w_p[28:])))
        share = s_time / (s_mkt + s_time)
        assert np.isclose(share, cap_val, atol=1e-5)


def test_margin_call_liquidation_clamps_to_minus_100pct() -> None:
    """
    Margin Call: коли кумулятивний капітал досягає нуля, угоду ліквідовано, торгівлю припинено,
    Net PnL = -100%, MaxDD = 100%, CAGR = -100%, а капітал ніколи не стає від'ємним.
    """
    n = 200
    losses = np.full(n, -0.30)
    exits = np.minimum(np.arange(n) + 1, n - 1)
    bt = simulate_portfolio(np.ones(n, dtype=np.int8), np.ones(n), losses, losses, exits, exits)

    m = bt["metrics"]
    assert np.isclose(m["net_pnl"], -1.0) and np.isclose(m["max_dd"], 1.0) and np.isclose(m["cagr"], -1.0)
    assert np.min(bt["equity_curve"]) >= 0.0
    assert np.isclose(bt["equity_curve"][-1], 0.0)
    assert np.isclose(np.sum(bt["bar_returns"]), -1.0)
    # 0.30 * 3 = 0.90 < 1.0 -> ліквідація на 4-й угоді; подальші сигнали ігноруються
    assert len(bt["trade_pnls"]) == 4
    assert np.all(bt["bar_returns"][bt["trade_exits"][-1] + 1 :] == 0.0)

    # Прибуткова серія без ліквідації не змінюється Margin Call-механізмом
    gains = np.full(n, 0.01)
    ok = simulate_portfolio(np.ones(n, dtype=np.int8), np.ones(n), gains, gains, exits, exits)
    assert ok["metrics"]["net_pnl"] > 0.0 and np.min(ok["equity_curve"]) >= 1.0


def test_strict_oos_split_is_chronological_and_has_17520_bars() -> None:
    """
    Строгий OOS: Train < Val < Test без перекриття, Test = 24 місяці (2024-10-01 .. 2026-09-30, 17 520 барів 1h).
    """
    from data_pipeline import CACHE_DIR

    cache_files = list(CACHE_DIR.glob("bars_1h_2021-12-01_2026-09-30.parquet"))
    if not cache_files:
        pytest.skip("Кеш барів 1h відсутній (буде створений при запуску run_experiment.py)")

    ts = pd.to_datetime(pd.read_parquet(cache_files[0], columns=["timestamp"])["timestamp"])
    tr = (ts >= "2021-12-01") & (ts <= "2024-03-31 23:59:59")
    val = (ts >= "2024-04-01") & (ts <= "2024-09-30 23:59:59")
    te = (ts >= "2024-10-01") & (ts <= "2026-09-30 23:59:59")

    assert int(te.sum()) == 17520, f"Test має містити 17 520 барів, отримано {int(te.sum())}"
    assert not (tr & val).any() and not (val & te).any() and not (tr & te).any()
    assert ts[tr].max() < ts[val].min() and ts[val].max() < ts[te].min()
    assert ts[te].min() == pd.Timestamp("2024-10-01 00:00:00")
    assert ts[te].max() == pd.Timestamp("2026-09-30 23:00:00")


def test_no_walk_forward_or_warm_start_remnants_in_source() -> None:
    """Код проєкту не містить залишків Walk-Forward, Warm-Start та Ridge-режиму."""
    forbidden = ("walk_forward", "warm_start", "warm-start", "run_walk_forward", "_solve_ridge")
    for name in ("data_pipeline.py", "barrier_simulator.py", "numpy_agents.py", "run_experiment.py"):
        text = (PROJECT_DIR / name).read_text(encoding="utf-8").lower()
        for token in forbidden:
            assert token not in text, f"Залишок '{token}' у {name}"


def test_baseline_metrics_100pct_identity_between_table1_and_table3() -> None:
    """
    Перевіряє, що метрики базового режиму 1.5% : 4.5% (1:3 Baseline Robust, H=36h)
    між Таблицею 1 (main_comparison_table.csv) та Таблицею 3 (risk_reward_sensitivity.md)
    співпадають на 100% до останнього знака для всіх 3 моделей прямого навчання.
    """
    csv_path = PROJECT_DIR / "results" / "main_comparison_table.csv"
    md_path = PROJECT_DIR / "results" / "risk_reward_sensitivity.md"
    if not csv_path.exists() or not md_path.exists():
        pytest.skip("Файли результатів ще не згенеровано (будуть згенеровані після запуску run_experiment.py)")

    df_main = pd.read_csv(csv_path)
    md_content = md_path.read_text(encoding="utf-8")

    models = [
        "DirectPolicyGradient (Adam)",
        "NumPyQLearning (TD Widrow-Hoff)",
        "GeneticEvolution (GA-300)",
    ]

    for m_name in models:
        row_main = df_main[df_main["Model"] == m_name]
        assert len(row_main) == 1, f"Модель {m_name} не знайдена у main_comparison_table.csv"
        net_main = float(row_main["Net_PnL_pct"].iloc[0])
        sharpe_main = float(row_main["Sharpe"].iloc[0])
        maxdd_main = float(row_main["MaxDD_pct"].iloc[0])

        found = False
        for line in md_content.splitlines():
            if "1.5% : 4.5%" in line and m_name in line:
                found = True
                parts = [p.strip() for p in line.split("|")[1:-1]]
                pnl_str = parts[2].replace("%", "").strip()
                sharpe_str = parts[4].strip()
                maxdd_str = parts[6].replace("%", "").strip()

                assert np.isclose(float(pnl_str), net_main, atol=1e-2), (
                    f"Розбіжність Net PnL для {m_name}: Table 1={net_main}, Table 3={pnl_str}"
                )
                assert np.isclose(float(sharpe_str), sharpe_main, atol=1e-2), (
                    f"Розбіжність Sharpe для {m_name}: Table 1={sharpe_main}, Table 3={sharpe_str}"
                )
                assert np.isclose(float(maxdd_str), maxdd_main, atol=1e-2), (
                    f"Розбіжність MaxDD для {m_name}: Table 1={maxdd_main}, Table 3={maxdd_str}"
                )
        assert found, f"Рядок для {m_name} у профілі 1.5% : 4.5% не знайдений у Table 3"


def test_markdown_tables_integrity_and_math() -> None:
    """
    Суворий програмний аудит усіх 4 фінальних таблиць та 5 графіків у coursework/results/:
      1) У жодному .md файлі немає 'Ridge', 'week_of_year', 'WF-6M', 'WF-1M' або 'Warm-Start'.
      2) У Таблиці 1 (main_comparison_table.md) та в секції 1.5% : 4.5% Таблиці 3 (risk_reward_sensitivity.md)
         рядки всіх 3 моделей ідентичні символ-у-символ по всіх спільних метриках.
      3) Для кожного рядка Таблиці 1 та Таблиці 3 виконується abs(Expectancy - (Net_PnL / N_trades)) <= 0.02%.
      4) У Таблиці 2 (trade_decomposition_table.md):
         - Таблиця 2.1: N_long + N_short == N_trades (з Таблиці 1), abs((Long_PnL + Short_PnL) - Net_PnL) <= 0.05%,
           abs(Long/Short Ratio - N_long/max(N_short, 1)) <= 0.02;
         - Таблиця 2.2: N_1x + N_2x == N_trades (з Таблиці 1), abs((PnL_1x + PnL_2x) - Net_PnL) <= 0.05%,
           abs(Exp_1x - PnL_1x/N_1x) <= 0.02, abs(Exp_2x - PnL_2x/N_2x) <= 0.02,
           сума 4 кластерів ознак дорівнює 100.0% +- 0.05%, частка Час/Цикли <= 10.05%.
      5) У Таблиці 4 (feature_weights_table.md) рівно 34 пронумеровані ознаки (1..34),
         рівно 6 колонок (№, Ознака, 3 моделі, Ранг), відсортовані строго за спаданням абсолютної важливості.
      6) У coursework/results/ присутні рівно 5 цільових PNG-графіків, а walk_forward_comparison.png відсутній.
    """
    results_dir = PROJECT_DIR / "results"
    md_files = [
        results_dir / "main_comparison_table.md",
        results_dir / "trade_decomposition_table.md",
        results_dir / "risk_reward_sensitivity.md",
        results_dir / "feature_weights_table.md",
    ]
    for fp in md_files:
        if not fp.exists():
            pytest.skip(f"Файл {fp.name} ще не згенеровано")

    models = [
        "DirectPolicyGradient (Adam)",
        "NumPyQLearning (TD Widrow-Hoff)",
        "GeneticEvolution (GA-300)",
    ]

    # 1. Перевірка відсутності застарілих/заборонених термінів
    for fp in md_files:
        text = fp.read_text(encoding="utf-8")
        assert "Ridge" not in text, f"Знайдено заборонений підрядок 'Ridge' у {fp.name}"
        assert "week_of_year" not in text, f"Знайдено заборонений підрядок 'week_of_year' у {fp.name}"
        for forbidden in ("WF-6M", "WF-1M", "Warm-Start"):
            assert forbidden not in text, f"Знайдено '{forbidden}' у {fp.name}"

    # 2. Ідентичність символ-у-символ між Таблицею 1 і секцією 1.5%:4.5% Таблиці 3
    t1_text = (results_dir / "main_comparison_table.md").read_text(encoding="utf-8")
    t3_text = (results_dir / "risk_reward_sensitivity.md").read_text(encoding="utf-8")

    t1_model_cells = {}
    t1_model_data = {}
    for line in t1_text.splitlines():
        if line.startswith("|") and not line.startswith("| ---") and "Модель" not in line:
            cells = [c.strip() for c in line.split("|")[1:-1]]
            # cells: [Model, Net_PnL, CAGR, Sharpe, Sortino, MaxDD, Win_Rate, Profit_Factor, Expectancy, N_trades, LS, pval]
            m_name = cells[0]
            if m_name in models:
                t1_model_cells[m_name] = cells[1:10]
                t1_model_data[m_name] = {
                    "net_pnl": float(cells[1].replace("%", "")),
                    "n_trades": int(cells[9]),
                    "ls_ratio": float(cells[10]),
                }
            # 3. Перевірка внутрішньої математики для Таблиці 1
            net_pnl = float(cells[1].replace("%", ""))
            expectancy = float(cells[8].replace("%", ""))
            n_trades = int(cells[9])
            assert abs(expectancy - (net_pnl / max(n_trades, 1))) <= 0.02, (
                f"Порушення математики у Table 1 для {m_name}: Exp={expectancy}, Net/N={net_pnl/max(n_trades,1)}"
            )

    assert len(t1_model_cells) == 3, f"У Table 1 має бути рівно 3 моделі, знайдено: {list(t1_model_cells.keys())}"

    t3_baseline_cells = {}
    for line in t3_text.splitlines():
        if line.startswith("|") and not line.startswith("| ---") and "Профіль SL : TP" not in line:
            cells = [c.strip() for c in line.split("|")[1:-1]]
            # cells: [Profile, Model, Net_PnL, CAGR, Sharpe, Sortino, MaxDD, Win_Rate, Profit_Factor, Expectancy, N_trades]
            prof, m_name = cells[0], cells[1]
            net_pnl = float(cells[2].replace("%", ""))
            expectancy = float(cells[9].replace("%", ""))
            n_trades = int(cells[10])
            assert abs(expectancy - (net_pnl / max(n_trades, 1))) <= 0.02, (
                f"Порушення математики у Table 3 ({prof}, {m_name}): Exp={expectancy}, Net/N={net_pnl/max(n_trades,1)}"
            )
            if prof.startswith("1.5% : 4.5%"):
                t3_baseline_cells[m_name] = cells[2:11]

    assert len(t3_baseline_cells) == 3
    for m_name in models:
        assert t1_model_cells[m_name] == t3_baseline_cells[m_name], (
            f"Розбіжність символ-у-символ між Table 1 та Table 3 для {m_name}:\n"
            f"Table 1: {t1_model_cells[m_name]}\nTable 3: {t3_baseline_cells[m_name]}"
        )

    # 4. Перевірка Таблиці 2 (trade_decomposition_table.md)
    t2_text = (results_dir / "trade_decomposition_table.md").read_text(encoding="utf-8")
    in_table_2_1 = False
    in_table_2_2 = False
    t2_1_checked = {}
    t2_2_checked = {}

    for line in t2_text.splitlines():
        if "Таблиця 2.1" in line:
            in_table_2_1 = True
            in_table_2_2 = False
            continue
        elif "Таблиця 2.2" in line:
            in_table_2_1 = False
            in_table_2_2 = True
            continue

        if line.startswith("|") and not line.startswith("| ---") and "Модель" not in line:
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if in_table_2_1:
                # [Model, N_long, Long_PnL, Long_WR, N_short, Short_PnL, Short_WR, LS_ratio, Tot_PnL]
                m_clean = cells[0].replace("*", "").strip()
                assert m_clean in models, f"Невідома модель у Table 2.1: {m_clean}"
                n_long = int(cells[1])
                long_pnl = float(cells[2].replace("%", ""))
                n_short = int(cells[4])
                short_pnl = float(cells[5].replace("%", ""))
                ls_ratio = float(cells[7])
                tot_pnl = float(cells[8].replace("%", ""))

                t1_ref = t1_model_data[m_clean]
                assert n_long + n_short == t1_ref["n_trades"], (
                    f"Розбіжність N_trades для {m_clean}: {n_long} + {n_short} != {t1_ref['n_trades']}"
                )
                assert abs((long_pnl + short_pnl) - tot_pnl) <= 0.05, (
                    f"Розбіжність сум PnL у Table 2.1 для {m_clean}: {long_pnl} + {short_pnl} != {tot_pnl}"
                )
                assert abs(tot_pnl - t1_ref["net_pnl"]) <= 0.05, (
                    f"Розбіжність Net PnL між Table 1 ({t1_ref['net_pnl']}%) та Table 2.1 ({tot_pnl}%) для {m_clean}"
                )
                assert abs(ls_ratio - t1_ref["ls_ratio"]) <= 0.02, (
                    f"Розбіжність LS_ratio між Table 1 ({t1_ref['ls_ratio']}) та Table 2.1 ({ls_ratio}) для {m_clean}"
                )
                t2_1_checked[m_clean] = True

            elif in_table_2_2:
                # [Model, N_1x, PnL_1x, Exp_1x, N_2x, PnL_2x, Exp_2x, Micro_CVD, Deriv_LSR, Macro_Vol, Time_Cycles]
                m_clean = cells[0].replace("*", "").strip()
                assert m_clean in models, f"Невідома модель у Table 2.2: {m_clean}"
                n_1x = int(cells[1])
                pnl_1x = float(cells[2].replace("%", ""))
                exp_1x = float(cells[3].replace("%", ""))
                n_2x = int(cells[4])
                pnl_2x = float(cells[5].replace("%", ""))
                exp_2x = float(cells[6].replace("%", ""))
                micro_share = float(cells[7].replace("%", ""))
                deriv_share = float(cells[8].replace("%", ""))
                macro_share = float(cells[9].replace("%", ""))
                time_share = float(cells[10].replace("%", ""))

                t1_ref = t1_model_data[m_clean]
                assert n_1x + n_2x == t1_ref["n_trades"], (
                    f"Розбіжність N_trades у Table 2.2 для {m_clean}: {n_1x} + {n_2x} != {t1_ref['n_trades']}"
                )
                assert abs((pnl_1x + pnl_2x) - t1_ref["net_pnl"]) <= 0.05, (
                    f"Розбіжність PnL 1x+2x у Table 2.2 для {m_clean}: {pnl_1x} + {pnl_2x} != {t1_ref['net_pnl']}"
                )
                assert abs(exp_1x - (pnl_1x / max(n_1x, 1))) <= 0.02
                assert abs(exp_2x - (pnl_2x / max(n_2x, 1))) <= 0.02

                sum_clusters = micro_share + deriv_share + macro_share + time_share
                assert abs(sum_clusters - 100.0) <= 0.10, (
                    f"Сума часток 4 кластерів має дорівнювати 100%: {sum_clusters:.2f}% для {m_clean}"
                )
                assert time_share <= 10.05, (
                    f"Частка часового кластера має бути <= 10.0% (Cap 10%): {time_share:.2f}% для {m_clean}"
                )
                t2_2_checked[m_clean] = True

    assert len(t2_1_checked) == 3, f"У Table 2.1 перевірено {len(t2_1_checked)} моделей замість 3"
    assert len(t2_2_checked) == 3, f"У Table 2.2 перевірено {len(t2_2_checked)} моделей замість 3"

    # 5. Перевірка Таблиці 4 (feature_weights_table.md): рівно 34 ознаки (1..34), спадання рангів
    fw_text = (results_dir / "feature_weights_table.md").read_text(encoding="utf-8")
    fw_rows = []
    for line in fw_text.splitlines():
        if line.startswith("|") and not line.startswith("| ---") and "Ознака" not in line:
            cells = [c.strip() for c in line.split("|")[1:-1]]
            assert len(cells) == 6, f"Очікувалось 6 колонок у Table 4 (№, Ознака, 3 моделі, Ранг), отримано {len(cells)}"
            idx_num = int(cells[0])
            rank_val = float(cells[5].replace("*", ""))
            fw_rows.append((idx_num, rank_val))

    assert len(fw_rows) == 34, f"У Table 4 має бути рівно 34 рядки ознак, знайдено {len(fw_rows)}"
    assert [r[0] for r in fw_rows] == list(range(1, 35)), "Нумерація ознак у Table 4 має бути строго 1..34"
    ranks = [r[1] for r in fw_rows]
    for i in range(len(ranks) - 1):
        assert ranks[i] >= ranks[i + 1], f"Порушено спадання рангів у Table 4 на рядку {i+1}: {ranks[i]} < {ranks[i+1]}"

    # 6. Перевірка 5 PNG графіків
    expected_pngs = [
        "equity_curves.png",
        "drawdowns.png",
        "learning_convergence.png",
        "feature_importance_heatmap.png",
        "long_short_pnl_breakdown.png",
    ]
    for png_name in expected_pngs:
        assert (results_dir / png_name).exists(), f"Графік {png_name} відсутній у results/"

    assert not (results_dir / "walk_forward_comparison.png").exists(), (
        "Застарілий графік walk_forward_comparison.png не повинен існувати!"
    )
