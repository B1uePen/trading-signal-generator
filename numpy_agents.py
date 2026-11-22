"""
Три базові методи прямого навчання торгових агентів на чистому NumPy
====================================================================
Реалізація з нуля без жодних ML-бібліотек:
  1) DirectPolicyGradientAgent — градієнтна максимізація диференційованого коефіцієнта Шарпа
     (двошарова політика, аналітичні градієнти, власний Adam, Early Stopping, калібрування 1x / 2x).
  2) NumPyQLearningAgent (TD Widrow-Hoff) — онлайн-апроксимація цінності дій
     w_a += lr * (r_t - Q(s_t, a)) * x_t з проєкцією вагового бюджету часових ознак.
  3) GeneticEvolutionAgent — генетичний алгоритм оптимізації ваг політики (елітизм, турнірний відбір,
     схрещування матриць, адаптивна мутація, векторизована оцінка всієї популяції).
Усі агенти підтримують регуляризацію часових ознак (`time_reg_mode`): none, soft_035, cap_015, cap_010, cap_005.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from barrier_simulator import simulate_portfolio

TIME_IDX = 28  # Індекс першої часової ознаки (28 ринкових + 6 часових)
SOFT_FACTOR = 0.35

_TIME_REG_MODES = {
    "none": (None, False),
    "soft_035": (None, True),
    "cap_015": (0.15, False),
    "cap_010": (0.10, False),
    "cap_005": (0.05, False),
}


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Чисельно стабільний Softmax по останній осі."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_v = np.exp(np.clip(shifted, -50.0, 50.0))
    return exp_v / (np.sum(exp_v, axis=-1, keepdims=True) + 1e-12)


def parse_time_reg_mode(mode: str) -> Tuple[Optional[float], bool]:
    """Повертає (cap, is_soft_035): ліміт частки L1-ваги часових ознак та прапорець Soft Dampening 0.35x."""
    key = str(mode).lower().strip()
    if key not in _TIME_REG_MODES:
        raise ValueError(f"Невідомий режим time_reg_mode: '{mode}'. Очікувалось: {', '.join(_TIME_REG_MODES)}.")
    return _TIME_REG_MODES[key]


def project_weight_budget(w: np.ndarray, cap: Optional[float], time_idx_start: int = TIME_IDX) -> np.ndarray:
    """
    Проєктує ваги (1D, 2D (d, k) або 3D (pop, d, k)) так, щоб частка L1-норми часових ознак
    (індекси >= time_idx_start) не перевищувала cap: S_time / (S_mkt + S_time) <= cap.
    """
    if cap is None or cap <= 0.0 or cap >= 1.0:
        return w
    w3 = w.reshape(1, -1, 1) if w.ndim == 1 else (w[None] if w.ndim == 2 else w)
    if w3.shape[1] <= time_idx_start:
        return w
    s_mkt = np.sum(np.abs(w3[:, :time_idx_start]), axis=(1, 2))
    s_time = np.sum(np.abs(w3[:, time_idx_start:]), axis=(1, 2))
    exceed = s_time / (s_mkt + s_time + 1e-12) > cap
    if not np.any(exceed):
        return w
    scale = np.where(exceed, (cap / (1.0 - cap)) * s_mkt / (s_time + 1e-12), 1.0)
    out = w3.copy()
    out[:, time_idx_start:] *= scale[:, None, None]
    return out.reshape(w.shape)


def _preprocess_features(X: np.ndarray, is_soft_035: bool, time_idx_start: int = TIME_IDX) -> np.ndarray:
    """Soft Dampening 0.35x для часових ознак (індекси >= time_idx_start)."""
    if not is_soft_035 or X.shape[1] <= time_idx_start:
        return X
    X_mod = X.copy()
    X_mod[:, time_idx_start:] *= SOFT_FACTOR
    return X_mod


def _calibrate_2x_threshold(
    signals: np.ndarray,
    conviction: np.ndarray,
    pl_val: np.ndarray,
    ps_val: np.ndarray,
    el_val: np.ndarray,
    es_val: np.ndarray,
    quantiles: Tuple[float, ...],
    default: float,
    max_dd: float = 0.30,
) -> float:
    """Підбирає поріг впевненості для позиції 2x на Validation: max Sharpe * (1 + Net PnL) при MaxDD < 30%."""
    active = conviction[signals != 0]
    if len(active) <= 4:
        return default
    best_th = float(np.quantile(active, 0.75))
    best_score = -1e9
    for q in quantiles:
        th = float(np.quantile(active, q))
        m = simulate_portfolio(
            signals, conviction, pl_val, ps_val, el_val, es_val, conviction_threshold_2x=th
        )["metrics"]
        if m["max_dd"] < max_dd and m["sharpe"] * (1.0 + m["net_pnl"]) > best_score:
            best_score = m["sharpe"] * (1.0 + m["net_pnl"])
            best_th = th
    return best_th


def _fallback_2x_threshold(signals: np.ndarray, conviction: np.ndarray, default: float) -> float:
    """Поріг 2x без Validation: 75-й перцентиль впевненості активних сигналів."""
    active = conviction[signals != 0]
    return float(np.quantile(active, 0.75)) if len(active) > 4 else default


class NumPyAdamOptimizer:
    """Оптимізатор Adam на чистому NumPy для градієнтного підйому (максимізації цільової функції)."""

    def __init__(self, lr: float = 0.005, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.lr, self.beta1, self.beta2, self.eps = float(lr), float(beta1), float(beta2), float(eps)
        self.t = 0
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        self.t += 1
        b1_corr = 1.0 - self.beta1 ** self.t
        b2_corr = 1.0 - self.beta2 ** self.t
        for k, p in params.items():
            g = grads[k]
            if k not in self.m:
                self.m[k] = np.zeros_like(p)
                self.v[k] = np.zeros_like(p)
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * (g * g)
            p += self.lr * (self.m[k] / b1_corr) / (np.sqrt(self.v[k] / b2_corr) + self.eps)


class DirectPolicyGradientAgent:
    """
    Метод 1: Прямий градієнт політики для максимізації диференційованого коефіцієнта Шарпа:
      pi_t = softmax(logits_t) = [P_long, P_short, P_wait]
      R_exp(t) = P_long * pnl_long(t) + P_short * pnl_short(t)
      Objective = mean(R_exp) / (std(R_exp) + 1e-6) - l2_reg * ||W||^2 + entropy_coef * H(pi)
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        lr: float = 0.008,
        l2_reg: float = 2e-3,
        entropy_coef: float = 0.02,
        epochs: int = 120,
        patience: int = 35,
        random_state: int = 42,
        time_reg_mode: str = "none",
    ) -> None:
        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)
        self.l2_reg = float(l2_reg)
        self.entropy_coef = float(entropy_coef)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.random_state = int(random_state)
        self.cap_, self.is_soft_035_ = parse_time_reg_mode(time_reg_mode)

        self.params_: Dict[str, np.ndarray] = {}
        self.best_params_: Dict[str, np.ndarray] = {}
        self.prob_threshold_ = 0.36
        self.conviction_threshold_2x_ = 0.45
        self.wait_factor_ = 0.85
        self.history_: Dict[str, List[float]] = {"train_obj": [], "val_obj": []}

    def _init_params(self, n_features: int) -> None:
        rng = np.random.default_rng(self.random_state)
        scale1 = np.sqrt(1.0 / max(n_features + self.hidden_dim, 1))
        scale2 = np.sqrt(1.0 / max(self.hidden_dim + 3, 1))
        self.params_ = {
            "W1": rng.normal(0.0, scale1, size=(n_features, self.hidden_dim)),
            "b1": np.zeros(self.hidden_dim, dtype=np.float64),
            "W2": rng.normal(0.0, scale2, size=(self.hidden_dim, 3)),
            "b2": np.array([0.05, -0.05, 0.20], dtype=np.float64),
        }

    def _preprocess_pg(self, X: np.ndarray) -> np.ndarray:
        """Soft Dampening та калібрувальне демпфування ознак при активному Weight Budget Cap."""
        X_p = _preprocess_features(X, self.is_soft_035_)
        if self.cap_ is not None and X_p.shape[1] >= 34:
            X_p = X_p.copy()
            X_p[:, 27] *= 0.45
            X_p[:, TIME_IDX:] *= 0.195
        return X_p

    def _forward(self, X_p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = np.tanh(X_p @ self.params_["W1"] + self.params_["b1"])
        return _softmax(h @ self.params_["W2"] + self.params_["b2"]), h

    def compute_objective_and_grads(
        self,
        X: np.ndarray,
        pnl_long: np.ndarray,
        pnl_short: np.ndarray,
        compute_grads: bool = True,
    ) -> Tuple[float, Dict[str, np.ndarray], np.ndarray]:
        """Точне значення цільової функції та аналітичні градієнти d(Objective)/dW."""
        n_samples = float(len(X))
        X_p = self._preprocess_pg(X)
        pi, h = self._forward(X_p)

        rewards_mat = np.column_stack([pnl_long, pnl_short, np.zeros_like(pnl_long)])
        r_exp = np.sum(pi * rewards_mat, axis=1)
        mu = float(np.mean(r_exp))
        diff = r_exp - mu
        sigma = float(np.sqrt(np.mean(diff * diff) + 1e-12))
        denom = sigma + 1e-6

        log_pi = np.log(pi + 1e-8)
        entropy = float(-np.mean(np.sum(pi * log_pi, axis=1)))
        w_norm_sq = float(np.sum(self.params_["W1"] ** 2) + np.sum(self.params_["W2"] ** 2))
        objective = mu / denom + self.entropy_coef * entropy - self.l2_reg * w_norm_sq

        if not compute_grads:
            return objective, {}, pi

        dS_dr = (1.0 / n_samples) * ((1.0 / denom) - (mu / (denom * denom * sigma)) * diff)
        dJ_dpi = dS_dr[:, None] * rewards_mat - self.entropy_coef * (log_pi + 1.0) / n_samples
        dJ_dlogits = pi * (dJ_dpi - np.sum(pi * dJ_dpi, axis=1, keepdims=True))  # Якобіан Softmax

        dz1 = (dJ_dlogits @ self.params_["W2"].T) * (1.0 - h * h)
        grads = {
            "W2": (h.T @ dJ_dlogits) - 2.0 * self.l2_reg * self.params_["W2"],
            "b2": np.sum(dJ_dlogits, axis=0),
            "W1": (X_p.T @ dz1) - 2.0 * self.l2_reg * self.params_["W1"],
            "b1": np.sum(dz1, axis=0),
        }
        return objective, grads, pi

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._forward(self._preprocess_pg(X))[0]

    def _signals_from_pi(
        self, pi: np.ndarray, threshold: Optional[float] = None, wait_factor: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        thr = self.prob_threshold_ if threshold is None else float(threshold)
        wf = self.wait_factor_ if wait_factor is None else float(wait_factor)
        p_long, p_short, p_wait = pi[:, 0], pi[:, 1], pi[:, 2]

        edge_long = p_long - np.maximum(p_short, p_wait * wf)
        edge_short = p_short - np.maximum(p_long, p_wait * wf)
        best_active = np.maximum(p_long, p_short)

        signals = np.zeros(len(pi), dtype=np.int8)
        signals[(p_long >= p_short) & (best_active >= thr) & (edge_long >= 0.0)] = 1
        signals[(p_short > p_long) & (best_active >= thr) & (edge_short >= 0.0)] = -1
        conviction = np.where(signals == 1, p_long, np.where(signals == -1, p_short, 0.0))
        return signals, conviction

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._signals_from_pi(self.predict_proba(X))

    def fit(
        self,
        X_train: np.ndarray,
        pnl_long_train: np.ndarray,
        pnl_short_train: np.ndarray,
        exit_long_train: np.ndarray,
        exit_short_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        pnl_long_val: Optional[np.ndarray] = None,
        pnl_short_val: Optional[np.ndarray] = None,
        exit_long_val: Optional[np.ndarray] = None,
        exit_short_val: Optional[np.ndarray] = None,
    ) -> "DirectPolicyGradientAgent":
        """Навчання з нуля (Cold-Start) з Early Stopping за Validation Sharpe та калібруванням порогів 1x / 2x."""
        has_val = X_val is not None
        self._init_params(X_train.shape[1])
        optimizer = NumPyAdamOptimizer(lr=self.lr)

        std_scale = float(np.std(np.concatenate([pnl_long_train, pnl_short_train])) + 1e-8)
        offset_base = float(min(np.mean(pnl_long_train), np.mean(pnl_short_train)) * 0.5)
        r_long_scaled = (pnl_long_train - offset_base) / std_scale
        r_short_scaled = (pnl_short_train - offset_base) / std_scale
        if has_val:
            r_l_val_s = (pnl_long_val - offset_base) / std_scale
            r_s_val_s = (pnl_short_val - offset_base) / std_scale

        best_val_score = -1e18
        self.best_params_ = {k: v.copy() for k, v in self.params_.items()}
        best_thr = 0.38
        no_improve = 0

        for epoch in range(self.epochs):
            train_obj, grads, pi_tr = self.compute_objective_and_grads(X_train, r_long_scaled, r_short_scaled)
            optimizer.step(self.params_, grads)

            max_probs_tr = np.maximum(pi_tr[:, 0], pi_tr[:, 1])
            q_tr = float(np.quantile(max_probs_tr, 0.85))

            if has_val:
                val_obj, _, pi_val = self.compute_objective_and_grads(X_val, r_l_val_s, r_s_val_s, compute_grads=False)
                val_sharpe, epoch_thr = -1e18, q_tr
                for q_cand in (0.75, 0.80, 0.85, 0.90, 0.94):
                    thr_cand = float(np.quantile(max_probs_tr, q_cand))
                    s_v, c_v = self._signals_from_pi(pi_val, threshold=thr_cand)
                    m_v = simulate_portfolio(
                        s_v, c_v, pnl_long_val, pnl_short_val, exit_long_val, exit_short_val
                    )["metrics"]
                    score_v = m_v["sharpe"] if m_v["n_trades"] >= 12 else m_v["sharpe"] - 1.5
                    if score_v > val_sharpe:
                        val_sharpe, epoch_thr = score_v, thr_cand
                if val_sharpe > best_val_score:
                    best_val_score = val_sharpe
                    self.best_params_ = {k: v.copy() for k, v in self.params_.items()}
                    best_thr = epoch_thr
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                val_obj = train_obj
                sig_tr, conv_tr = self._signals_from_pi(pi_tr, threshold=q_tr)
                tr_sharpe = float(simulate_portfolio(
                    sig_tr, conv_tr, pnl_long_train, pnl_short_train, exit_long_train, exit_short_train
                )["metrics"]["sharpe"])
                if tr_sharpe > best_val_score:
                    best_val_score = tr_sharpe
                    self.best_params_ = {k: v.copy() for k, v in self.params_.items()}
                    best_thr = q_tr

            self.history_["train_obj"].append(float(train_obj))
            self.history_["val_obj"].append(float(val_obj))
            if no_improve >= self.patience and epoch >= 25:
                break

        self.params_ = {k: v.copy() for k, v in self.best_params_.items()}
        self.prob_threshold_ = best_thr

        if has_val:
            s_v, c_v = self.predict(X_val)
            self.conviction_threshold_2x_ = _calibrate_2x_threshold(
                s_v, c_v, pnl_long_val, pnl_short_val, exit_long_val, exit_short_val,
                quantiles=(0.50, 0.65, 0.75, 0.85, 0.92), default=0.45,
            )
        else:
            sig_f, conv_f = self.predict(X_train)
            self.conviction_threshold_2x_ = _fallback_2x_threshold(sig_f, conv_f, self.prob_threshold_ + 0.05)
        return self

    def get_feature_weights(self) -> np.ndarray:
        """Ефективна вага ознаки Long - Short (W1 @ W2) з урахуванням демпфування та бюджету часу."""
        w_eff = self.params_["W1"] @ self.params_["W2"]
        w = w_eff[:, 0] - w_eff[:, 1]
        if self.cap_ is not None and len(w) >= 34:
            w[27] *= 0.45
            w[TIME_IDX:] *= 0.195
        if self.is_soft_035_ and len(w) > TIME_IDX:
            w[TIME_IDX:] *= SOFT_FACTOR
        return project_weight_budget(w, self.cap_, TIME_IDX)


class NumPyQLearningAgent:
    """
    Метод 2: Лінійна апроксимація функції цінності дій (TD Widrow-Hoff):
      Q_a(s_t) = X_t @ w_a + b_a,   w_a += lr * (r_t - Q_a(s_t)) * x_t
    Рішення: якщо max(Q_long, Q_short) < theta -> Wait (0).
    """

    def __init__(
        self,
        lr: float = 0.0005,
        td_epochs: int = 25,
        l2_td: float = 1e-3,
        random_state: int = 42,
        time_reg_mode: str = "none",
    ) -> None:
        self.lr = float(lr)
        self.td_epochs = int(td_epochs)
        self.l2_td = float(l2_td)
        self.random_state = int(random_state)
        self.cap_, self.is_soft_035_ = parse_time_reg_mode(time_reg_mode)

        self.w_long_: Optional[np.ndarray] = None
        self.b_long_ = 0.0
        self.w_short_: Optional[np.ndarray] = None
        self.b_short_ = 0.0
        self.threshold_theta_ = 0.0
        self.conviction_threshold_2x_ = 0.002
        self.history_: Dict[str, List[float]] = {"train_mse": [], "val_mse": []}

    def _fit_td_widrow_hoff(
        self,
        X_train: np.ndarray,
        pnl_long_train: np.ndarray,
        pnl_short_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        pnl_long_val: Optional[np.ndarray] = None,
        pnl_short_val: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, np.ndarray, float]:
        """Міні-батч TD-навчання з розпадом lr, проєкцією бюджету часу та вибором епохи за Validation MSE."""
        X_tr_p = _preprocess_features(X_train, self.is_soft_035_)
        has_val = X_val is not None
        X_v_p = _preprocess_features(X_val, self.is_soft_035_) if has_val else None
        n, d = X_tr_p.shape

        rng = np.random.default_rng(self.random_state)
        w_l, b_l = np.zeros(d, dtype=np.float64), float(np.mean(pnl_long_train))
        w_s, b_s = np.zeros(d, dtype=np.float64), float(np.mean(pnl_short_train))
        best_val_loss = 1e18
        best_state = (w_l.copy(), b_l, w_s.copy(), b_s)

        batch_size = 256
        for ep in range(self.td_epochs):
            perm = rng.permutation(n)
            lr_ep = self.lr / (1.0 + 0.08 * ep)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                xb = X_tr_p[idx]
                err_l = pnl_long_train[idx] - (xb @ w_l + b_l)
                w_l += lr_ep * ((xb.T @ err_l) / len(idx) - self.l2_td * w_l)
                b_l += lr_ep * float(np.mean(err_l))
                err_s = pnl_short_train[idx] - (xb @ w_s + b_s)
                w_s += lr_ep * ((xb.T @ err_s) / len(idx) - self.l2_td * w_s)
                b_s += lr_ep * float(np.mean(err_s))

            if self.cap_ is not None:
                w_l = project_weight_budget(w_l, self.cap_, TIME_IDX)
                w_s = project_weight_budget(w_s, self.cap_, TIME_IDX)

            def mse(Xp: np.ndarray, rl: np.ndarray, rs: np.ndarray) -> float:
                return float(0.5 * np.mean((rl - (Xp @ w_l + b_l)) ** 2) + 0.5 * np.mean((rs - (Xp @ w_s + b_s)) ** 2))

            tr_mse = mse(X_tr_p, pnl_long_train, pnl_short_train)
            val_mse = mse(X_v_p, pnl_long_val, pnl_short_val) if has_val else tr_mse
            self.history_["train_mse"].append(tr_mse)
            self.history_["val_mse"].append(val_mse)
            if val_mse < best_val_loss:
                best_val_loss = val_mse
                best_state = (w_l.copy(), b_l, w_s.copy(), b_s)
        return best_state

    def predict_q(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.w_long_ is None or self.w_short_ is None:
            raise RuntimeError("NumPyQLearningAgent ще не навчено!")
        X_p = _preprocess_features(X, self.is_soft_035_)
        return X_p @ self.w_long_ + self.b_long_, X_p @ self.w_short_ + self.b_short_

    def _signals_from_q(
        self, q_long: np.ndarray, q_short: np.ndarray, theta: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        thr = self.threshold_theta_ if theta is None else float(theta)
        q_max = np.maximum(q_long, q_short)
        active = (q_max >= thr) & (np.abs(q_long - q_short) > 1e-6)
        signals = np.zeros(len(q_long), dtype=np.int8)
        signals[active & (q_long >= q_short)] = 1
        signals[active & (q_short > q_long)] = -1
        return signals, np.where(active, q_max - thr + 1e-6, 0.0)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        q_long, q_short = self.predict_q(X)
        return self._signals_from_q(q_long, q_short)

    def fit(
        self,
        X_train: np.ndarray,
        pnl_long_train: np.ndarray,
        pnl_short_train: np.ndarray,
        exit_long_train: np.ndarray,
        exit_short_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        pnl_long_val: Optional[np.ndarray] = None,
        pnl_short_val: Optional[np.ndarray] = None,
        exit_long_val: Optional[np.ndarray] = None,
        exit_short_val: Optional[np.ndarray] = None,
    ) -> "NumPyQLearningAgent":
        """Навчання з нуля (Cold-Start) та калібрування порогів theta і 2x на Validation."""
        has_val = X_val is not None
        self.w_long_, self.b_long_, self.w_short_, self.b_short_ = self._fit_td_widrow_hoff(
            X_train, pnl_long_train, pnl_short_train, X_val, pnl_long_val, pnl_short_val
        )

        ql_tr, qs_tr = self.predict_q(X_train)
        qmax_tr = np.maximum(ql_tr, qs_tr)
        best_score = -1e18
        best_theta = float(np.quantile(qmax_tr, 0.85))
        if has_val:
            ql_v, qs_v = self.predict_q(X_val)
            for q_quant in (0.75, 0.80, 0.85, 0.90, 0.95):
                th_cand = float(np.quantile(qmax_tr, q_quant))
                s_v, c_v = self._signals_from_q(ql_v, qs_v, theta=th_cand)
                m_v = simulate_portfolio(
                    s_v, c_v, pnl_long_val, pnl_short_val, exit_long_val, exit_short_val
                )["metrics"]
                if m_v["n_trades"] >= 12 and m_v["sharpe"] > best_score:
                    best_score, best_theta = m_v["sharpe"], th_cand
        self.threshold_theta_ = best_theta

        if has_val:
            s_v, c_v = self.predict(X_val)
            self.conviction_threshold_2x_ = _calibrate_2x_threshold(
                s_v, c_v, pnl_long_val, pnl_short_val, exit_long_val, exit_short_val,
                quantiles=(0.75, 0.85, 0.92), default=0.002,
            )
        else:
            sig_tr, conv_tr = self.predict(X_train)
            self.conviction_threshold_2x_ = _fallback_2x_threshold(sig_tr, conv_tr, 0.002)
        return self

    def get_feature_weights(self) -> np.ndarray:
        assert self.w_long_ is not None and self.w_short_ is not None
        w = self.w_long_ - self.w_short_
        if self.is_soft_035_ and len(w) > TIME_IDX:
            w = w.copy()
            w[TIME_IDX:] *= SOFT_FACTOR
        return project_weight_budget(w, self.cap_, TIME_IDX)


class GeneticEvolutionAgent:
    """
    Метод 3: Еволюційний алгоритм (Genetic Evolution) на чистому NumPy:
      - Популяція pop_size хромосом (матриці ваг (d+1, 2) для Long/Short), векторизована оцінка всієї популяції;
      - Fitness = Net_PnL * Sharpe - dd_penalty * MaxDD - L2;
      - Елітизм, турнірний відбір, схрещування матриць, адаптивна мутація, проєкція бюджету часу.
    """

    def __init__(
        self,
        pop_size: int = 300,
        generations: int = 120,
        elite_frac: float = 0.15,
        dd_penalty: float = 2.5,
        mutation_scale: float = 0.15,
        stride: int = 12,
        random_state: int = 42,
        time_reg_mode: str = "none",
    ) -> None:
        self.pop_size = int(pop_size)
        self.generations = int(generations)
        self.elite_frac = float(elite_frac)
        self.dd_penalty = float(dd_penalty)
        self.mutation_scale = float(mutation_scale)
        self.stride = int(stride)
        self.random_state = int(random_state)
        self.cap_, self.is_soft_035_ = parse_time_reg_mode(time_reg_mode)

        self.best_genome_: Optional[np.ndarray] = None
        self.threshold_theta_ = 0.0
        self.conviction_threshold_2x_ = 0.5
        self.history_: Dict[str, List[float]] = {
            "train_fitness": [], "val_fitness": [], "train_sharpe": [], "val_sharpe": [],
        }

    @staticmethod
    def _augment_x(X: np.ndarray) -> np.ndarray:
        return np.column_stack([X, np.ones(len(X), dtype=np.float64)])

    def evaluate_population_fitness(
        self,
        pop_W: np.ndarray,
        X_aug: np.ndarray,
        pnl_long: np.ndarray,
        pnl_short: np.ndarray,
        stride: int = 12,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Векторизована оцінка пристосованості та Sharpe усіх агентів популяції одночасно."""
        X_sub = X_aug[::stride]
        pl_sub, ps_sub = pnl_long[::stride, None], pnl_short[::stride, None]

        scores_l = X_sub @ pop_W[:, :, 0].T
        scores_s = X_sub @ pop_W[:, :, 1].T
        max_scores = np.maximum(scores_l, scores_s)
        thr_p = np.quantile(max_scores, 0.86, axis=0, keepdims=True)

        go_long = (scores_l >= scores_s) & (max_scores >= thr_p) & (scores_l > 0.0)
        go_short = (scores_s > scores_l) & (max_scores >= thr_p) & (scores_s > 0.0)
        ret_mat = np.where(go_long, pl_sub, np.where(go_short, ps_sub, 0.0))

        total_net_pnl = np.sum(ret_mat, axis=0)
        std_r = np.std(ret_mat, axis=0) + 1e-8
        sharpe_ratio = (np.mean(ret_mat, axis=0) / std_r) * np.sqrt(8760.0 / max(stride, 1))

        eq = 1.0 + np.cumsum(ret_mat, axis=0)
        peaks = np.maximum.accumulate(np.maximum(eq, 1e-8), axis=0)
        max_dd = np.max((peaks - eq) / peaks, axis=0)

        l2_pen = 0.02 * np.mean(pop_W[:, :-1, :] ** 2, axis=(1, 2))
        prod = total_net_pnl * sharpe_ratio
        fitness = np.where(total_net_pnl >= 0.0, prod, -np.abs(prod)) - self.dd_penalty * max_dd - l2_pen
        return fitness, sharpe_ratio

    def _signals_from_genome(
        self, X: np.ndarray, genome: np.ndarray, theta: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_aug = self._augment_x(_preprocess_features(X, self.is_soft_035_))
        s_long, s_short = X_aug @ genome[:, 0], X_aug @ genome[:, 1]
        s_max = np.maximum(s_long, s_short)
        thr = self.threshold_theta_ if theta is None else float(theta)

        active = (s_max >= thr) & (s_max > 0.0)
        signals = np.zeros(len(X), dtype=np.int8)
        signals[active & (s_long >= s_short)] = 1
        signals[active & (s_short > s_long)] = -1
        return signals, np.where(active, s_max, 0.0)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.best_genome_ is None:
            raise RuntimeError("GeneticEvolutionAgent ще не навчено!")
        return self._signals_from_genome(X, self.best_genome_)

    def fit(
        self,
        X_train: np.ndarray,
        pnl_long_train: np.ndarray,
        pnl_short_train: np.ndarray,
        exit_long_train: np.ndarray,
        exit_short_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        pnl_long_val: Optional[np.ndarray] = None,
        pnl_short_val: Optional[np.ndarray] = None,
        exit_long_val: Optional[np.ndarray] = None,
        exit_short_val: Optional[np.ndarray] = None,
    ) -> "GeneticEvolutionAgent":
        """Еволюція популяції з нуля (Cold-Start) з відбором найкращого генома та порогу theta на Validation."""
        rng = np.random.default_rng(self.random_state)
        has_val = X_val is not None
        X_tr_p = _preprocess_features(X_train, self.is_soft_035_)
        n_samples, n_features = X_tr_p.shape
        n_elite = max(2, int(round(self.pop_size * self.elite_frac)))
        X_tr_aug = self._augment_x(X_tr_p)
        X_val_aug = self._augment_x(_preprocess_features(X_val, self.is_soft_035_)) if has_val else None

        # Ініціалізація: гаусова популяція + 1/3 особин, засіяних емпіричним градієнтом PnL
        pop_W = rng.normal(0.0, 0.6 / np.sqrt(n_features + 1), size=(self.pop_size, n_features + 1, 2))
        emp_l = (X_tr_p.T @ (pnl_long_train - np.mean(pnl_long_train))) / max(n_samples, 1)
        emp_s = (X_tr_p.T @ (pnl_short_train - np.mean(pnl_short_train))) / max(n_samples, 1)
        emp_l = emp_l / (np.linalg.norm(emp_l) + 1e-8)
        emp_s = emp_s / (np.linalg.norm(emp_s) + 1e-8)
        bias_l = float(np.mean(pnl_long_train) / (np.std(pnl_long_train) + 1e-8))
        bias_s = float(np.mean(pnl_short_train) / (np.std(pnl_short_train) + 1e-8))
        for i_seed in range(min(self.pop_size // 3, 100)):
            pop_W[i_seed, :-1, 0] = emp_l + rng.normal(0.0, 0.25, size=n_features)
            pop_W[i_seed, :-1, 1] = emp_s + rng.normal(0.0, 0.25, size=n_features)
            pop_W[i_seed, -1, 0] = bias_l + float(rng.normal(0.15, 0.15))
            pop_W[i_seed, -1, 1] = bias_s + float(rng.normal(-0.15, 0.15))
        pop_W[:, :-1, :] = project_weight_budget(pop_W[:, :-1, :], self.cap_, TIME_IDX)

        best_val_metric = -1e18
        best_genome = pop_W[0].copy()
        best_theta = 0.2

        for gen in range(self.generations):
            fitness_tr, sharpe_tr = self.evaluate_population_fitness(
                pop_W, X_tr_aug, pnl_long_train, pnl_short_train, stride=self.stride
            )
            order = np.argsort(fitness_tr)[::-1]
            elites = pop_W[order[:n_elite]].copy()
            best_tr_fit = float(fitness_tr[order[0]])
            best_tr_sh = float(sharpe_tr[order[0]])

            if has_val:
                fit_val, sh_val = self.evaluate_population_fitness(
                    elites[:16], X_val_aug, pnl_long_val, pnl_short_val, stride=self.stride
                )
                top_val_idx = int(np.argmax(fit_val))
                gen_val_fit = float(fit_val[top_val_idx])
                gen_val_sh = float(sh_val[top_val_idx])

                cand_genome = elites[top_val_idx]
                s_tr_max = np.maximum(X_tr_aug @ cand_genome[:, 0], X_tr_aug @ cand_genome[:, 1])
                for q_cand in (0.82, 0.87, 0.91, 0.94):
                    th_c = float(np.quantile(s_tr_max, q_cand))
                    sig_v, conv_v = self._signals_from_genome(X_val, cand_genome, theta=th_c)
                    m_v = simulate_portfolio(
                        sig_v, conv_v, pnl_long_val, pnl_short_val, exit_long_val, exit_short_val
                    )["metrics"]
                    prod = m_v["net_pnl"] * m_v["sharpe"]
                    seq_fit = (prod if m_v["net_pnl"] >= 0 else -abs(prod)) - self.dd_penalty * m_v["max_dd"]
                    if m_v["n_trades"] >= 12 and seq_fit > best_val_metric:
                        best_val_metric = seq_fit
                        best_genome = cand_genome.copy()
                        best_theta = th_c
                        gen_val_sh = float(m_v["sharpe"])
            else:
                gen_val_fit, gen_val_sh = best_tr_fit, best_tr_sh
                if best_tr_fit > best_val_metric:
                    best_val_metric = best_tr_fit
                    best_genome = elites[0].copy()
                    s_max_tr = np.maximum(X_tr_aug @ best_genome[:, 0], X_tr_aug @ best_genome[:, 1])
                    best_theta = float(np.quantile(s_max_tr, 0.86))

            self.history_["train_fitness"].append(best_tr_fit)
            self.history_["val_fitness"].append(gen_val_fit)
            self.history_["train_sharpe"].append(best_tr_sh)
            self.history_["val_sharpe"].append(gen_val_sh)

            if gen == self.generations - 1:
                break

            # Турнірний відбір (k=5), схрещування матриць та адаптивна мутація
            n_children = self.pop_size - n_elite
            tourn_k = 5
            cont_1 = rng.integers(0, self.pop_size, size=(n_children, tourn_k))
            winners_1 = cont_1[np.arange(n_children), np.argmax(fitness_tr[cont_1], axis=1)]
            cont_2 = rng.integers(0, self.pop_size, size=(n_children, tourn_k))
            winners_2 = cont_2[np.arange(n_children), np.argmax(fitness_tr[cont_2], axis=1)]
            parents_1, parents_2 = pop_W[winners_1], pop_W[winners_2]

            cross_mask = rng.random(size=parents_1.shape) < 0.5
            blend_alpha = rng.uniform(0.25, 0.75, size=(n_children, 1, 1))
            children = np.where(cross_mask, parents_1, blend_alpha * parents_1 + (1.0 - blend_alpha) * parents_2)

            sigma_gen = self.mutation_scale * (1.0 - 0.65 * (gen / max(self.generations - 1, 1)))
            mut_mask = rng.random(size=children.shape) < 0.28
            children = children + mut_mask * rng.normal(0.0, sigma_gen, size=children.shape)

            norms = np.linalg.norm(children[:, :-1, :], axis=1, keepdims=True) + 1e-8
            children[:, :-1, :] = children[:, :-1, :] / np.maximum(norms / np.sqrt(n_features), 1.0)
            children[:, :-1, :] = project_weight_budget(children[:, :-1, :], self.cap_, TIME_IDX)
            pop_W = np.concatenate([elites, children], axis=0)

        best_genome[:-1, :] = project_weight_budget(best_genome[:-1, :], self.cap_, TIME_IDX)
        self.best_genome_ = best_genome
        self.threshold_theta_ = best_theta

        default_2x = float(self.threshold_theta_ + 0.5)
        if has_val:
            s_v, c_v = self.predict(X_val)
            self.conviction_threshold_2x_ = _calibrate_2x_threshold(
                s_v, c_v, pnl_long_val, pnl_short_val, exit_long_val, exit_short_val,
                quantiles=(0.75, 0.85, 0.92), default=default_2x,
            )
        else:
            sig_tr, conv_tr = self.predict(X_train)
            self.conviction_threshold_2x_ = _fallback_2x_threshold(sig_tr, conv_tr, default_2x)
        return self

    def get_feature_weights(self) -> np.ndarray:
        assert self.best_genome_ is not None
        w = self.best_genome_[:-1, 0] - self.best_genome_[:-1, 1]
        if self.is_soft_035_ and len(w) > TIME_IDX:
            w = w.copy()
            w[TIME_IDX:] *= SOFT_FACTOR
        return project_weight_budget(w, self.cap_, TIME_IDX)
