"""룰 그리드 · nested 선택 · ML walk-forward."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import (LinearRegression, LogisticRegressionCV, RidgeCV,
                                  LassoCV, ElasticNetCV)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from core import *

warnings.filterwarnings("ignore")
SEED = 42


# ── Level 1: 파라미터화된 단순 룰 (패밀리 → {파라미터: 포지션}) ──
def rule_grid(f: pd.DataFrame) -> dict[str, dict[str, pd.Series]]:
    fam: dict[str, dict[str, pd.Series]] = {}
    ind = lambda s: (s > 0).astype(float).where(s.notna())
    fam["TSMOM"] = {f"TSMOM{L}": ind(f[f"mom{L}"]) for L in [1, 2, 3, 6, 9, 12, 18, 24]}
    fam["PxMA"] = {f"PxMA{N}": ind(f[f"pma{N}"]) for N in
                   [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 24]}
    fam["DualMA"] = {f"MA{s}/{l}": ind(np.log(f[f"ma{s}"] / f[f"ma{l}"]))
                     for s in [2, 3, 6] for l in [9, 12, 18, 24]}
    fam["DD"] = {f"DD12>-{X}%": (f["dd12"] > -X / 100).astype(float).where(f["dd12"].notna())
                 for X in [5, 10, 15, 20, 25]}
    vol_med = lambda W: f[f"vol{W}"].expanding(12).median()  # 과거까지의 중앙값만 사용
    fam["VolRegime"] = {f"Vol{W}<med": (f[f"vol{W}"] < vol_med(W)).astype(float)
                        .where(vol_med(W).notna()) for W in [3, 6, 12]}
    fam["VolTarget"] = {f"VT{W}": (vol_med(W) / f[f"vol{W}"]).clip(upper=1.0)
                        for W in [3, 6, 12]}
    fam["RelStr"] = {f"Rel{L}": ind(f[f"rel{L}"]) for L in [1, 3, 6, 12]}
    fam["RelStr"].update({f"QpMA{N}": ind(f[f"qpma{N}"]) for N in [6, 12]})
    fam["MASlope"] = {f"Slope{N}": ind(f[f"maslope{N}"]) for N in [6, 12]}
    return fam


def committee(f: pd.DataFrame) -> pd.Series:
    """사전 고정 추세 위원회: TSMOM{3,6,9,12} + PxMA{3,6,9,12} 8표 평균 → 연속 포지션."""
    votes = [(f[f"mom{L}"] > 0).astype(float) for L in [3, 6, 9, 12]] + \
            [(f[f"pma{N}"] > 0).astype(float) for N in [3, 6, 9, 12]]
    return pd.concat(votes, axis=1).mean(axis=1)


def fold_bounds(year: int) -> tuple[str, str, str]:
    """(훈련 수익월 끝, OOS 시작, OOS 끝)."""
    return f"{year-1}-12", f"{year}-01", (DEV_END if year == 2024 else f"{year}-12")


def nested_rule(cands: dict[str, pd.Series], r: pd.Series, window: str) -> tuple[pd.Series, list]:
    """각 fold 훈련구간에서 Sharpe(20bps) 최대 파라미터 선택 → 다음 OOS 적용."""
    pos = pd.Series(np.nan, index=r.index)
    picks = []
    for y in WF_YEARS:
        tr_end, o0, o1 = fold_bounds(y)
        tr_start = EVAL_START if window == "expanding" else \
            str(pd.Period(tr_end) - ROLL_WIN + 1)
        best, bs = None, -np.inf
        for name, p in cands.items():
            x = strat_returns(p, r, SEL_COST_BPS).loc[tr_start:tr_end]
            s = sharpe(x.values)
            if s > bs:
                best, bs = name, s
        picks.append((y, best, round(bs, 3)))
        # t월말 결정 포지션: OOS 수익월 o0..o1 → 결정월 o0-1 .. o1-1
        dec = pd.period_range(pd.Period(o0) - 1, pd.Period(o1) - 1, freq="M")
        pos.loc[dec] = cands[best].loc[dec]
    return pos, picks


def nested_all_rules(fam: dict, r: pd.Series, window: str) -> tuple[pd.Series, list]:
    flat = {k: v for d in fam.values() for k, v in d.items()}
    return nested_rule(flat, r, window)


def freeze_rule(cands: dict[str, pd.Series], r: pd.Series) -> str:
    """개발구간 전체(EVAL_START~DEV_END)로 최종 파라미터 확정 — holdout용."""
    sc = {k: sharpe(strat_returns(p, r, SEL_COST_BPS).loc[EVAL_START:DEV_END].values)
          for k, p in cands.items()}
    return max(sc, key=sc.get)


# ── Level 2/3: ML ──
def _prune_corr(X: pd.DataFrame, thr: float = 0.8) -> list[str]:
    """훈련 데이터 내부에서만: 앞 순서(=더 단순한) 피처 우선 유지."""
    keep: list[str] = []
    c = X.corr().abs()
    for col in X.columns:
        if all(c.loc[col, k] < thr for k in keep):
            keep.append(col)
    return keep


def _models():
    tss = TimeSeriesSplit(n_splits=3)
    return {
        "OLS": ("reg", lambda: LinearRegression()),
        "Ridge": ("reg", lambda: RidgeCV(alphas=np.logspace(-2, 3, 20), cv=tss)),
        "Lasso": ("reg", lambda: LassoCV(cv=tss, random_state=SEED, max_iter=20000)),
        "ENet": ("reg", lambda: ElasticNetCV(l1_ratio=[.2, .5, .8], cv=tss,
                                              random_state=SEED, max_iter=20000)),
        "Logit": ("clf", lambda: LogisticRegressionCV(Cs=np.logspace(-3, 1, 12), cv=tss,
                                                      scoring="neg_log_loss", max_iter=5000)),
        # 트리 하이퍼파라미터: 임의값(미튜닝) — 표본이 작아 얕게 고정
        "RF": ("clf", lambda: RandomForestClassifier(n_estimators=500, max_depth=3,
                                                     min_samples_leaf=10, random_state=SEED,
                                                     n_jobs=1)),
        "GBM": ("clf", lambda: GradientBoostingClassifier(n_estimators=100, max_depth=2,
                                                          learning_rate=0.05, subsample=0.8,
                                                          random_state=SEED)),
        "LGBM": ("clf", lambda: lgb.LGBMClassifier(n_estimators=100, num_leaves=4,
                                                   min_child_samples=10, learning_rate=0.05,
                                                   subsample=0.8, subsample_freq=1,
                                                   colsample_bytree=0.8, random_state=SEED,
                                                   num_threads=1, verbose=-1)),
    }


def ml_walkforward(f: pd.DataFrame, r: pd.Series, window: str,
                   hold: bool = False) -> tuple[dict[str, pd.Series], dict]:
    """반환: {모델_포지션타입: 포지션}, 진단 정보.
    hold=True면 개발구간 전체로 1회 학습 → holdout 결정월에 적용(동결)."""
    X_all = f[ML_FEATS]
    y_next = r.shift(-1)                      # t행의 라벨 = r[t+1]
    specs = _models()
    out: dict[str, pd.Series] = {}
    diag: dict = {"feats": [], "coef": []}
    for k in specs:
        for pt in ["bin", "cont", "tri"]:
            out[f"{k}_{pt}"] = pd.Series(np.nan, index=r.index)
    folds = [("H", DEV_END, HOLD_START, LAST_MONTH)] if hold else \
            [(y, *fold_bounds(y)) for y in WF_YEARS]
    for y, tr_end, o0, o1 in folds:
        tr_start = str(pd.Period(EVAL_START) - 1) if window == "expanding" else \
            str(pd.Period(tr_end) - ROLL_WIN)
        # 라벨 r[t+1]이 tr_end 이하인 행만
        rows = X_all.loc[tr_start:str(pd.Period(tr_end) - 1)].dropna()
        yv = y_next.loc[rows.index]
        cols = _prune_corr(rows)
        diag["feats"].append((y, cols))
        sc = StandardScaler().fit(rows[cols])
        Xt = sc.transform(rows[cols])
        dec = pd.period_range(pd.Period(o0) - 1, pd.Period(o1) - 1, freq="M")
        Xo = sc.transform(X_all.loc[dec, cols])
        for k, (kind, mk) in specs.items():
            mdl = mk()
            if kind == "reg":
                mdl.fit(Xt, yv.values)
                pr_tr, pr = mdl.predict(Xt), mdl.predict(Xo)
                # 연속: 예측수익 / 훈련예측의 |최대| 로 0~1 매핑 (음수→0)
                scale = np.abs(pr_tr).max() or 1.0
                cont = np.clip(0.5 + pr / (2 * scale), 0, 1)
                binp = (pr > 0).astype(float)
                if k in ("OLS", "Lasso"):
                    diag["coef"].append((y, k, dict(zip(cols, np.round(mdl.coef_, 4)))))
            else:
                mdl.fit(Xt, (yv.values > 0).astype(int))
                pr_tr, pr = mdl.predict_proba(Xt)[:, 1], mdl.predict_proba(Xo)[:, 1]
                cont = pr
                binp = (pr > 0.5).astype(float)
                if k == "Logit":
                    diag["coef"].append((y, k, dict(zip(cols, np.round(mdl.coef_[0], 4)))))
            # 3단계: 훈련 예측값의 삼분위수로 0/0.5/1 (임계값 수작업 설정 회피)
            lo, hi = np.quantile(pr_tr, [1 / 3, 2 / 3])
            tri = np.where(pr > hi, 1.0, np.where(pr > lo, 0.5, 0.0))
            out[f"{k}_bin"].loc[dec] = binp
            out[f"{k}_cont"].loc[dec] = cont
            out[f"{k}_tri"].loc[dec] = tri
    # Level 4: 확률 앙상블 (Logit+RF+GBM 평균)
    out["EnsML_cont"] = (out["Logit_cont"] + out["RF_cont"] + out["GBM_cont"]) / 3
    out["EnsML_bin"] = (out["EnsML_cont"] > 0.5).astype(float).where(out["EnsML_cont"].notna())
    return out, diag
