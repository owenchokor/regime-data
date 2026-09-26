"""실험 2: (A) 위험 고려 KOSPI 타이밍 재분석, (B) KOSPI 레짐 게이트 × 개별주 전략.
현금 수익·무위험수익 = ECOS CD(91일) 월평균/12. 거시 게이트 피처는 공표 시점 보수화를 위해 1개월 지연."""
from __future__ import annotations
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import lightgbm as lgb
from core import *
from strategies import rule_grid, fold_bounds

SEED = 42
TOPN = 20                    # 보유 종목 수 (임의값, 10/30/50 강건성)
UNIV = 500                   # 시총 상위 N (미확정 항목, 300/1000 강건성)
STOCK_SEL_COST = 50          # 개별주 nested 선택 비용 bps (세금 포함 보수값, 임의값)
IDX_SEL_COST = 20
OBJECTIVES = ["CAGR", "Sharpe", "Sortino", "Calmar", "CAGR|MDD"]

# ── 공통 데이터 ──
d = load_daily(); m = to_monthly(d); f = features(m); r = m["kospi"].pct_change()
ec = pd.read_parquet(f"{DATA}/ecos_721Y001.parquet")
ec["p"] = pd.PeriodIndex(ec["time"], freq="M")
rate = ec.pivot_table(index="p", columns="item_name", values="value")
rf = (rate["CD(91일)"] / 100 / 12).reindex(r.index)          # t월 보유 현금 수익(실현값)
term = (rate["국고채(10년)"] - rate["국고채(3년)"]).reindex(r.index)
cred = (rate["회사채(3년, BBB-)"] - rate["회사채(3년, AA-)"]).reindex(r.index)
cd = rate["CD(91일)"].reindex(r.index)


def gates() -> dict[str, pd.Series]:
    fam = rule_grid(f)
    g = {"ALWAYS": pd.Series(1.0, index=r.index)}
    g.update({k: fam["DD"][k] for k in ["DD12>-5%", "DD12>-10%"]})
    g.update({k: fam["PxMA"][k] for k in ["PxMA10", "PxMA12"]})
    g["TSMOM12"] = fam["TSMOM"]["TSMOM12"]
    g["VT6"] = fam["VolTarget"]["VT6"]
    ind = lambda s: (s > 0).astype(float).where(s.notna())
    # 거시: 1개월 지연(공표 보수화)
    g["TermUp"] = ind(term.shift(1) - term.shift(1).expanding(12).median())
    g["CredEasing"] = ind(-(cred.shift(1) - cred.shift(4)))
    g["RateFalling"] = ind(-(cd.shift(1) - cd.shift(4)) + 1e-9)
    return g


def rmetrics(x: pd.Series, rfx: pd.Series) -> dict:
    x = x.dropna(); e = x - rfx.reindex(x.index)
    n = len(x)
    if n < 12:
        return dict(n=n, CAGR=np.nan, Vol=np.nan, Sharpe=np.nan, Sortino=np.nan, MDD=np.nan,
                    Calmar=np.nan, CVaR5=np.nan, MaxUW=np.nan, WinRate=np.nan)
    eq = (1 + x).cumprod()
    cagr = eq.iloc[-1] ** (12 / n) - 1
    mdd = (eq / eq.cummax().clip(lower=1) - 1).min()
    dd = np.sqrt((np.minimum(e, 0) ** 2).mean()) * np.sqrt(12)
    uw = (eq < eq.cummax().clip(lower=1)).astype(int)
    longest = (uw.groupby((uw == 0).cumsum()).cumsum()).max()
    k = max(1, int(np.ceil(0.05 * n)))
    return dict(n=n, CAGR=cagr, Vol=x.std() * np.sqrt(12),
                Sharpe=e.mean() / e.std() * np.sqrt(12) if e.std() > 0 else np.nan,
                Sortino=e.mean() * 12 / dd if dd > 0 else np.nan,
                MDD=mdd, Calmar=cagr / abs(mdd) if mdd < 0 else np.nan,
                CVaR5=np.sort(x.values)[:k].mean(), MaxUW=int(longest), WinRate=(x > 0).mean())


def objective(x: pd.Series, rfx: pd.Series, name: str, bh_mdd: float) -> float:
    mt = rmetrics(x, rfx)
    if name == "CAGR|MDD":      # 훈련구간 B&H보다 MDD가 나쁘면 탈락
        return mt["CAGR"] if mt["MDD"] > bh_mdd else -np.inf
    return -np.inf if pd.isna(mt[name]) else mt[name]


# ── (A) 지수 타이밍: 현금=CD ──
def idx_returns(pos: pd.Series, cost: float) -> pd.Series:
    held = pos.reindex(r.index).shift(1)
    turn = held.diff().abs(); first = held.notna() & held.shift(1).isna(); turn[first] = held[first].abs()
    return held * r + (1 - held) * rf - turn * cost / 1e4


def nested(cands: dict[str, pd.Series], retfn, obj: str, cost: float, bench: pd.Series):
    """cands: 결정월 포지션/식별자, retfn(name, cost)→수익. 연 단위 expanding 재선택."""
    cache = {k: retfn(k, cost) for k in cands}
    out = pd.Series(np.nan, index=r.index); picks = []
    for y in WF_YEARS:
        tr_end, o0, o1 = fold_bounds(y)
        bh_mdd = rmetrics(bench.loc[EVAL_START:tr_end], rf)["MDD"]
        sc = {k: objective(v.loc[EVAL_START:tr_end], rf, obj, bh_mdd) for k, v in cache.items()}
        best = max(sc, key=lambda k: -np.inf if pd.isna(sc[k]) else sc[k])
        picks.append((y, best))
        out.loc[o0:o1] = cache[best].loc[o0:o1]
    return out, picks


def freeze(cands, retfn, obj, cost, bench):
    cache = {k: retfn(k, cost) for k in cands}
    bh_mdd = rmetrics(bench.loc[EVAL_START:DEV_END], rf)["MDD"]
    sc = {k: objective(v.loc[EVAL_START:DEV_END], rf, obj, bh_mdd) for k, v in cache.items()}
    return max(sc, key=lambda k: -np.inf if pd.isna(sc[k]) else sc[k])


# ── (B) 개별주 ──
S = pickle.load(open("stocks.pkl", "rb"))["MP"]


def stock_weights(univ: int = UNIV, topn: int = TOPN) -> dict[str, pd.DataFrame]:
    F, el, rk = S["feats"], S["elig"], S["rank"]
    U = el & (rk <= univ)
    def top(score: pd.DataFrame, asc=False):
        sc = score.where(U & score.notna())
        rr = sc.rank(axis=1, ascending=asc, method="first")
        w = (rr <= topn).astype(float)
        return w.div(w.sum(axis=1).replace(0, np.nan), axis=0)
    W = {
        "MOM12-1": top(F["mom12_1"]), "MOM6-1": top(F["mom6_1"]), "52WH": top(F["high52"]),
        "REV1": top(F["mom1"], asc=True), "REV3": top(F["mom3"], asc=True),
        "LOWVOL": top(F["vol12"], asc=True), "RAMOM": top(F["ramom"]),
        "MOM+LV": top(F["mom12_1"].where(U).rank(axis=1, pct=True) - F["vol12"].where(U).rank(axis=1, pct=True)),
    }
    ew = U.astype(float); W["UNIV-EW"] = ew.div(ew.sum(axis=1), axis=0)
    return W


def ml_weights(univ: int = UNIV, topn: int = TOPN, window="expanding") -> dict[str, pd.DataFrame]:
    """횡단면 순위 회귀: 월별 순위 정규화 피처 → 다음 달 수익 순위(원/위험조정)."""
    F, el, rk, fwd = S["feats"], S["elig"], S["rank"], S["fwd"]
    U = el & (rk <= univ)
    names = ["mom12_1", "mom6_1", "mom1", "mom3", "high52", "vol12", "ramom"]
    X = {k: F[k].where(U).rank(axis=1, pct=True) - 0.5 for k in names}
    X["size"] = rk.where(U).rank(axis=1, pct=True) - 0.5
    long = pd.concat({k: v.stack() for k, v in X.items()}, axis=1).dropna()
    yr = fwd.where(U).rank(axis=1, pct=True).stack().rename("y_raw") - 0.5
    ya = (fwd / F["vol12"]).where(U).rank(axis=1, pct=True).stack().rename("y_risk") - 0.5
    D = long.join(yr).join(ya)
    per = D.index.get_level_values(0)
    scores = {f"{m}_{t}": pd.Series(np.nan, index=D.index) for m in ["Ridge", "LGBM"] for t in ["raw", "risk"]}
    folds = [(y, *fold_bounds(y)) for y in WF_YEARS] + [("H", DEV_END, HOLD_START, LAST_MONTH)]
    for y, tr_end, o0, o1 in folds:
        lo = pd.Period("2014-12", "M") if window == "expanding" else pd.Period(tr_end) - 37
        tr = D[(per > lo) & (per <= pd.Period(tr_end) - 1)]
        te_mask = (per >= pd.Period(o0) - 1) & (per <= pd.Period(o1) - 1)
        te = D[te_mask]
        cols = list(X)
        for t in ["raw", "risk"]:
            trt = tr.dropna(subset=[f"y_{t}"])
            m1 = Ridge(alpha=1.0).fit(trt[cols], trt[f"y_{t}"])        # alpha 임의값
            scores[f"Ridge_{t}"].loc[te.index] = m1.predict(te[cols])
            m2 = lgb.LGBMRegressor(n_estimators=200, num_leaves=15, min_child_samples=200,
                                   learning_rate=0.03, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.8, random_state=SEED, num_threads=1,
                                   verbose=-1).fit(trt[cols], trt[f"y_{t}"])   # 하이퍼파라미터 임의값
            scores[f"LGBM_{t}"].loc[te.index] = m2.predict(te[cols])
    W = {}
    for k, s in scores.items():
        sc = s.unstack().reindex(index=U.index, columns=U.columns)
        rr = sc.rank(axis=1, ascending=False, method="first")
        w = (rr <= topn).astype(float)
        W[f"ML:{k}"] = w.div(w.sum(axis=1).replace(0, np.nan), axis=0)
    return W


def stock_port(W: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    fwd = S["fwd"].reindex(columns=W.columns).fillna(0.0)    # 형성 직후 상폐 등 미관측 → 0 (한계에 명시)
    gross = (W.fillna(0) * fwd).sum(axis=1).where(W.notna().any(axis=1) & (W.sum(axis=1) > 0))
    return gross, W.fillna(0)


def gated(W: pd.DataFrame, g: pd.Series, cost: float) -> pd.Series:
    """t월 결정: 비중 = g_t × W_t, 나머지 현금(CD). 수익은 t+1월 인덱스로 정렬."""
    gross, Wf = stock_port(W)
    gg = g.reindex(Wf.index).astype(float)
    Wg = Wf.mul(gg, axis=0)
    traded = (Wg - Wg.shift(1).fillna(0)).abs().sum(axis=1)
    ret_dec = gg * gross + (1 - gg) * rf.shift(-1).reindex(Wf.index) - traded * cost / 1e4
    out = ret_dec.shift(1).reindex(r.index)
    return out.where(gg.shift(1).reindex(r.index).notna() & gross.shift(1).reindex(r.index).notna())
