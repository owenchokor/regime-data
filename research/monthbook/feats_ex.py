"""사전(결정일) 정보만으로 후보 종목-월 피처 계산 → 사후 수익과 결합."""
import pickle, numpy as np, pandas as pd
from variants import Sim
import stocks
sim = Sim(); ra = sim.ra; S = sim.S
c, sh, v, mc = stocks.load_panels()
amt = pd.read_parquet("/home/claude/data/panel_amount.parquet"); amt.index = pd.to_datetime(amt.index)
amt = amt.reindex(index=c.index, columns=c.columns)
book = pickle.load(open("monthbook.pkl", "rb"))
days = ra.index; per = days.to_period("M"); me = pd.Series(days).groupby(per).max(); pos = {d: i for i, d in enumerate(days)}
idx = (1 + ra).cumprod().where(c.notna())
hi = idx.rolling(252, min_periods=200).max(); lo = idx.rolling(252, min_periods=200).min()
near = (idx / hi >= 0.95)
turn = (v / sh).replace([np.inf, -np.inf], np.nan)
k = pd.read_parquet("/home/claude/data/kospi.parquet")["종가"]; k.index = pd.to_datetime(k.index)
sec = pd.read_parquet("/home/claude/data/sector_monthly.parquet"); sec["p"] = pd.to_datetime(sec["date"]).dt.to_period("M")
rows = []
for t, b in book.items():
    d = me[t]; i = pos[d]; tk = list(b["table"].ticker)
    W = lambda n: days[max(0, i - n + 1):i + 1]
    I = idx.loc[:, tk]
    f = pd.DataFrame(index=tk)
    for n, nm in [(21, "r1m"), (63, "r3m"), (126, "r6m"), (252, "r12m")]:
        f[nm] = I.loc[d] / I.iloc[max(0, i - n)] - 1
    f["dist_low"] = I.loc[d] / lo.loc[d, tk] - 1
    f["h52"] = I.loc[d] / hi.loc[d, tk]
    nr = near.loc[W(252), tk][::-1]
    f["trend_age"] = nr.cumprod().sum()                       # 연속으로 고점 5% 이내에 머문 거래일 수
    newhi = (idx.loc[W(20), tk] >= hi.loc[W(20), tk] - 1e-12).sum(); f["newhi20"] = newhi
    rr = ra.loc[W(60), tk]; f["vol60"] = rr.std() * np.sqrt(252); f["vol20"] = ra.loc[W(20), tk].std() * np.sqrt(252)
    p60 = (1 + rr).cumprod(); f["dd60"] = (p60 / p60.cummax() - 1).min()
    f["turn_ratio"] = turn.loc[W(20), tk].mean() / turn.loc[W(120), tk].mean()
    f["amt_ratio"] = amt.loc[W(20), tk].mean() / amt.loc[W(120), tk].mean()
    f["log_amt"] = np.log(amt.loc[W(20), tk].mean())
    f["log_mcap"] = np.log(mc.loc[d, tk])
    f["zero_vol_days"] = (v.loc[W(20), tk] == 0).sum()
    ss = sec[sec.p == sec.p[sec.p <= t].max()]; s = ss.drop_duplicates("ticker").set_index("ticker")["업종명"].reindex(tk)
    f["sector"] = s.values; f["crowd"] = s.map(s.value_counts()).values
    f["ret"] = b["table"].set_index("ticker").ret.reindex(tk)
    f["name"] = b["table"].set_index("ticker")["name"].reindex(tk)
    f["t"] = t; f["ticker"] = tk
    rows.append(f)
F = pd.concat(rows, ignore_index=True)
# 월 단위 거시 (결정일 기준)
km = k.groupby(k.index.to_period("M")).last()
ec = pd.read_parquet("/home/claude/data/ecos_721Y001.parquet"); ec["p"] = pd.PeriodIndex(ec["time"], freq="M")
rate = ec.pivot_table(index="p", columns="item_name", values="value")
fx = pd.read_parquet("/home/claude/data/fred_DEXKOUS.parquet").iloc[:, 0]; fx = fx.groupby(fx.index.to_period("M")).last()
u10 = pd.read_parquet("/home/claude/data/fred_DGS10.parquet").iloc[:, 0]; u10 = u10.groupby(u10.index.to_period("M")).last()
nq = pd.read_parquet("/home/claude/data/fred_NASDAQCOM.parquet").iloc[:, 0]; nq = nq.groupby(nq.index.to_period("M")).last()
M = pd.DataFrame({"kospi_1m": km.pct_change(), "kospi_3m": km.pct_change(3), "kospi_12m": km.pct_change(12),
                  "kospi_vs_ma10": km / km.rolling(10).mean() - 1,
                  "cd": rate["CD(91일)"], "cd_3m_chg": rate["CD(91일)"].diff(3), "ktb3_3m_chg": rate["국고채(3년)"].diff(3),
                  "term": rate["국고채(10년)"] - rate["국고채(3년)"], "credit": rate["회사채(3년, BBB-)"] - rate["회사채(3년, AA-)"],
                  "usdkrw_3m": fx.pct_change(3), "us10_3m_chg": u10.diff(3), "nq_3m": nq.pct_change(3)})
M["credit_3m_chg"] = M["credit"].diff(3)
# ECOS 는 해당 월 평균 → 결정 시점에는 전월 값까지만 안다고 보고 1개월 지연
for col in ["cd", "cd_3m_chg", "ktb3_3m_chg", "term", "credit", "credit_3m_chg"]:
    M[col] = M[col].shift(1)
pickle.dump(dict(F=F, M=M), open("feats_ex.pkl", "wb"))
print(F.shape); print(F.describe().T[["mean", "50%"]].round(3).to_string())
