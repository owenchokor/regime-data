import pickle, numpy as np, pandas as pd
from ic import ic_table
X = pickle.load(open("feats_ex.pkl", "rb")); F = pickle.load(open("F_grp.pkl", "rb"))
fd = pd.read_parquet("/home/claude/data/fund_monthly.parquet"); fd["p"] = pd.to_datetime(fd.date).dt.to_period("M")
fd = fd.drop_duplicates(["p", "ticker"]).set_index(["p", "ticker"])
c = pd.read_parquet("/home/claude/data/panel_close.parquet"); c.index = pd.to_datetime(c.index)
cm = c.groupby(c.index.to_period("M")).last()
def get(col, lag=0):
    return [fd[col].get((t - lag, tk), np.nan) for t, tk in zip(F.t, F.ticker)]
F["EPS"] = get("EPS"); F["BPS"] = get("BPS"); F["DIV"] = get("DIV"); F["PBR"] = get("PBR")
F["EPS_12"] = get("EPS", 12); F["EPS_3"] = get("EPS", 3); F["BPS_12"] = get("BPS", 12)
px = np.array([cm[tk].get(t, np.nan) if tk in cm.columns else np.nan for t, tk in zip(F.t, F.ticker)])
F["EP"] = F.EPS / px; F["BP"] = F.BPS / px
F["PER"] = np.where(F.EPS > 0, px / F.EPS, np.nan)
F["loss"] = (F.EPS <= 0).astype(float).where(F.EPS.notna())
F["eps_g12"] = (F.EPS - F.EPS_12) / F.EPS_12.abs().replace(0, np.nan)
F["eps_g3"] = (F.EPS - F.EPS_3) / F.EPS_3.abs().replace(0, np.nan)
F["bps_g12"] = F.BPS / F.BPS_12 - 1
F["impair"] = (F.BPS <= 0).astype(float).where(F.BPS.notna())
pickle.dump(F, open("F_all.pkl", "wb"))
print("coverage", F[["EPS", "BPS", "PBR", "EPS_12"]].notna().mean().round(2).to_dict(), "loss rate", round(F.loss.mean(), 3))
FEATS = ["r1m", "r3m", "r6m", "r12m", "dist_low", "h52", "trend_age", "newhi20", "vol60", "vol20", "dd60", "turn_ratio", "amt_ratio", "log_amt", "log_mcap", "crowd",
         "EP", "BP", "PER", "PBR", "DIV", "loss", "eps_g12", "eps_g3", "bps_g12"]
T = ic_table(F, FEATS); pd.set_option("display.width", 200); print(T.round(3).to_string())
T.to_pickle("ic_all.pkl")
