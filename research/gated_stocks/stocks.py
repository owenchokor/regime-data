"""개별주 패널: 수정 수익률 복원 → 월간 지수·피처·유니버스."""
from __future__ import annotations
import numpy as np
import pandas as pd

DATA = "../../data"
JUMP = 0.12        # 가격 점프 탐지 임계 (임의값) — 단, 주식수 역비율 매칭이 있어야만 조정
WIN_PRE, WIN_POST = 5, 40   # 주식수 변화 탐색 창(거래일). 레퍼런스: 무상증자 신주상장 최대 +35 달력일
TOL = 0.10         # |ln(q)+ln(p)| 허용치 (임의값)
TOPN_MCAP = 500    # 유니버스: 형성월 시총 상위 N (미확정 항목 — 300/1000 강건성 병행)


def load_panels():
    c = pd.read_parquet(f"{DATA}/panel_close.parquet").astype("float64")
    s = pd.read_parquet(f"{DATA}/panel_shares.parquet").astype("float64")
    v = pd.read_parquet(f"{DATA}/panel_volume.parquet").astype("float64")
    mc = pd.read_parquet(f"{DATA}/panel_mcap.parquet").astype("float64")
    for x in (c, s, v, mc):
        x.index = pd.to_datetime(x.index)
    return c, s, v, mc


def adjusted_returns(c: pd.DataFrame, s: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """가격 점프 중 주식수 역비율 변화가 창 안에서 매칭되는 것만 기업행동으로 보고 제거.
    매칭 안 되는 점프는 실제 변동(정리매매 등)일 수 있어 그대로 둔다 → 플래그만."""
    r = c / c.shift(1) - 1
    lr = np.log1p(r)
    ls = np.log(s / s.shift(1))
    C, S = lr.values, ls.values
    adj = C.copy()
    ev = []
    T, N = C.shape
    cand = np.argwhere(np.abs(np.nan_to_num(C)) > np.log1p(JUMP))
    used = np.zeros_like(S, dtype=bool)
    for t, j in cand:
        lo, hi = max(0, t - WIN_PRE), min(T, t + WIN_POST + 1)
        seg = S[lo:hi, j]
        ok = np.where(~np.isnan(seg) & (np.abs(seg) > 0.005) & ~used[lo:hi, j])[0]
        match = None
        # 창 안 단일 변화 또는 누적 변화(분할 상장) 모두 허용
        for k in ok:
            if abs(seg[k] + C[t, j]) < TOL:
                match = [k]; break
        if match is None and len(ok) > 1:
            cum = np.cumsum(seg[ok])
            hit = np.where(np.abs(cum + C[t, j]) < TOL)[0]
            if len(hit):
                match = list(ok[: hit[0] + 1])
        if match is not None:
            q = sum(seg[k] for k in match)
            adj[t, j] = C[t, j] + q
            for k in match:
                used[lo + k, j] = True
            ev.append((c.index[t], c.columns[j], np.expm1(C[t, j]), "matched", t - lo - match[0]))
        elif C[t, j] > np.log1p(0.30) or C[t, j] < np.log1p(-0.30):
            ev.append((c.index[t], c.columns[j], np.expm1(C[t, j]), "unmatched", np.nan))
            # +30% 초과 단일일 상승은 가격제한폭상 정상 거래로 불가 → 기업행동 미매칭으로 보고 0 처리.
            # 하락 쪽은 정리매매(제한폭 없음)일 수 있어 유지 — 보수적 비대칭
            if C[t, j] > 0:
                adj[t, j] = 0.0
    ra = pd.DataFrame(np.expm1(adj), index=c.index, columns=c.columns)
    E = pd.DataFrame(ev, columns=["date", "ticker", "raw_ret", "type", "lag"])
    return ra, E


def monthly_panel(ra, c, v, mc, unmatched: pd.DataFrame):
    idx = (1 + ra.fillna(0)).cumprod().where(c.notna())
    per = c.index.to_period("M")
    g = lambda x: x.groupby(per)
    M = g(idx).last()                               # 월말(또는 월중 마지막 유효가 — 상폐 달)
    last_vol = g(v).last()
    mcap = g(mc).last()
    dret = ra
    feats = {}
    P = M
    feats["mom12_1"] = P.shift(1) / P.shift(12) - 1
    feats["mom6_1"] = P.shift(1) / P.shift(6) - 1
    feats["mom1"] = P / P.shift(1) - 1
    feats["mom3"] = P / P.shift(3) - 1
    hi252 = idx.rolling(252, min_periods=200).max()
    feats["high52"] = g(idx / hi252).last()
    vol = dret.rolling(252, min_periods=200).std() * np.sqrt(252)
    feats["vol12"] = g(vol).last()
    feats["ramom"] = feats["mom12_1"] / feats["vol12"]
    # 데이터 품질 필터(과거 정보만): 최근 12개월 내 미매칭 30%+ 점프 종목 제외
    um = unmatched.assign(p=unmatched.date.dt.to_period("M"))
    bad = pd.DataFrame(0.0, index=M.index, columns=M.columns)
    for (p, tk), _ in um.groupby(["p", "ticker"]):
        if p in bad.index:
            bad.loc[p, tk] = 1.0
    bad12 = bad.rolling(12, min_periods=1).max()
    common = np.array([t.endswith("0") for t in M.columns])   # 우선주(끝자리≠0) 제외
    age = M.notna().cumsum()
    elig = (M.notna() & (last_vol > 0) & (age >= 13) & (bad12 == 0)) & common
    rank = mcap.where(elig).rank(axis=1, ascending=False)
    fwd = M.shift(-1) / M - 1                          # t월 형성 → t+1월 수익 (상폐 달은 마지막 가격까지)
    return dict(M=M, feats=feats, elig=elig, rank=rank, fwd=fwd, mcap=mcap)
