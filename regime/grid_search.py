"""
그리드 탐색 — IS(2014-2020) 구간에서 최적 파라미터 선택
평가 지표: Fβ(β=0.5) — PPV에 2배 가중

실행: python regime/grid_search.py --close data/close.parquet
                                    --kospi data/kospi.parquet
                                    --kosdaq data/kosdaq.parquet
                                    --krw data/fred_DEXKOUS.parquet
                                    --us10 data/fred_DGS10.parquet
"""
import argparse, itertools, json
from pathlib import Path
import numpy as np, pandas as pd
from regime.core import RegimeCFG, build_regime

IS_END = "2020-12-31"
BETA   = 0.5
MIN_COVER = 0.15   # 강세 판정 비중 최소 15% — 너무 보수적인 조합 제거

# 탐색 그리드 (전부 임의값, 결과 보고 후 좁힐 것)
GRID = dict(
    ma_n     = [50, 120, 200],
    b_hi     = [0.50, 0.55, 0.60, 0.65],
    b_lo     = [0.30, 0.35, 0.40, 0.45],
    dmax     = [4, 6, 99],
    fxmax    = [float("inf"), 0.03, 0.05],
    rmax     = [float("inf"), 0.3, 0.5],
    idx_mode = ["kospi", "kosdaq", "either", "both"],
    confirm  = [1, 3, 5],
)

def fb_score(ppv: float, rec: float, beta: float = BETA) -> float:
    b2 = beta ** 2
    denom = b2 * ppv + rec
    return (1 + b2) * ppv * rec / denom if denom else 0.0

def evaluate(state: np.ndarray, y_bull: np.ndarray, y_bear: np.ndarray,
             mask: np.ndarray) -> dict | None:
    b = (state == 2)[mask]; w = (state == 0)[mask]
    t_bull = y_bull[mask]; t_bear = y_bear[mask]
    if b.mean() < MIN_COVER or b.sum() == 0: return None
    ppv  = (b & t_bull).sum() / b.sum()
    rec  = (b & t_bull).sum() / max(t_bull.sum(), 1)
    bppv = (w & t_bear).sum() / max(w.sum(), 1)
    return dict(fb=fb_score(ppv, rec), ppv=ppv, rec=rec,
                cover=float(b.mean()), bear_ppv=float(bppv))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--close",  required=True)
    ap.add_argument("--kospi",  required=True)
    ap.add_argument("--kosdaq", required=True)
    ap.add_argument("--krw",    required=True)
    ap.add_argument("--us10",   required=True)
    ap.add_argument("--out",    default="results/grid.csv")
    args = ap.parse_args()

    close  = pd.read_parquet(args.close);  close.index  = pd.to_datetime(close.index)
    kospi  = pd.read_parquet(args.kospi);  kospi.index  = pd.to_datetime(kospi.index)
    kosdaq = pd.read_parquet(args.kosdaq); kosdaq.index = pd.to_datetime(kosdaq.index)
    krw    = pd.read_parquet(args.krw).iloc[:, 0]
    us10   = pd.read_parquet(args.us10).iloc[:, 0]

    d      = close.index
    ok     = np.ones(len(d), bool)
    is_    = (d <= IS_END).values
    oos_   = ~is_

    # 정답 라벨 — forward FWD=20 영업일 개별주 중앙값
    fwd_ret = close.pct_change(20, fill_method=None).shift(-20).clip(-0.5, 0.5).median(axis=1)
    ok &= fwd_ret.notna().values
    q_lo, q_hi = fwd_ret[is_ & ok].quantile([1/3, 2/3])
    y_bull = (fwd_ret >= q_hi).values
    y_bear = (fwd_ret <= q_lo).values
    print(f"기저율 PPV(강세): {y_bull[is_ & ok].mean():.3f} | IS n={is_.sum()} OOS n={oos_.sum()}")

    keys = list(GRID.keys()); rows = []
    total = 1
    for v in GRID.values(): total *= len(v)
    print(f"총 조합: {total}")

    for i, vals in enumerate(itertools.product(*GRID.values())):
        p = dict(zip(keys, vals))
        cfg = RegimeCFG(**p)
        try:
            res = build_regime(close, kospi, kosdaq, krw, us10, cfg)
        except Exception as e:
            continue
        st = res["state"].values
        s_is  = evaluate(st, y_bull, y_bear, is_ & ok)
        s_oos = evaluate(st, y_bull, y_bear, oos_ & ok)
        if s_is and s_oos:
            rows.append({**p,
                         **{f"is_{k}":  v for k, v in s_is.items()},
                         **{f"oos_{k}": v for k, v in s_oos.items()}})
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{total} done, valid so far: {len(rows)}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    grid = pd.DataFrame(rows).sort_values("is_fb", ascending=False)
    grid.to_csv(out, index=False)
    print(f"\n유효 조합: {len(grid)}")
    print(grid.head(5)[list(GRID.keys()) + ["is_fb","is_ppv","oos_fb","oos_ppv","oos_cover"]])

    # 최적 파라미터 저장
    best = grid.iloc[0][list(GRID.keys())].to_dict()
    (out.parent / "best_params.json").write_text(
        json.dumps({**best, "q_lo": q_lo, "q_hi": q_hi}, ensure_ascii=False, indent=2))
    print("best:", best)

if __name__ == "__main__":
    main()
