# 3차: 52WH × 게이트 변형 선별

의존: `../gated_stocks`, `../monthly_timing`. 실행: run3.py → run3b.py(V7) → figs3.py → build3.py
`variants.py` Sim: 일별 경로로 종목 스톱·포트폴리오 스톱·일간 게이트 시뮬레이션. 스톱 체결 = 트리거 다음날 종가, 재선정 시 재매수 비용 2×50bp.

| 변형 | WF Sharpe | 판정 |
|---|---|---|
| 기준형 | 0.91 | – |
| V1 과열제외 1M 상위10% | 0.63 | 탈락 |
| V2 역변동성 가중 | 0.82 | 탈락 |
| V3 회전율 증가 | 0.93 | 탈락(위험 개선 없음) |
| V4 종목 트레일링 15% | 0.88 | 탈락(이웃 20% 실패) |
| V5 포트폴리오 스톱 10% | 0.80 | 탈락 |
| V6 일간 200일선 게이트 | 0.91 | 탈락 |
| V7 업종당 최대 4 | 0.77 | 탈락 |
| V8 거래대금 하위 30% 제외 | – | 거래대금 수집 후 판정 |

사후 발견: 종목 스톱 10%(V4 이웃값) Sharpe 0.99, MDD −9.7% → forward ② 신규가설로 운용.
V8 실행: `amount = pd.read_parquet(panel_amount.parquet)`(index datetime) → `select_liq(sim, 0.30, amount)` → `sim.month_returns(W, {}, gate, rf)` − `turnover_cost`, 이웃 0.2/0.4 동일. build3.py에 V8 행 추가.
