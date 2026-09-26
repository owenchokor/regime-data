# 2차: KOSPI 레짐 게이트 × 개별주 · 위험 고려 재검증

의존: `../monthly_timing/` (core.py·strategies.py). 데이터: `panel_*.parquet`, `kospi/kosdaq.parquet`, `ecos_721Y001.parquet`.
실행: prep_stocks.py → run2.py → ev2a.py → ev2b.py → ev2c.py → figs2.py → build2.py

- `stocks.py`: 수정주가 복원(주식수 역비율 매칭, 미매칭 +30% 초과 → 0, −30% 미만 유지)·월간 패널·유니버스(보통주, 상장 13개월+, 월말 거래량>0, 12개월 미해명 점프 없음, 시총 상위 500).
- `exp2.py`: 게이트·위험지표·nested 선택·개별주/ML.

결론: 견고한 증거 없음. 52WH×PxMA10 — WF Sharpe 0.73 vs KOSPI 0.28, 연 알파 +10.5%(NW p=0.055), SPA p≈0.55~0.67, holdout CAGR −7.2% / MDD −51.6%(무작위보다 나쁨, 2026-06·07 급락). 위험 목적함수 선택 자체가 과최적화 축.
이 조합은 이후 '후보 생성기'로 재정의되어 forward에 사용 (`../README.md`).
