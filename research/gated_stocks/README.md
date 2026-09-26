# KOSPI 레짐 게이트 × 개별주 · 위험 고려 재검증 (2차)

의존: `../monthly_timing/` 의 core.py·strategies.py·figs.py 를 PYTHONPATH에 추가.
데이터: release data-latest 의 panel_*.parquet, kospi/kosdaq.parquet, ecos_721Y001.parquet → `../../data/`
실행: prep_stocks.py → run2.py → ev2a.py → ev2b.py → ev2c.py → figs2.py → build2.py

- ECOS: `collect/fetch_ecos.py` + `.github/workflows/fetch-ecos.yml` (시크릿 ECOS_KEY)
- 결론: No robust evidence (52WH×게이트 WF 양호, SPA p≈0.6, holdout 붕괴)
