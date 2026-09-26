# regime-data

국내주식 랠리 신호 시스템의 데이터 수집·연구·forward 모의운용 리포.
목표: 코스피를 이기는 CTA형 운용법. 기계 신호는 **후보 생성기**이고, 초과수익은 사람의 정성 재선별 층이 만든다는 전제로 운용한다.

## 현재 상태 (2026-09-26)
- 월간 기술적 신호 단독으로 KOSPI를 이기는 견고한 증거 없음 (1~7차 검증, `research/README.md`).
- 채택 구조: **52WH 상위 20 후보 × KOSPI PxMA10 게이트 → 월말 저녁 정성 채점 → 익일 종가에 상위 k개 매수** (8차 A안).
- forward 모의운용 가동 중 (첫 보유월 2026-10, `forward/README.md`). 정성 채점·IC 측정 인프라는 구축 예정.
- 수집 중: 거래대금 패널(V8 판정용), OpenDART 재무(재무위험 필터용).

## 구조
| 경로 | 내용 |
|---|---|
| `collect/` | 수집 스크립트 (KRX·FRED·ECOS·OpenDART) |
| `.github/workflows/` | 수집·forward Actions |
| `research/` | 사전등록 연구 1~8차 (`research/README.md`) |
| `forward/` | forward 모의운용 코드·산출물 |
| `docs/HANDOFF.md` | 세션 인계서 (다음 작업·함정) |
| `regime/` | 이전 국면 판정 모듈 (오닐 기반). 현 연구 라인에서는 미사용 |

## 데이터 (release `data-latest`)
`https://github.com/owenchokor/regime-data/releases/download/data-latest/<파일>`

| 파일 | 내용 | 수집 |
|---|---|---|
| `kospi` · `kosdaq.parquet` | KRX 지수 일봉 2014~ (가격지수) | daily-collect, forward |
| `panel_close` · `shares` · `volume` · `mcap.parquet` | 전종목 일별 횡단면(날짜×티커, 미수정, 상폐 포함) | build-panel, forward(증분) |
| `panel_amount.parquet` | 거래대금 패널 | build-amount (진행 중) |
| `fred_DGS10` · `DGS2` · `DEXKOUS` | 미 금리·환율 | daily-collect |
| `fred_NASDAQCOM` · `DFF` | 나스닥·연방기금금리 | fetch-us |
| `ecos_721Y001` · `722Y001` · `ecos_items.json` | 시장금리(CD 등)·기준금리 월별 | fetch-ecos |
| `index_fund_1001` · `2001` | 지수 PER·PBR·배당수익률 (TR 근사) | fetch-div, forward |
| `index_extra_1293` · `1294` | 코스피200 NTR (참고) | fetch-extra |
| `sector_monthly.parquet` | KRX 업종 월초 스냅샷 | fetch-extra |
| `admin_snapshot_*.parquet` | 현재 관리종목 (과거 이력 없음) | fetch-extra |
| `dart_multi` · `dart_corp.parquet` | OpenDART 다중회사 주요계정 (rcept_no 접수일 = 시점) | fetch-dart (진행 중) |

외부: 상폐 목록 `FinanceData/fdr_krx_data_cache` `data/listing/delisting/<날짜>.csv`.

## 시크릿
`KRX_ID` · `KRX_PW` · `ECOS_KEY` · `DART_KEY`. KRX·ECOS·FRED·DART 도메인은 Claude 샌드박스에서 접근 불가 → Actions로만 수집.

## 확인 필요
VKOSPI 소스, 관리종목·투자경고 과거 이력 소스, DART 정정공시 시 원공시 값 보존 여부.
