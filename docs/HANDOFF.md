# 인계서 — 52WH 후보 × 정성 재선별 (2026-09-26 기준)

## 한 줄 요약
월간 기술적 신호 단독으로는 KOSPI 초과의 견고한 증거 없음(1~7차). 방향 전환: **52WH 20개 = 후보 생성기, 초과수익은 정성 재선별 IC가 만든다.**
8차 진단상 익일 체결·k=5 기준 IC ≈ 0.1이면 KOSPI 근접/초과, 0.2면 연 +23%p(중앙값). IC는 forward 채점으로만 측정 가능.

## 확정된 결정 (사용자)
- forward 트랙: ① 기준형, ② +종목 스톱 10%, ③ 정성 재선별 A안(월말 저녁 채점 → 익일 종가 매수).
- 채점 후보 20개, 보유 하한 5개 → forward IC ≥ 0.3 확인 후 3개.
- 초기자금 기준 1,000만원. 결과 확인 채널 = `forward/STATUS.md`.
- 운용 철학: 기계 신호는 후보일 뿐, 정성평가를 반드시 거친다. 적합 종목 없으면 유지 또는 현금화.
- 재무위험 종목 배제, 피크아웃/랠리 초입 구분은 정량 필터로 먼저 백테스트 (사용자 가설, holdout 결과 보기 전 제시).

## 리포·브랜치
- `main`: forward(①② + NAV) 머지 완료.
- `consolidate`: extra-data·nday·dart 통합 + 문서 재작성 + 8차 코드. **머지 필요** (PR 생성은 PAT 403 → 링크 전달).
- `extra-data`: build-amount 진행 중(06:41 UTC 75%). 완료 시 release `panel_amount.parquet`.
- `dart`: fetch-dart 진행 중 → release `dart_multi.parquet`, `dart_corp.parquet`, `dart_log.json`.
- 새 수집은 `on: push: branches:[<브랜치>] + paths` 방식 (workflow_dispatch는 default 브랜치 필요). schedule은 main에서만 동작.

## 다음 작업 (순서)
1. 거래대금 완료 → V8 판정(0.30, 이웃 0.2/0.4) → `research/variants/README.md` 갱신.
2. DART 완료 → `dart_log.json` status 분포·행 수 확인 → 재무위험 필터 + 추세 단계 필터 사전등록 → WF 판정(3차 기준).
   - point-in-time: `rcept_no` 앞 8자리 접수일 이후부터 사용. 정정공시 처리 확인 필요.
3. ③ 인프라: 월말 `forward/decisions/{월}.csv` 자동 생성(후보·요약) → 사용자 채점 커밋 → 워크플로가 IC(1M·3M)·선택군 성과·NAV를 STATUS.md에 표시. 익일 체결 기준 성과 병행 기록.
4. 8차 스펙트럼을 필터 적용 후보풀로 재계산 (필터 통과 시).
5. 확인 필요: 52WH 효과의 월말·월초 집중이 월말 수급 효과인지.

## 샌드박스 실행 요령
```
pip install -q pyarrow statsmodels arch lightgbm plotly tabulate --break-system-packages
git clone https://github.com/owenchokor/regime-data.git  # PAT: /mnt/project/Regime_github_PAT, 출력은 sed 마스킹
release 파일 → /home/claude/data, research 모듈 전부 한 작업 폴더로 복사 후 DATA 경로를 /home/claude/data 로 치환
prep_stocks.py 로 stocks.pkl 생성 후 Sim 사용
```
forward 코드 로컬 검증: `DATA_DIR=... FWD_FIRST_DECISION=2026-05 python forward/run_forward.py` (과거 구간 재현, 산출물은 커밋 금지).

## 함정
- bash 1회 300초 제한 → sleep ≤270초 분할 폴링. 기본 셸 sh → 복잡한 문법은 `bash -c`.
- plotly PNG 저장 불가 → HTML(`include_plotlyjs="cdn"`) 저장.
- `core.LAST_MONTH`(2026-08) 하드코딩 → forward는 자체 월 판정 사용.
- 개별주 월간 패널은 미완성 월 포함 → 수익 정렬 시 `r.index`로 reindex.
- SPA 입력 NaN → 첫 결측월 dropna. 전액 현금 부트스트랩 → nanpercentile.
- 진행 중 Actions 잡 로그는 API로 안 보임 → 완료 후 조회.
