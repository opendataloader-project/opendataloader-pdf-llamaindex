# 코드리뷰 라운드 기록

## 변경 범위
- BASE: 2c91f374fbb326ec81f961c4b6bd13cca5cb0b01
- HEAD: bb96d0f1dbe94a1bb249156f61f77d6c5aa59187

---

## Round 1 (2026-04-09)

### 리뷰 결과 요약
- Critical: 0건
- Important: 4건 (#1 metadata format 불일치, #2 nested try 가독성, #3 ClassVar, #4 dependency-groups)
- Minor: 4건 (#6 nested with, #7 sed 패턴, #8 py.typed, #9 model_config)

### 판정
- **수용**: #1 metadata format → fmt 전달, #3 ClassVar 명시, #4 optional-dependencies 전환, #8 py.typed 추가
- **거부**: #2 LangChain 동일 구조 유지, #6 사소, #7 LangChain 동일 패턴, #9 YAGNI

### 수정 내역
- `base.py`: _split_into_pages, _split_json_into_pages에 fmt 인자 추가, self.format → fmt
- `base.py`: _PAGE_SPLIT_SEPARATOR를 ClassVar[str]로 변경
- `pyproject.toml`: [dependency-groups] → [project.optional-dependencies]
- `py.typed`: 빈 마커 파일 추가
- 테스트: _split_into_pages, _split_json_into_pages 직접 호출에 fmt 인자 추가

### 테스트: 30 unit + 18 integration = 48 passed