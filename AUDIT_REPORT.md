# ONTIX Universal — Audit Report

**Date**: 2026-02-07
**Auditor**: Claude Code (Opus 4.6)
**Project**: ONTIX Universal — Social Media to Knowledge Graph Pipeline
**Repo Root**: `C:\Users\hanso\ontix-universal`

---

## 1. Executive Summary

| Metric | Value |
|--------|-------|
| **프로젝트 전체 완성도** | **78%** |
| **오픈소스 공개 준비도** | **25%** |
| **Critical 이슈** | **6개** |
| **High 이슈** | **7개** |
| **Medium 이슈** | **9개** |
| **Low 이슈** | **5개** |

**핵심 판단**: 코어 파이프라인(Crawl→Transform→Filter→Process→Save)과 RAG 엔진은 **프로덕션 수준**으로 잘 구현되어 있습니다. 그러나 **보안 취약점**(민감정보 노출, 약한 패스워드 해싱)과 **파일 포맷 오류**(markdown 코드펜스 래핑)가 오픈소스 공개를 가로막는 최대 장벽입니다.

---

## 2. 프로젝트 구조

```
ontix-universal/
├── app/                          # 메인 애플리케이션
│   ├── main.py                   # FastAPI 엔트리포인트
│   ├── api/v1/                   # REST API 엔드포인트 (15개 라우터)
│   ├── core/                     # 핵심 엔진 (Pipeline, Engine, Auth, Security)
│   ├── data_pipeline/            # SNS 데이터 파이프라인 (★ 핵심)
│   │   ├── adapters/             # 플랫폼별 어댑터 (Instagram/YouTube/TikTok/Twitter)
│   │   ├── crawlers/             # Apify 크롤러 클라이언트
│   │   ├── processors/           # LLM 기반 Knowledge Graph 생성
│   │   ├── repositories/         # Neo4j 저장소
│   │   └── domain/               # 도메인 모델 (DTO)
│   ├── features/                 # 플러그인 기능 모듈 (6개)
│   ├── filters/                  # 응답 품질 필터 (Quality/Relevance/Trust)
│   ├── generators/               # LLM 응답 생성기 (5종)
│   ├── retrievers/               # RAG 검색기 (6종)
│   ├── services/                 # 공유 서비스 (LLM, Neo4j, Redis, Vector)
│   ├── models/                   # Pydantic 모델
│   └── interfaces/               # 추상 인터페이스
├── configs/                      # YAML 설정
│   ├── brands/                   # 브랜드별 설정 (3개 + 템플릿)
│   └── platform/                 # 플랫폼 설정
├── prompts/                      # LLM 프롬프트 템플릿
├── scripts/                      # 유틸리티 스크립트 (14개)
├── tests/                        # 테스트 (unit + integration)
├── lambda/                       # AWS Lambda 함수
├── data/                         # 데이터 파일
└── logs/                         # 로그 파일
```

**Tech Stack**: Python 3.11, FastAPI, Neo4j, OpenAI (GPT-5-mini), LangChain, Apify, Redis, Pydantic

---

## 3. 파이프라인 상태

### 3.1 SNS Data Pipeline (CRAWL → TRANSFORM → FILTER → PROCESS → SAVE)

| Stage | 파일 | 완성도 | 주요 이슈 |
|-------|------|--------|-----------|
| **CRAWL** | `data_pipeline/crawlers/apify_client.py` | **95%** | Apify SDK 기반, 4 플랫폼 지원. Actor ID 하드코딩 |
| **TRANSFORM** | `data_pipeline/adapters/*.py` | **90%** | 4 플랫폼 어댑터 완성. bare `except:` 다수 |
| **FILTER** | `data_pipeline/pipeline.py` (FilterStageExecutor) | **90%** | Neo4j 기반 중복 필터링. 트랜잭션 관리 미비 |
| **PROCESS** | `data_pipeline/processors/llm_processor.py` | **95%** | LLM 기반 KG 생성, 재시도 로직, 비용 추적 |
| **SAVE** | `data_pipeline/repositories/neo4j_repo.py` | **90%** | 파라미터화된 Cypher, 배치 저장. 트랜잭션 미비 |

### 3.2 RAG Pipeline (Retrieval → Generation)

| Component | 파일 | 완성도 | 비고 |
|-----------|------|--------|------|
| **Engine** | `core/engine.py` | **95%** | 싱글톤, 미들웨어, 폴백, 검증 |
| **Pipeline** | `core/pipeline.py` | **95%** | 병렬/순차 Retrieval, 훅 시스템, 추적 |
| **Routing** | `core/routing.py` | **90%** | 질문 타입 분류 |
| **Retrievers** | `retrievers/*.py` (6종) | **90%** | Graph, Vector, Stats, Product, Hybrid, Tavily |
| **Generators** | `generators/*.py` (5종) | **90%** | Factual, Insight, Conversational, Advisor, Recommendation |
| **Filters** | `filters/*.py` (4파일) | **85%** | Quality, Relevance, Trust, Validation |

### 3.3 지원 플랫폼

| Platform | Crawl | Transform | KG Generation | Status |
|----------|-------|-----------|---------------|--------|
| Instagram | ✅ | ✅ | ✅ | **Stable** |
| YouTube | ✅ | ✅ | ✅ | **Stable** |
| TikTok | ✅ | ✅ | ✅ | **Stable** |
| Twitter/X | ✅ | ✅ | ✅ | **Stable** |

### 3.4 기타 기능

| Feature | Status | 비고 |
|---------|--------|------|
| RAG 챗봇 | ✅ 정상 | 멀티브랜드 지원, 스트리밍 |
| AI Advisor | ✅ 정상 | 비즈니스 인사이트 생성 |
| Content Generation | ✅ 정상 | SNS 콘텐츠 자동 생성 |
| Social Monitoring | ✅ 정상 | 감성 분석 포함 |
| Product Recommendation | ✅ 정상 | KG 기반 추천 |
| Analytics | ✅ 정상 | 이벤트 추적, 메트릭 |
| 시각화/대시보드 | ❌ 미구현 | 프론트엔드 없음 (API만) |
| CLI 인터페이스 | ⚠️ 부분완성 | data_pipeline에만 CLI |
| 테스트 코드 | ⚠️ 부분완성 | 4개 unit + 1개 integration |

---

## 4. Critical Issues (반드시 공개 전 수정)

### C-1. `.env` 파일에 실제 API 키 노출 (DISK)
- **파일**: `.env` (디스크에만 존재, git 미추적)
- **내용**: OpenAI API Key (`sk-proj-3N27...`), Neo4j 비밀번호, Apify 토큰, Postgres 비밀번호
- **위험도**: `.gitignore`에 포함되어 git에는 없지만, 로컬 디스크에 평문으로 노출
- **조치**: API 키 즉시 순환(rotate), 기존 키 폐기 필요

### C-2. `data/users.json`이 git에 추적됨 (패스워드 해시 포함)
- **파일**: `data/users.json` (git tracked)
- **내용**: admin 사용자 패스워드 해시, 이메일, 역할 정보
- **위험도**: SHA-256 해시는 레인보우 테이블로 쉽게 복원 가능
- **조치**: git history에서 제거 필요 (`git filter-branch` 또는 BFG)

### C-3. 패스워드 해싱에 SHA-256 단순 해시 사용 (솔트 없음)
- **파일**: `app/core/auth.py:167`
- **코드**: `hashlib.sha256(password.encode()).hexdigest()`
- **위험도**: 레인보우 테이블 공격에 취약, 현대 보안 기준 미달
- **조치**: `bcrypt` 또는 `argon2`로 교체 필수

### C-4. 하드코딩된 기본 비밀번호
- **파일**: `app/core/auth.py:182, 208`
- **내용**: 기본 admin 비밀번호, 브랜드별 기본 비밀번호 패턴 (`{brand_id.capitalize()}@2026`)
- **위험도**: 공개 코드에서 비밀번호 패턴 노출
- **조치**: 기본 비밀번호 패턴 제거, 환경변수 또는 초기 설정 프로세스로 대체

### C-5. 여러 파일이 Markdown 코드펜스로 감싸져 있어 작동 불가
- **영향 파일**:
  - `README.md` — `### **README.md**\n\n```markdown\n...`로 시작
  - `Dockerfile` — `### **Dockerfile**\n\n```dockerfile\n...`로 시작
  - `docker-compose.yml` — `### **docker-compose.yml**\n\n```yaml\n...`로 시작
  - `docker-compose-fixed.yml` — 동일 문제 추정
  - `.env.example` — `### **.env.example**\n\n```bash\n...`로 시작
  - `.env.production` — `### **.env.production**\n\n```bash\n...`로 시작
- **위험도**: Docker 빌드 실패, README 깨짐, 설치 가이드 작동 불가
- **조치**: 모든 파일에서 markdown 래핑 제거, 순수 내용만 남기기

### C-6. Rate Limiting & Security Headers 비활성화
- **파일**: `app/main.py:141-145`
- **코드**: `# app.add_middleware(RateLimitMiddleware, ...)` / `# app.add_middleware(SecurityHeadersMiddleware)`
- **위험도**: 프로덕션에서 DoS 공격, 보안 헤더 미적용
- **조치**: 프로덕션 환경에서는 반드시 활성화

---

## 5. High Priority Issues (공개 후 빠르게 수정)

### H-1. `docker-compose.yml`에 Neo4j 비밀번호 하드코딩
- **파일**: `docker-compose.yml:22, 61` — `NEO4J_PASSWORD=password`
- **조치**: `${NEO4J_PASSWORD}` 환경변수 참조로 변경

### H-2. CORS 기본값이 `*` (모든 오리진 허용)
- **파일**: `app/core/config.py:85` — `CORS_ORIGINS: str = "*"`
- **조치**: 기본값을 `http://localhost:3000`으로 변경

### H-3. JWT Secret 기본값 존재
- **파일**: `app/core/auth.py:25` — 환경변수 미설정 시 기본 시크릿 사용
- **조치**: 기본값 제거, 미설정 시 시작 실패하도록 변경

### H-4. `.env.production` 파일이 git에 추적됨
- **파일**: `.env.production` (git tracked)
- **내용**: 프로덕션 설정 템플릿이지만 구조가 노출됨
- **조치**: `.gitignore`에 추가하거나 `.env.production.example`로 이름 변경

### H-5. LICENSE 파일 누락
- **상태**: README에 "MIT License" 언급하지만 실제 `LICENSE` 파일 없음
- **조치**: MIT LICENSE 파일 생성

### H-6. `docs/` 폴더 미존재 (README에서 참조)
- **파일**: `README.md:126-129` — 4개 문서 링크가 모두 깨짐
- **조치**: 문서 작성 또는 링크 제거

### H-7. Neo4j 저장소에 트랜잭션 관리 부재
- **파일**: `app/data_pipeline/repositories/neo4j_repo.py`
- **위험도**: 멀티 스테이지 저장 시 부분 실패로 데이터 불일치 가능
- **조치**: 명시적 트랜잭션 래핑 추가

---

## 6. Medium Priority Issues

### M-1. Bare `except:` 절 다수 사용
- **파일**: `adapters/instagram.py:133,221`, `youtube.py:73`, `tiktok.py:102`, `twitter.py:104`
- **조치**: 구체적 예외 타입 (`ValueError`, `KeyError` 등) 지정

### M-2. In-memory Rate Limiting (분산 환경 미지원)
- **파일**: `app/core/security.py:124-200`
- **조치**: Redis 기반 rate limiting으로 교체

### M-3. `config_manager.py` 머지 버그
- **파일**: `app/services/platform/config_manager.py:406`
- **내용**: retrieval용 merge 함수가 generation 설정에도 사용됨
- **조치**: 별도 merge 함수 또는 범용 dict merge 사용

### M-4. Audit Action Enum 잘못된 사용
- **파일**: `app/api/v1/auth.py:206` — 사용자 생성에 `BRAND_CREATE` 사용
- **파일**: `app/api/v1/admin.py:273` — 사용자 수정에 `BRAND_UPDATE` 사용
- **조치**: 올바른 Enum 값 사용

### M-5. MD5 해시를 Trace ID로 사용
- **파일**: `app/core/pipeline.py:761`
- **조치**: `uuid4()` 또는 `secrets.token_hex()` 사용 권장

### M-6. IP Spoofing 가능성
- **파일**: `app/core/security.py:160` — `X-Forwarded-For` 헤더 검증 없이 사용
- **조치**: 신뢰할 수 있는 프록시 IP 목록 검증 추가

### M-7. 사용하지 않는 의존성
- **파일**: `requirements.txt` — `psycopg2-binary`, `sqlalchemy` (PostgreSQL 현재 미사용)
- **조치**: 실제 사용하는 의존성만 남기기

### M-8. 불필요한 파일 존재
- `response.json`, `response_start.json`, `response_start2.json`, `response_stop.json` — 디버그/테스트 잔재
- `nul` — Windows 리다이렉트 잔재
- `docker-compose-fixed.yml`, `Dockerfile-clean` — 중복 파일
- `logs/audit.log` — 로그 파일
- **조치**: `.gitignore`에 추가 및 삭제

### M-9. 캐시 접근 시 레이스 컨디션
- **파일**: `app/core/engine.py:649+` — 캐시 읽기/쓰기에 락 없음
- **조치**: 스레드 안전한 캐시 접근 구현

---

## 7. Low Priority Issues

### L-1. 필터 시스템 한국어 전용
- **파일**: `app/filters/quality.py:554` — 영어 분석 미구현
- **조치**: 다국어 지원 추가

### L-2. Trust Filter 오탐 가능성
- **파일**: `app/filters/trust.py:380-381` — 키워드 기반 모순 감지
- **조치**: 시맨틱 유사도 기반으로 개선

### L-3. prompts/ 폴더 구조 불일치
- `prompts/raceon/`, `prompts/serkan/` — 브랜드별 프롬프트가 남아있음 (공개 시 제거 필요)

### L-4. `.env.example` 포맷 오류
- `.env.example`이 마크다운으로 래핑되어 복사 시 작동하지 않음

### L-5. 테스트 커버리지 부족
- unit 테스트 4개, integration 1개만 존재
- 크롤러, API 엔드포인트, 인증 테스트 없음

---

## 8. 오픈소스 공개 전 TODO 체크리스트

- [ ] **C-1**: .env 파일의 API 키 순환(rotate) — OpenAI, Neo4j, Apify
- [ ] **C-2**: `data/users.json` git history에서 완전 제거 (BFG Repo-Cleaner)
- [ ] **C-3**: bcrypt/argon2 패스워드 해싱으로 교체
- [ ] **C-4**: 하드코딩된 기본 비밀번호 패턴 제거
- [ ] **C-5**: Markdown 코드펜스 래핑 제거 (README.md, Dockerfile, docker-compose.yml, .env.example, .env.production)
- [ ] **C-6**: Rate Limiting 활성화 (환경변수 기반 토글)
- [ ] **H-1**: docker-compose.yml 비밀번호 환경변수화
- [ ] **H-2**: CORS 기본값 수정
- [ ] **H-4**: `.env.production` → `.env.production.example`
- [ ] **H-5**: MIT LICENSE 파일 추가
- [ ] **H-6**: docs/ 참조 제거 또는 최소 문서 작성
- [ ] **M-7**: 불필요한 의존성 제거
- [ ] **M-8**: 디버그/테스트 잔재 파일 삭제
- [ ] **L-3**: `prompts/raceon/`, `prompts/serkan/` 제거 (특정 고객 데이터)
- [ ] `.gitignore` 보완: `data/users.json`, `.env.production`, `*.json` (response 파일들)
- [ ] 새 README.md 작성 (아래 초안 참고)
- [ ] `requirements.txt` 정리 (미사용 패키지 제거)
- [ ] 기본 실행 테스트 통과 확인

---

## 9. README.md 초안

> **NOTE**: 현재 README.md는 markdown 코드펜스로 감싸져 있어 깨져 보입니다. 아래 내용으로 교체해야 합니다.

```markdown
# ONTIX Universal

> The first open-source pipeline that automatically transforms social media data into Knowledge Graphs.

## What is this?

Social media generates massive unstructured data every second — posts, comments, videos, interactions.
ONTIX Universal automatically crawls this data from multiple platforms, extracts entities and relationships
using LLMs, and builds a queryable Knowledge Graph in Neo4j. This enables semantic search, trend analysis,
and AI-powered insights that go far beyond simple keyword monitoring.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ONTIX Universal Pipeline                  │
│                                                             │
│  ┌─────────┐  ┌───────────┐  ┌────────┐  ┌─────────┐  ┌──────┐
│  │  CRAWL  │→ │ TRANSFORM │→ │ FILTER │→ │ PROCESS │→ │ SAVE │
│  │ (Apify) │  │(Adapters) │  │(Dedup) │  │  (LLM)  │  │(Neo4j│
│  └─────────┘  └───────────┘  └────────┘  └─────────┘  └──────┘
│       ↓             ↓             ↓            ↓           ↓
│   Raw JSON     Common DTO    New Items    KG Triples    Graph DB
└─────────────────────────────────────────────────────────────┘
```

## Supported Platforms

| Platform | Crawl | Transform | KG Generation | Status |
|----------|-------|-----------|---------------|--------|
| Instagram | ✅ | ✅ | ✅ | Stable |
| YouTube | ✅ | ✅ | ✅ | Stable |
| TikTok | ✅ | ✅ | ✅ | Stable |
| Twitter/X | ✅ | ✅ | ✅ | Stable |

## Quick Start

### Prerequisites
- Python 3.11+
- Neo4j 5.x (AuraDB or local)
- OpenAI API key
- Apify API token

### Installation

```bash
git clone https://github.com/AIM-Lab/ontix-universal.git
cd ontix-universal
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your credentials
```

### Run the Pipeline

```bash
# Full pipeline (Instagram example)
python -m app.data_pipeline.pipeline \
  --platform instagram \
  --brand-id mybrand \
  --brand-name "My Brand" \
  --max-items 50

# Start the API server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker-compose up -d
```

## Configuration

Copy `.env.example` to `.env` and fill in:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `NEO4J_URI` | Yes | Neo4j connection URI |
| `NEO4J_USERNAME` | Yes | Neo4j username |
| `NEO4J_PASSWORD` | Yes | Neo4j password |
| `APIFY_TOKEN` | Yes | Apify API token for crawling |
| `REDIS_HOST` | No | Redis host (optional caching) |

## Tech Stack

- **Runtime**: Python 3.11, FastAPI, uvicorn
- **Database**: Neo4j (Knowledge Graph), Redis (Cache)
- **AI/ML**: OpenAI GPT-5-mini, LangChain, text-embedding-3-small
- **Crawling**: Apify SDK (4 platform actors)
- **Auth**: JWT + RBAC

## Project Structure

```
app/
├── data_pipeline/     # Core: CRAWL → TRANSFORM → FILTER → PROCESS → SAVE
├── core/              # Engine, Pipeline, Auth, Security
├── api/v1/            # REST API endpoints
├── features/          # Plugin modules (Advisor, Analytics, etc.)
├── filters/           # Response quality filters
├── generators/        # LLM response generators
├── retrievers/        # RAG retrieval modules
└── services/          # Shared services (LLM, Neo4j, Redis)
```

## Roadmap

- [ ] Web UI Dashboard
- [ ] REST API documentation (OpenAPI)
- [ ] Multi-language NER support
- [ ] Wikidata entity linking
- [ ] HTTPS / rate limiting hardening
- [ ] Comprehensive test suite

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License — see [LICENSE](LICENSE) for details.

## About

Built by **AIM Lab** — AI Entity Architecture Agency
```

---

## 10. 의존성 분석

### 필수 외부 서비스 (API 키 필요)
| Service | 용도 | 환경변수 |
|---------|------|----------|
| **OpenAI** | LLM (GPT-5-mini), Embeddings | `OPENAI_API_KEY` |
| **Neo4j** | Knowledge Graph DB | `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` |
| **Apify** | SNS 크롤링 | `APIFY_TOKEN` |

### 선택적 서비스
| Service | 용도 | 환경변수 |
|---------|------|----------|
| Redis | 캐시, Rate Limiting | `REDIS_HOST` |
| Tavily | 웹 검색 보강 | `TAVILY_API_KEY` |
| PostgreSQL | 채팅 저장 (현재 미사용) | `POSTGRES_*` |

### Python 패키지 (핵심)
| Package | Version | 용도 |
|---------|---------|------|
| fastapi | >=0.109.0 | Web framework |
| neo4j | >=5.15.0 | Graph DB driver |
| langchain | >=1.2.0 | LLM orchestration |
| langchain-openai | >=1.1.5 | OpenAI integration |
| openai | >=2.3.0 | OpenAI API |
| apify-client | >=1.7.1 | Apify SDK |
| pydantic | >=2.7.4 | Data validation |
| redis | >=5.0.1 | Cache client |

### 불필요한 패키지 (제거 권장)
| Package | 이유 |
|---------|------|
| `psycopg2-binary` | PostgreSQL 현재 미사용 |
| `sqlalchemy` | ORM 현재 미사용 |
| `prometheus-client` | Prometheus 통합 미구현 |

---

## 11. 권장 개선사항 (선택)

1. **bcrypt 패스워드 해싱**: `pip install bcrypt` 추가, `auth.py` 수정
2. **Alembic DB 마이그레이션**: 사용자 데이터를 JSON → PostgreSQL/Neo4j로 이전
3. **API 문서 자동 생성**: FastAPI의 `/docs` 엔드포인트 커스터마이징
4. **CI/CD 파이프라인**: GitHub Actions로 테스트 자동화
5. **Docker multi-stage build**: 이미지 크기 최적화
6. **로그 구조화**: JSON 로거 활성화 (python-json-logger 이미 의존성에 있음)
7. **Semantic Versioning**: pyproject.toml 도입, 버전 관리 체계화
8. **pre-commit hooks**: black, flake8, isort 자동 적용

---

## 12. Git History 민감정보 경고

**다음 파일들이 git history에 한 번이라도 커밋된 적 있습니다:**

| 파일 | 커밋 | 위험도 | 내용 |
|------|------|--------|------|
| `data/users.json` | Initial commit | **CRITICAL** | 패스워드 해시, 이메일 |
| `.env.production` | Initial commit | **HIGH** | 프로덕션 설정 구조 |
| `docker-compose.yml` | Initial commit | **MEDIUM** | 기본 비밀번호 `password` |

**권장 조치**: 오픈소스 공개 전 `git filter-branch` 또는 [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/)로 민감 파일을 git history에서 완전 제거하세요. 또는 새로운 repository를 생성하여 clean 상태에서 시작하세요.

---

*Generated by Claude Code (Opus 4.6) — 2026-02-07*
