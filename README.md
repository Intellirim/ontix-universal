### **README.md**

```markdown
# 🚀 ONTIX Universal Platform

완전 범용 멀티 브랜드 AI 플랫폼

## ✨ Features

- 🎯 **브랜드 독립적**: 하나의 코드베이스로 무한대 브랜드 지원
- 🔧 **설정 기반**: YAML 파일로 3분 안에 브랜드 추가
- 🧠 **RAG 파이프라인**: Retrieval + Generation 자동화
- 📊 **벡터 검색**: Neo4j + OpenAI Embeddings
- ⚡ **고성능**: Redis 캐싱, 연결 풀링
- 🔌 **확장 가능**: 플러그인 방식 Feature 시스템

## 🏗️ Architecture

```
configs/brands/*.yaml  → ConfigManager → UniversalEngine
                                              ↓
                                         QuestionRouter
                                              ↓
                                          Pipeline
                                       ↙          ↘
                                  Retrievers    Generators
                                       ↓             ↓
                                   Neo4j/Vector    LLM
```

## 📦 Installation

```bash
# 1. Clone
git clone https://github.com/your-org/ontix-universal.git
cd ontix-universal

# 2. 환경 설정
cp .env.example .env
# .env 파일 수정

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 실행
python app/main.py
```

## 🐳 Docker

```bash
# 전체 스택 실행 (Neo4j + Redis + App)
docker-compose up -d

# 로그 확인
docker-compose logs -f app

# 종료
docker-compose down
```

## 🎯 Quick Start

### 1. 브랜드 추가

```bash
# 템플릿 복사
cp configs/brands/_template.yaml configs/brands/mybrand.yaml

# YAML 수정
vim configs/brands/mybrand.yaml
```

### 2. API 호출

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "brand_id": "mybrand",
    "message": "안녕하세요"
  }'
```

## 🌐 SNS Data Pipeline

범용 SNS 데이터 수집 및 지식그래프 생성 파이프라인이 포함되어 있습니다.

### 지원 플랫폼
- Instagram
- YouTube
- TikTok
- Twitter/X

### 사용법

```bash
# 테스트 실행
python scripts/test_pipeline.py

# Instagram 데이터 수집
python scripts/sync_sns.py \
  --platform instagram \
  --actor-id apify/instagram-scraper \
  --username travel_photographer \
  --limit 10

# YouTube 데이터 수집
python scripts/sync_sns.py \
  --platform youtube \
  --actor-id apify/youtube-scraper \
  --video-id dQw4w9WgXcQ \
  --limit 10
```

### 아키텍처

```
Apify Crawler → Adapter → LLM Processor → Neo4j Repository
     ↓              ↓           ↓              ↓
  Raw Data    Common DTO   Knowledge Graph  Graph DB
```

## 📚 Documentation

- [브랜드 가이드](docs/brand_guide.md)
- [기능 가이드](docs/feature_guide.md)
- [설정 레퍼런스](docs/config_reference.md)
- [SNS 파이프라인 가이드](docs/sns_pipeline.md)

## 🛠️ Development

```bash
# 테스트
pytest

# 코드 포맷팅
black app/

# Linting
flake8 app/
```

## 📄 License

MIT License

## 👥 Contributors

- Your Name (@yourname)
```

---