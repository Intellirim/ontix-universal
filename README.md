# ONTIX Universal

> The first open-source pipeline that automatically transforms social media data into Knowledge Graphs.

## What is this?

Social media generates massive unstructured data every second — posts, comments, videos, interactions. ONTIX Universal automatically crawls this data from multiple platforms, extracts entities and relationships using LLMs, and builds a queryable Knowledge Graph in Neo4j. This enables semantic search, trend analysis, and AI-powered insights that go far beyond simple keyword monitoring.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ONTIX Universal Pipeline                      │
│                                                                 │
│  ┌─────────┐  ┌───────────┐  ┌────────┐  ┌─────────┐  ┌──────┐│
│  │  CRAWL  │→ │ TRANSFORM │→ │ FILTER │→ │ PROCESS │→ │ SAVE ││
│  │ (Apify) │  │(Adapters) │  │(Dedup) │  │  (LLM)  │  │(Neo4j││
│  └─────────┘  └───────────┘  └────────┘  └─────────┘  └──────┘│
│       ↓             ↓             ↓            ↓           ↓    │
│   Raw JSON     Common DTO    New Items    KG Triples    Graph DB│
└─────────────────────────────────────────────────────────────────┘
```

Additionally, a full **RAG (Retrieval-Augmented Generation)** pipeline is built on top of the Knowledge Graph:

```
configs/brands/*.yaml  →  ConfigManager  →  UniversalEngine
                                                  ↓
                                             QuestionRouter
                                                  ↓
                                              Pipeline
                                           ↙          ↘
                                      Retrievers    Generators
                                           ↓             ↓
                                       Neo4j/Vector    LLM
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
# Crawl a specific Instagram account
python -m app.data_pipeline.pipeline \
  --platform instagram \
  --brand-id mybrand \
  --brand-name "My Brand" \
  --target @cristiano @neymarjr \
  --max-items 50

# Crawl by hashtag
python -m app.data_pipeline.pipeline \
  --platform instagram \
  --brand-id mybrand \
  --brand-name "My Brand" \
  --target "#fitness" "#gym"

# Crawl YouTube channels
python -m app.data_pipeline.pipeline \
  --platform youtube \
  --brand-id mybrand \
  --brand-name "My Brand" \
  --target @mkbhd

# Skip LLM processing (crawl + transform only)
python -m app.data_pipeline.pipeline \
  --platform twitter \
  --brand-id mybrand \
  --brand-name "My Brand" \
  --target @elonmusk \
  --skip-llm

# Start the API server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Target format:** `@username` for accounts, `#hashtag` for tags, plain text for search. If no `--target` is provided, the brand name is used as a hashtag search.

### Docker

```bash
# Set environment variables
export NEO4J_PASSWORD=your-secure-password
export OPENAI_API_KEY=sk-your-key
export APIFY_TOKEN=your-apify-token
export JWT_SECRET=$(openssl rand -hex 32)

# Start full stack (Neo4j + Redis + App)
docker-compose up -d

# View logs
docker-compose logs -f app
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
| `JWT_SECRET` | Prod | JWT signing secret |
| `REDIS_HOST` | No | Redis host (optional caching) |

### Brand Configuration

Add a new brand by creating a YAML config:

```bash
cp configs/brands/_template.yaml configs/brands/mybrand.yaml
# Edit the YAML file with brand details
```

### Customizing Entity Extraction

The pipeline uses an LLM prompt to extract entities and relationships from social media text. A generic default prompt (`prompts/default_extraction.txt`) is included and works out of the box. To improve extraction quality for your domain, write a custom prompt and point to it:

```bash
# Set custom prompt path in .env
ENTITY_EXTRACTION_PROMPT_PATH=./prompts/my_custom_extraction.txt
```

The default prompt produces usable results, but a domain-tuned prompt will significantly improve entity and relationship extraction quality.

## Tech Stack

- **Runtime**: Python 3.11, FastAPI, uvicorn
- **Database**: Neo4j (Knowledge Graph), Redis (Cache)
- **AI/ML**: OpenAI GPT-5-mini, LangChain, text-embedding-3-small
- **Crawling**: Apify SDK (4 platform actors)
- **Auth**: JWT + RBAC (bcrypt password hashing)
- **Validation**: 4-layer quality/trust/relevance/validation filters

## Project Structure

```
app/
├── data_pipeline/     # Core: CRAWL → TRANSFORM → FILTER → PROCESS → SAVE
│   ├── adapters/      # Platform-specific data parsers
│   ├── crawlers/      # Apify client wrapper
│   ├── processors/    # LLM-based KG generation
│   ├── repositories/  # Neo4j persistence
│   └── domain/        # Universal data models (DTOs)
├── core/              # Engine, Pipeline, Auth, Security
├── api/v1/            # REST API endpoints (15 routers)
├── features/          # Plugin modules (Advisor, Analytics, etc.)
├── filters/           # Response quality filters
├── generators/        # LLM response generators (5 types)
├── retrievers/        # RAG retrieval modules (6 types)
└── services/          # Shared services (LLM, Neo4j, Redis, Vector)
```

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### API Examples

**Start a crawl job (Instagram accounts):**

```bash
curl -X POST http://localhost:8000/api/v1/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{
    "brand_id": "mybrand",
    "platform": "instagram",
    "target_type": "accounts",
    "targets": ["cristiano", "neymarjr"],
    "max_items": 50
  }'
# Response: {"job_id": "a1b2c3d4", "status": "pending", "message": "..."}
```

**Start a crawl job (hashtag search):**

```bash
curl -X POST http://localhost:8000/api/v1/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{
    "brand_id": "mybrand",
    "platform": "tiktok",
    "target_type": "hashtags",
    "targets": ["fitness", "workout"],
    "max_items": 100
  }'
```

**Check crawl job status:**

```bash
curl http://localhost:8000/api/v1/pipeline/status/a1b2c3d4
# Response: {"job_id": "a1b2c3d4", "status": "completed", "statistics": {...}}
```

**List all jobs:**

```bash
curl http://localhost:8000/api/v1/pipeline/jobs?brand_id=mybrand&limit=10
```

**Query the Knowledge Graph:**

```bash
curl http://localhost:8000/api/v1/brands/mybrand/graph?limit=100
# Response: {"nodes": [...], "relationships": [...], "stats": {...}}

curl http://localhost:8000/api/v1/brands/mybrand/graph-summary
# Response: {"node_count": 342, "relationship_count": 891, "key_concepts": [...]}
```

**RAG chatbot (ask questions about your data):**

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "brand_id": "mybrand",
    "message": "What are the most popular topics this week?",
    "conversation_history": []
  }'
# Response: {"message": "Based on the data...", "retrieval_contexts": [...], "metadata": {...}}
```

**RAG chatbot with session (conversation memory):**

```bash
curl -X POST http://localhost:8000/api/v1/chat/with-session \
  -H "Content-Type: application/json" \
  -d '{
    "brand_id": "mybrand",
    "message": "Show me the top performing posts",
    "session_id": null,
    "use_history": true
  }'
# Response includes session_id for follow-up messages
```

## Neo4j Vector Index Setup

The RAG pipeline uses vector similarity search on `Concept` nodes. After running the pipeline at least once, create the vector index in Neo4j Browser (`http://localhost:7474`):

```cypher
-- Create vector index for semantic search (required for RAG)
CREATE VECTOR INDEX ontix_global_concept_index IF NOT EXISTS
FOR (c:Concept)
ON c.embedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }
};

-- Verify the index was created
SHOW VECTOR INDEXES;
```

| Setting | Value | Notes |
|---------|-------|-------|
| Index name | `ontix_global_concept_index` | Must match `NEO4J_VECTOR_INDEX` in `.env` |
| Node label | `Concept` | Keywords extracted from content |
| Property | `embedding` | Auto-populated by `text-embedding-3-small` |
| Dimensions | `1536` | OpenAI `text-embedding-3-small` output size |
| Similarity | `cosine` | Best for normalized text embeddings |

The embeddings are generated automatically when the app connects to Neo4j with a valid OpenAI API key. The vector index enables the RAG chatbot to find semantically similar concepts even when exact keywords don't match.

## Development

```bash
# Run tests
pytest

# Run specific test
pytest tests/unit/test_filters.py -v
```

## Roadmap

- [ ] Web UI Dashboard
- [ ] Comprehensive test suite
- [ ] Multi-language NER support
- [ ] Wikidata entity linking
- [ ] HTTPS / production hardening
- [ ] Webhook notifications
- [ ] Export to RDF/OWL

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
