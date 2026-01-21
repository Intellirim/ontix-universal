"""
Debug script for Neo4j retrieval
"""
import sys
sys.path.insert(0, 'c:/Users/hanso/ontix-universal')

from app.services.shared.neo4j import get_neo4j_client
from app.retrievers.graph import KeywordExtractor

# Neo4j 클라이언트 연결
neo4j = get_neo4j_client()

print("=" * 50)
print("1. Neo4j Concept 데이터 확인")
print("=" * 50)

query = """
MATCH (c:Concept)
WHERE c.brand_id = 'futurebiofficial'
RETURN c.id as id, c.description as description, c.type as type
LIMIT 20
"""

results = neo4j.query(query, {})
print(f"Found {len(results)} concepts:")
for r in results:
    print(f"  - ID: {r['id']}, Type: {r.get('type')}, Desc: {r.get('description', '')[:50]}")

print("\n" + "=" * 50)
print("2. 키워드 추출 테스트")
print("=" * 50)

test_questions = [
    "노화 방지에 대해 알려줘",
    "안티에이징 제품 추천해줘",
    "피부 관리 방법 알려줘",
    "활력",
    "NAD+ 가 뭐야",
]

for q in test_questions:
    keywords = KeywordExtractor.extract(q)
    print(f"Q: {q}")
    print(f"   Keywords: {keywords}")

print("\n" + "=" * 50)
print("3. 검색 쿼리 테스트")
print("=" * 50)

# 테스트 키워드로 검색
test_keywords = ['노화', '활력', '안티에이징']

query = """
MATCH (c:Concept)
WHERE c.brand_id = $brand_id
  AND (
    any(kw IN $keywords WHERE toLower(c.id) CONTAINS kw)
    OR any(kw IN $keywords WHERE toLower(coalesce(c.description, '')) CONTAINS kw)
  )
RETURN c.id as id, c.description as description, c.type as type
LIMIT 10
"""

results = neo4j.query(query, {
    'brand_id': 'futurebiofficial',
    'keywords': test_keywords,
})

print(f"Search with keywords {test_keywords}:")
print(f"Found {len(results)} results")
for r in results:
    print(f"  - ID: {r['id']}, Type: {r.get('type')}")

print("\n" + "=" * 50)
print("4. 정확 매칭 테스트 (toLower 없이)")
print("=" * 50)

# Concept ID 목록 가져오기
id_query = """
MATCH (c:Concept)
WHERE c.brand_id = 'futurebiofficial'
RETURN c.id as id
"""
id_results = neo4j.query(id_query, {})
concept_ids = [r['id'] for r in id_results]

print(f"All Concept IDs: {concept_ids}")

# 키워드가 Concept ID에 포함되어 있는지 Python에서 확인
print("\nPython-side matching test:")
for kw in test_keywords:
    matches = [cid for cid in concept_ids if kw in cid.lower()]
    print(f"  Keyword '{kw}' matches: {matches}")

print("\n" + "=" * 50)
print("5. Content 데이터 확인")
print("=" * 50)

content_query = """
MATCH (c:Content)
WHERE c.brand_id = 'futurebiofficial'
RETURN c.id as id, substring(c.text, 0, 100) as text_preview
LIMIT 10
"""

content_results = neo4j.query(content_query, {})
print(f"Found {len(content_results)} contents:")
for r in content_results:
    print(f"  - ID: {r['id']}")
    print(f"    Text: {r.get('text_preview', 'N/A')}")
