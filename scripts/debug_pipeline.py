"""
Debug script for Pipeline retrieval
"""
import sys
sys.path.insert(0, 'c:/Users/hanso/ontix-universal')

import logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

from app.services.platform.config_manager import ConfigManager
from app.core.context import QueryContext
from app.core.pipeline import Pipeline

# 브랜드 설정 로드
config_manager = ConfigManager()
brand_config = config_manager.load_brand_config('futurebiofficial')

print("=" * 60)
print("Brand Config - Retrieval Section:")
print("=" * 60)
print(f"retrieval: {brand_config.get('retrieval', {})}")

# Pipeline 생성
pipeline = Pipeline(brand_config)

print("\n" + "=" * 60)
print("Pipeline Debug Info:")
print("=" * 60)
print(f"Retrievers: {list(pipeline.retrievers.keys())}")
print(f"Generators: {list(pipeline.generators.keys())}")

# QueryContext 생성
question = "노화 방지에 대해 알려줘"
context = QueryContext(
    brand_id='futurebiofficial',
    question=question,
    conversation_history=[]
)
context.set_question_type('advisor')  # advisor 타입으로 설정

print("\n" + "=" * 60)
print(f"Testing retrieval with question: {question}")
print("=" * 60)

# Retrieval 실행
context = pipeline._retrieve(context)

print(f"\nTotal retrieval count: {context.get_total_retrieval_count()}")
print(f"Retrieval results: {len(context.retrieval_results)}")

for i, result in enumerate(context.retrieval_results):
    print(f"\n--- Result {i+1} ---")
    print(f"Source: {result.source}")
    print(f"Items: {len(result.items) if hasattr(result, 'items') else 'N/A'}")
    if hasattr(result, 'items') and result.items:
        for j, item in enumerate(result.items[:3]):  # 처음 3개만
            print(f"  Item {j+1}: {item.content[:100] if hasattr(item, 'content') else item}...")

# GraphRetriever 직접 테스트
print("\n" + "=" * 60)
print("Direct GraphRetriever Test:")
print("=" * 60)

graph_retriever = pipeline.retrievers['graph']

# 새 context로 테스트
test_context = QueryContext(
    brand_id='futurebiofficial',
    question=question,
    conversation_history=[]
)

result = graph_retriever._do_retrieve(test_context)
print(f"Source: {result.source}")
print(f"Items found: {len(result.items)}")
print(f"Metadata: {result.metadata}")

if result.items:
    for i, item in enumerate(result.items[:5]):
        print(f"  {i+1}. [{item.node_type}] {item.id}: {item.content[:80]}...")
