"""
Test LLM directly
"""
import sys
sys.path.insert(0, 'c:/Users/hanso/ontix-universal')

import logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

from app.services.shared.llm import get_llm_client

llm = get_llm_client()

print(f"LLM Model: {llm.config.default_model}")
print("=" * 50)

# 간단한 테스트
prompt = """You are an AI assistant for 퓨처바이오피셜.
Brand Description: 안티에이징을 대중화시켜 모두의 노화를 느리게 한다.
Industry: 안티에이징

Question: 역노화 루틴에 대해 알려줘

Relevant Information:
- [Concept] 역노화는 노화의 과정을 늦추거나 되돌리려는 노력을 의미하며, 건강한 생활습관과 식습관이 중요하다.
- [Content] 식사 순서의 마법: 혈당 피크 73.5% 감소. 방법: 식이섬유(채소 5분) → 단백질·지방 → 탄수화물 순서로 식사하세요.

Instructions:
- Answer the question using the Relevant Information provided above
- Be informative and helpful
- Answer in Korean

Response:"""

print("Sending request to LLM...")
response = llm.invoke(prompt, model_variant="full")

print(f"\n=== Response ({len(response)} chars) ===")
print(response)
