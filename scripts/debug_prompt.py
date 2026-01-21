"""
Debug script for prompt construction
"""
import sys
sys.path.insert(0, 'c:/Users/hanso/ontix-universal')

import logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

from app.services.platform.config_manager import ConfigManager
from app.core.context import QueryContext
from app.core.pipeline import Pipeline
from app.generators.conversational import ConversationalGenerator

# 브랜드 설정 로드
config_manager = ConfigManager()
brand_config = config_manager.load_brand_config('futurebiofficial')

# Pipeline 생성
pipeline = Pipeline(brand_config)

# 질문 설정
question = "역노화 루틴에 대해 알려줘"

# QueryContext 생성
context = QueryContext(
    brand_id='futurebiofficial',
    question=question,
    conversation_history=[]
)

print("=" * 60)
print(f"Question: {question}")
print("=" * 60)

# Retrieval 실행
context = pipeline._retrieve(context)

print(f"\nTotal retrieval count: {context.get_total_retrieval_count()}")
print(f"Retrieval sources: {context.get_retrieval_sources()}")

# ConversationalGenerator 생성
gen = ConversationalGenerator(brand_config)

# _format_context_info 테스트
context_info = gen._format_context_info(context)

# Build prompt 테스트
from app.generators.conversational import EmotionType, ConversationIntent, IntentDetector, EmotionDetector

emotion = EmotionDetector.detect(question)
intent = IntentDetector.detect(question)

user_prompt = gen._build_user_prompt(context, emotion, intent)

# 파일로 저장
with open('debug_output.txt', 'w', encoding='utf-8') as f:
    f.write(f"Question: {question}\n")
    f.write(f"Retrieval count: {context.get_total_retrieval_count()}\n")
    f.write(f"Emotion: {emotion}, Intent: {intent}\n\n")
    f.write("=== Context Info ===\n")
    f.write(context_info[:2000] if len(context_info) > 2000 else context_info)
    f.write("\n\n=== User Prompt ===\n")
    f.write(user_prompt[:3000] if len(user_prompt) > 3000 else user_prompt)

print(f"Output saved to debug_output.txt")
print(f"Prompt length: {len(user_prompt)} chars")
print(f"Context info length: {len(context_info)} chars")
