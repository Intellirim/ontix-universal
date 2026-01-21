"""
Conversational Generator - Production Grade v2.0
대화형 응답 생성

Features:
    - 자연스러운 대화 흐름
    - 감정 인식 및 대응
    - 컨텍스트 기반 응답
    - 후속 질문 제안
    - 톤 조절
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
import re
import logging

from app.generators.base import (
    BaseGenerator,
    GeneratorType,
    GeneratorConfig,
    GenerationMetrics,
    ResponseFormatter,
    ResponseTone,
)
from app.core.context import QueryContext

logger = logging.getLogger(__name__)


# ============================================================
# Conversational Types
# ============================================================

class EmotionType(str, Enum):
    """감정 유형"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    CURIOUS = "curious"
    CONFUSED = "confused"
    FRUSTRATED = "frustrated"
    GRATEFUL = "grateful"


class ConversationIntent(str, Enum):
    """대화 의도"""
    GREETING = "greeting"
    FAREWELL = "farewell"
    THANKS = "thanks"
    QUESTION = "question"
    FEEDBACK = "feedback"
    SMALLTALK = "smalltalk"
    HELP = "help"


@dataclass
class ConversationState:
    """대화 상태"""
    turn_count: int = 0
    detected_emotion: EmotionType = EmotionType.NEUTRAL
    detected_intent: ConversationIntent = ConversationIntent.QUESTION
    topics: List[str] = field(default_factory=list)
    user_name: Optional[str] = None
    last_response_tone: ResponseTone = ResponseTone.FRIENDLY


@dataclass
class ConversationResponse:
    """대화 응답 구조"""
    message: str
    emotion: EmotionType
    follow_ups: List[str] = field(default_factory=list)
    tone: ResponseTone = ResponseTone.FRIENDLY

    def to_formatted_string(self, include_follow_ups: bool = True) -> str:
        """포맷된 응답"""
        result = self.message

        if include_follow_ups and self.follow_ups:
            result += "\n\n"
            for fu in self.follow_ups[:2]:
                result += f"- {fu}\n"

        return result


# ============================================================
# Emotion Detector
# ============================================================

class EmotionDetector:
    """감정 감지"""

    EMOTION_PATTERNS = {
        EmotionType.HAPPY: [
            r'좋아', r'최고', r'멋져', r'감사', r'고마워', r'행복',
            r'기뻐', r'love', r'great', r'awesome',
        ],
        EmotionType.CURIOUS: [
            r'\?', r'뭐', r'어떻게', r'왜', r'언제', r'어디',
            r'궁금', r'알고\s*싶',
        ],
        EmotionType.CONFUSED: [
            r'모르겠', r'헷갈', r'이해.*안', r'뭔\s*말',
            r'다시', r'confused',
        ],
        EmotionType.FRUSTRATED: [
            r'짜증', r'안\s*돼', r'왜\s*안', r'문제', r'에러',
            r'싫', r'별로', r'짜증',
        ],
        EmotionType.GRATEFUL: [
            r'감사', r'고마워', r'덕분', r'도움\s*됐',
            r'thank', r'appreciate',
        ],
    }

    @classmethod
    def detect(cls, text: str) -> EmotionType:
        """감정 감지"""
        text_lower = text.lower()
        scores = {emotion: 0 for emotion in EmotionType}

        for emotion, patterns in cls.EMOTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    scores[emotion] += 1

        # 최고 점수 감정 반환
        max_emotion = max(scores.items(), key=lambda x: x[1])
        if max_emotion[1] > 0:
            return max_emotion[0]

        return EmotionType.NEUTRAL


class IntentDetector:
    """대화 의도 감지"""

    INTENT_PATTERNS = {
        ConversationIntent.GREETING: [
            r'^안녕', r'^하이', r'^헬로', r'^hi', r'^hello',
            r'반가워', r'처음\s*뵙',
        ],
        ConversationIntent.FAREWELL: [
            r'잘\s*가', r'바이', r'bye', r'다음에', r'나중에',
            r'끝', r'그만',
        ],
        ConversationIntent.THANKS: [
            r'감사', r'고마워', r'땡큐', r'thank',
        ],
        ConversationIntent.FEEDBACK: [
            r'좋았', r'별로', r'괜찮', r'도움\s*됐', r'유용',
        ],
        ConversationIntent.HELP: [
            r'도와', r'help', r'어떻게', r'방법',
        ],
        ConversationIntent.SMALLTALK: [
            r'심심', r'뭐\s*해', r'재밌', r'놀자',
        ],
    }

    @classmethod
    def detect(cls, text: str) -> ConversationIntent:
        """의도 감지"""
        text_lower = text.lower()

        for intent, patterns in cls.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return intent

        return ConversationIntent.QUESTION


# ============================================================
# Follow-up Generator
# ============================================================

class FollowUpGenerator:
    """후속 질문 생성"""

    FOLLOW_UPS = {
        'product': [
            "다른 제품도 추천해 드릴까요?",
            "가격대나 성분에 대해 더 알고 싶으신가요?",
        ],
        'analytics': [
            "특정 기간의 데이터를 더 보시겠어요?",
            "다른 지표도 분석해 드릴까요?",
        ],
        'general': [
            "더 궁금하신 점이 있으신가요?",
            "추가로 도움이 필요하시면 말씀해 주세요.",
        ],
    }

    @classmethod
    def generate(cls, context: QueryContext) -> List[str]:
        """후속 질문 생성"""
        question_type = context.question_type
        if hasattr(question_type, 'value'):
            qtype = question_type.value
        else:
            qtype = str(question_type)

        # 질문 타입별 후속 질문
        if 'product' in qtype or 'recommendation' in qtype:
            return cls.FOLLOW_UPS['product']
        elif 'analytics' in qtype or 'insight' in qtype:
            return cls.FOLLOW_UPS['analytics']

        return cls.FOLLOW_UPS['general']


# ============================================================
# Conversational Generator
# ============================================================

class ConversationalGenerator(BaseGenerator):
    """
    대화형 생성기 - Production Grade

    Features:
        - 자연스러운 대화
        - 감정 인식 및 공감
        - 톤 자동 조절
        - 후속 질문 제안
        - 컨텍스트 유지
    """

    # 기본 설정
    DEFAULT_TEMPERATURE = 0.8
    DEFAULT_MODEL = "full"
    DEFAULT_MAX_TOKENS = 1500

    def __init__(self, brand_config: Dict):
        super().__init__(brand_config, GeneratorType.CONVERSATIONAL)

        # Conversational 전용 설정
        self.config.temperature = self.DEFAULT_TEMPERATURE
        self.config.model_variant = self.DEFAULT_MODEL
        self.config.max_tokens = self.DEFAULT_MAX_TOKENS

        # 대화 설정
        self.include_follow_ups = True
        self.adapt_tone = True
        self.max_history_turns = 5

        # 상태
        self._conversation_state = ConversationState()

    def generate(self, context: QueryContext) -> str:
        """
        대화형 응답 생성

        Args:
            context: 쿼리 컨텍스트

        Returns:
            생성된 응답
        """
        metrics = self._create_metrics(context)

        try:
            # 감정 및 의도 감지
            emotion = EmotionDetector.detect(context.question)
            intent = IntentDetector.detect(context.question)

            # 상태 업데이트
            self._update_state(context, emotion, intent)

            # 특수 의도 처리
            special_response = self._handle_special_intent(intent, context)
            if special_response:
                metrics.complete()
                self._last_metrics = metrics
                return special_response

            # 프롬프트 로드
            system_prompt = self._load_system_prompt(context.question_type)
            user_prompt = self._build_user_prompt(context, emotion, intent)

            context.set_prompts(system_prompt, user_prompt)

            # LLM 호출
            response = self._invoke_llm(user_prompt, system_prompt, metrics)

            # 응답이 비어있으면 컨텍스트 기반 기본 응답 생성
            if not response or len(response.strip()) < 10:
                logger.warning(f"LLM returned empty or very short response: '{response}'")
                # 검색 결과가 있으면 첫 번째 항목 요약
                if context.retrieval_results and context.get_total_retrieval_count() > 0:
                    for result in context.retrieval_results:
                        items = getattr(result, 'data', None) or getattr(result, 'items', None) or []
                        for item in items[:1]:
                            if isinstance(item, dict) and item.get('content'):
                                content = item['content'][:300]
                                response = f"다음과 같은 정보를 찾았습니다:\n\n{content}..."
                                break
                        if response and len(response) > 50:
                            break

            # 여전히 비어있으면 기본 응답
            if not response or len(response.strip()) < 10:
                response = f"{self.brand_name}에 대해 무엇이든 물어보세요."

            # 후속 질문 추가
            if self.include_follow_ups:
                follow_ups = FollowUpGenerator.generate(context)
                if follow_ups and len(response) < 500:
                    response += f"\n\n{follow_ups[0]}"

            # 메트릭 완료
            metrics.complete()
            self._last_metrics = metrics

            return response

        except Exception as e:
            logger.error(f"Conversational generation error: {e}", exc_info=True)
            metrics.complete(success=False, error=str(e))
            self._last_metrics = metrics

            # 에러 발생 시에도 검색 결과가 있으면 사용
            if context.retrieval_results and context.get_total_retrieval_count() > 0:
                for result in context.retrieval_results:
                    items = getattr(result, 'data', None) or getattr(result, 'items', None) or []
                    for item in items[:1]:
                        if isinstance(item, dict) and item.get('content'):
                            content = item['content'][:300]
                            return f"관련 정보를 찾았습니다:\n\n{content}...\n\n더 궁금하신 점이 있으신가요?"

            return f"안녕하세요! {self.brand_name}입니다. 무엇을 도와드릴까요?"

    def generate_structured(self, context: QueryContext) -> ConversationResponse:
        """
        구조화된 대화 응답 생성

        Args:
            context: 쿼리 컨텍스트

        Returns:
            ConversationResponse 객체
        """
        emotion = EmotionDetector.detect(context.question)
        message = self.generate(context)
        follow_ups = FollowUpGenerator.generate(context)

        return ConversationResponse(
            message=message,
            emotion=emotion,
            follow_ups=follow_ups,
            tone=self.config.response_tone
        )

    def _build_user_prompt(
        self,
        context: QueryContext,
        emotion: EmotionType,
        intent: ConversationIntent
    ) -> str:
        """사용자 프롬프트 구성"""
        # 대화 히스토리
        history_str = self._format_history(context)

        # 컨텍스트 정보 (있으면)
        context_str = self._format_context_info(context)

        # 감정/의도 힌트
        emotional_hint = self._get_emotional_guidance(emotion)

        # 프롬프트 구성
        has_context = bool(context_str.strip())

        # 브랜드 정보
        brand_desc = self.brand_config.get('brand', {}).get('description', '')
        brand_industry = self.brand_config.get('brand', {}).get('industry', '')

        if history_str:
            prompt = f"""You are an AI assistant for {self.brand_name}.
Brand Description: {brand_desc}
Industry: {brand_industry}

Conversation History:
{history_str}

Current Question: {context.question}
User Emotion: {emotion.value}
{emotional_hint}
{context_str}

Instructions:
- Continue the conversation naturally
- If Relevant Information is provided, use it to enhance your answer
- If no context is available, use your knowledge about {brand_industry} to provide helpful information
- Do NOT say generic greetings like "안녕하세요! 무엇을 도와드릴까요?" when the user asks a specific question
- Be {self.config.response_tone.value} and helpful
- Answer in Korean with substantive content

Response:"""
        else:
            if has_context:
                prompt = f"""You are an AI assistant for {self.brand_name}.
Brand Description: {brand_desc}
Industry: {brand_industry}

Question: {context.question}
User Emotion: {emotion.value}
{emotional_hint}
{context_str}

Instructions:
- Answer the question using the Relevant Information provided above
- Enhance the answer with your knowledge about {brand_industry} if needed
- Do NOT respond with generic greetings - provide informative content
- Synthesize the information into a helpful, detailed answer
- Be {self.config.response_tone.value} and natural
- Answer in Korean

Response:"""
            else:
                prompt = f"""You are an AI assistant for {self.brand_name}.
Brand Description: {brand_desc}
Industry: {brand_industry}

Question: {context.question}
User Emotion: {emotion.value}
{emotional_hint}

Instructions:
- Answer the question using your knowledge about {brand_industry}
- Since no specific data is available, provide general but helpful information
- Do NOT say "안녕하세요! 무엇을 도와드릴까요?" when the user asked a specific question
- Be informative and helpful based on the brand's expertise in {brand_industry}
- Be {self.config.response_tone.value} and natural
- Answer in Korean with substantive content

Response:"""

        return prompt

    def _format_history(self, context: QueryContext) -> str:
        """대화 히스토리 포맷팅"""
        if not context.conversation_history:
            return ""

        parts = []
        recent = context.conversation_history[-self.max_history_turns:]

        for msg in recent:
            if hasattr(msg, 'role'):
                role = msg.role.upper()
                content = msg.content
            elif isinstance(msg, dict):
                role = msg.get('role', 'user').upper()
                content = msg.get('content', '')
            else:
                continue

            # 긴 내용은 잘라내기
            if len(content) > 200:
                content = content[:200] + "..."

            parts.append(f"{role}: {content}")

        return "\n".join(parts)

    def _format_context_info(self, context: QueryContext) -> str:
        """컨텍스트 정보 포맷팅"""
        if not context.retrieval_results:
            return ""

        total = context.get_total_retrieval_count()
        if total == 0:
            return ""

        # 간단한 컨텍스트 요약 - data 또는 items 속성 확인
        sources = []
        for r in context.retrieval_results:
            items = getattr(r, 'items', None) or getattr(r, 'data', None)
            if items:
                sources.append(r.source)

        if not sources:
            return ""

        # 실제 검색 결과 내용 포함
        context_parts = [f"\nAvailable Context: {total} items from {', '.join(sources)}"]
        context_parts.append("\nRelevant Information:")

        for result in context.retrieval_results:
            # data 또는 items 속성 확인
            items = getattr(result, 'items', None) or getattr(result, 'data', None) or []
            if not items:
                continue
            for item in items[:5]:  # 각 소스에서 최대 5개
                # 딕셔너리 또는 RetrievalItem 객체 처리
                if isinstance(item, dict):
                    node_type = item.get('node_type', 'info')
                    content = item.get('content', '')
                else:
                    node_type = getattr(item, 'node_type', 'info')
                    content = getattr(item, 'content', '')

                if content:
                    content = content[:200] if len(content) > 200 else content
                    context_parts.append(f"- [{node_type}] {content}")

        return "\n".join(context_parts)

    def _get_emotional_guidance(self, emotion: EmotionType) -> str:
        """감정별 응답 가이드"""
        guidance = {
            EmotionType.HAPPY: "User seems happy - match their positive energy!",
            EmotionType.CURIOUS: "User is curious - be informative and engaging.",
            EmotionType.CONFUSED: "User seems confused - be clear and patient.",
            EmotionType.FRUSTRATED: "User seems frustrated - be empathetic and helpful.",
            EmotionType.GRATEFUL: "User is grateful - acknowledge warmly.",
            EmotionType.NEUTRAL: "Maintain a friendly, helpful tone.",
        }
        return guidance.get(emotion, "")

    def _handle_special_intent(
        self,
        intent: ConversationIntent,
        context: QueryContext
    ) -> Optional[str]:
        """특수 의도 처리 (빠른 응답)"""
        question = context.question.strip().lower()

        # 질문에 실제 내용이 있으면 특수 의도 처리 건너뛰기
        # 예: "안녕 노화에 대해 알려줘" -> 인사 처리 안 함
        substantive_keywords = ['about', 'what', 'how', 'why', 'tell', 'explain', 'describe',
                                '알려', '설명', '뭐', '무엇', '어떻게', '왜', '대해']
        has_substantive_content = any(kw in question for kw in substantive_keywords)

        if intent == ConversationIntent.GREETING:
            # 실제 질문이 포함되어 있으면 인사 응답 건너뛰기
            if has_substantive_content or len(question) > 20:
                return None

            brand_name = self.brand_name
            greetings = [
                f"안녕하세요! {brand_name}입니다. 무엇을 도와드릴까요?",
                f"반갑습니다! {brand_name}에 오신 것을 환영해요. 어떻게 도와드릴까요?",
            ]
            return greetings[0]

        elif intent == ConversationIntent.FAREWELL:
            return "감사합니다! 다음에 또 찾아주세요. 좋은 하루 보내세요!"

        elif intent == ConversationIntent.THANKS:
            return "천만에요! 도움이 되었다니 기쁩니다. 더 궁금하신 점 있으시면 언제든 물어보세요!"

        return None

    def _update_state(
        self,
        context: QueryContext,
        emotion: EmotionType,
        intent: ConversationIntent
    ):
        """대화 상태 업데이트"""
        self._conversation_state.turn_count += 1
        self._conversation_state.detected_emotion = emotion
        self._conversation_state.detected_intent = intent

        # 톤 자동 조절
        if self.adapt_tone:
            if emotion == EmotionType.FRUSTRATED:
                self.config.response_tone = ResponseTone.EMPATHETIC
            elif emotion == EmotionType.HAPPY:
                self.config.response_tone = ResponseTone.CASUAL
            else:
                self.config.response_tone = ResponseTone.FRIENDLY

    def get_conversation_state(self) -> Dict[str, Any]:
        """현재 대화 상태"""
        return {
            'turn_count': self._conversation_state.turn_count,
            'emotion': self._conversation_state.detected_emotion.value,
            'intent': self._conversation_state.detected_intent.value,
            'current_tone': self.config.response_tone.value,
        }

    def reset_state(self):
        """대화 상태 초기화"""
        self._conversation_state = ConversationState()
        self.config.response_tone = ResponseTone.FRIENDLY

    def get_debug_info(self) -> Dict[str, Any]:
        """디버그 정보"""
        base_info = super().get_debug_info()
        base_info.update({
            'include_follow_ups': self.include_follow_ups,
            'adapt_tone': self.adapt_tone,
            'conversation_state': self.get_conversation_state(),
        })
        return base_info
