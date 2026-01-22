"""
Question Router - Production Grade v2.0
질문 분류 및 라우팅

Features:
    - 다단계 분류 (키워드 -> 패턴 -> LLM)
    - 엔티티 추출
    - 신뢰도 점수
    - 분류 캐싱
    - 도메인 특화 패턴
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import re
import logging
import hashlib

from app.core.context import QueryContext, QuestionType

logger = logging.getLogger(__name__)


class IntentType(str, Enum):
    """의도 유형"""
    QUERY = "query"  # 정보 요청
    COMMAND = "command"  # 명령/요청
    COMPARISON = "comparison"  # 비교
    RECOMMENDATION = "recommendation"  # 추천
    EXPLANATION = "explanation"  # 설명
    GREETING = "greeting"  # 인사
    FEEDBACK = "feedback"  # 피드백
    OTHER = "other"


@dataclass
class ClassificationResult:
    """분류 결과"""
    question_type: QuestionType
    confidence: float
    intent: IntentType
    entities: Dict[str, List[str]]
    matched_patterns: List[str] = field(default_factory=list)
    method: str = "keyword"  # keyword, pattern, llm


@dataclass
class RouterConfig:
    """라우터 설정"""
    use_llm_fallback: bool = False
    min_confidence: float = 0.3
    cache_enabled: bool = True
    cache_ttl: int = 3600


class PatternMatcher:
    """패턴 기반 매칭 (범용)"""

    # 질문 유형별 정규식 패턴 (브랜드 독립적)
    PATTERNS = {
        QuestionType.PRODUCT_RECOMMENDATION: [
            r'추천.*(해줘|해주세요|알려줘|알려주세요)',
            r'(좋은|인기|베스트|최고).*(제품|상품|아이템|것)',
            r'어떤.*(제품|상품|아이템|것|거)',
            r'(뭐|무엇|어떤거).*(좋을까|살까|쓸까|할까)',
            r'(가성비|저렴|싼).*(추천|제품|상품)',
            r'(구매|구입|사고).*(싶|할까)',
        ],
        QuestionType.ANALYTICS: [
            r'(인기|좋아요|조회수|반응).*(많은|높은|top|좋은)',
            r'(통계|분석|데이터|수치)',
            r'(트렌드|트렌딩|급상승|화제)',
            r'(비교|순위|랭킹|top)',
            r'(최근|이번주|이번달|요즘).*(인기|트렌드|반응)',
            r'(성과|실적|결과).*(어때|어떻게)',
        ],
        QuestionType.ADVISOR: [
            r'어떻게.*(하면|해야|할까|할 수)',
            r'(방법|팁|가이드|조언|노하우)',
            r'(루틴|순서|단계|절차|프로세스)',
            r'(효과적|올바른|제대로|잘)',
            r'(전략|방안|계획).*(세우|알려)',
        ],
        QuestionType.FACTUAL: [
            r'^(무엇|뭐|어디|언제|누구|왜|어떤)',
            r'(정보|설명|알려|내용)',
            r'(뭐야|뭔가요|무엇인가요)',
            r'(차이|다른점|비교|구분)',
            r'(있어|있나요|있는지)',
        ],
        QuestionType.CONVERSATIONAL: [
            r'^(안녕|하이|헬로|hi|hello)',
            r'(고마워|감사|땡큐|thank)',
            r'(잘했어|좋아|최고|멋져|굿)',
            r'(잘가|바이|bye|수고)',
            r'^(네|응|그래|알겠)',
        ],
    }

    @classmethod
    def match(cls, question: str) -> Tuple[Optional[QuestionType], float, List[str]]:
        """
        패턴 매칭으로 질문 유형 분류

        Returns:
            (question_type, confidence, matched_patterns)
        """
        question_lower = question.lower()
        matches = {}

        for qtype, patterns in cls.PATTERNS.items():
            matched = []
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    matched.append(pattern)

            if matched:
                matches[qtype] = matched

        if not matches:
            return None, 0.0, []

        # 가장 많이 매칭된 유형 선택
        best_type = max(matches.items(), key=lambda x: len(x[1]))
        qtype = best_type[0]
        matched_patterns = best_type[1]

        # 신뢰도 계산 (매칭 수 기반)
        confidence = min(0.5 + len(matched_patterns) * 0.15, 0.95)

        return qtype, confidence, matched_patterns


class EntityExtractor:
    """엔티티 추출 (범용)"""

    # 범용 엔티티 패턴 (브랜드 독립적)
    ENTITY_PATTERNS = {
        'content_type': [
            r'(게시물|포스트|글|콘텐츠|피드|스토리|릴스)',
            r'(post|content|story|reel|feed)',
        ],
        'action': [
            r'(구매|주문|배송|결제|환불|교환)',
            r'(문의|상담|예약|신청)',
        ],
        'sentiment': [
            r'(좋은|나쁜|최고|최악|만족|불만)',
            r'(추천|비추천|별로)',
        ],
        'price_range': [
            r'(\d+)만원?(대|이하|이상)',
            r'(저렴|가성비|고가|프리미엄|비싼|싼)',
        ],
        'platform': [
            r'(인스타그램|인스타|유튜브|틱톡|트위터|페이스북)',
            r'(instagram|youtube|tiktok|twitter|facebook)',
        ],
        'time_period': [
            r'(오늘|어제|이번주|이번달|최근|지난주|지난달)',
            r'(today|yesterday|this week|this month|recent)',
        ],
        'quantity': [
            r'(\d+)(개|건|명|회)',
            r'(전체|모든|일부|몇)',
        ],
    }

    @classmethod
    def extract(cls, question: str) -> Dict[str, List[str]]:
        """
        질문에서 엔티티 추출

        Returns:
            엔티티 유형 -> 추출된 값 리스트
        """
        question_lower = question.lower()
        entities = {}

        for entity_type, patterns in cls.ENTITY_PATTERNS.items():
            extracted = []
            for pattern in patterns:
                matches = re.findall(pattern, question_lower)
                for match in matches:
                    if isinstance(match, tuple):
                        extracted.extend(match)
                    else:
                        extracted.append(match)

            if extracted:
                entities[entity_type] = list(set(extracted))

        return entities


class IntentDetector:
    """의도 감지"""

    INTENT_PATTERNS = {
        IntentType.RECOMMENDATION: [
            r'추천', r'좋은', r'베스트', r'어떤게', r'뭐가',
        ],
        IntentType.COMPARISON: [
            r'비교', r'차이', r'vs', r'더 좋', r'어떤게 더',
        ],
        IntentType.EXPLANATION: [
            r'뭐야', r'무엇', r'설명', r'알려줘', r'정보',
        ],
        IntentType.COMMAND: [
            r'해줘', r'해주세요', r'보여줘', r'찾아줘',
        ],
        IntentType.GREETING: [
            r'^안녕', r'^하이', r'^헬로', r'반가워',
        ],
        IntentType.FEEDBACK: [
            r'고마워', r'감사', r'좋아', r'별로', r'싫어',
        ],
    }

    @classmethod
    def detect(cls, question: str) -> IntentType:
        """의도 감지"""
        question_lower = question.lower()

        for intent, keywords in cls.INTENT_PATTERNS.items():
            for keyword in keywords:
                if re.search(keyword, question_lower):
                    return intent

        return IntentType.QUERY


class QuestionRouter:
    """
    프로덕션급 질문 라우터

    다단계 분류를 통해 정확한 질문 유형을 결정합니다.
    1. 키워드 기반 분류
    2. 패턴 기반 분류
    3. LLM 기반 분류 (선택적)
    """

    # 질문 타입별 키워드 (범용)
    TYPE_KEYWORDS = {
        QuestionType.PRODUCT_RECOMMENDATION: {
            '추천', '상품', '제품', '어떤', '뭐', '무엇', '좋은', '인기',
            '베스트', '최고', '가성비', '싼', '저렴', '구매', '구입',
        },
        QuestionType.ANALYTICS: {
            '인기', '좋아요', '많은', 'top', '순위', '베스트', '통계',
            '분석', '게시물', '포스트', '트렌드', '조회수', '성과', '실적',
        },
        QuestionType.ADVISOR: {
            '어떻게', '방법', '조언', '해야', '하면', '좋을까',
            '팁', '가이드', '순서', '전략', '노하우',
        },
        QuestionType.FACTUAL: {
            '무엇', '뭐야', '어디', '언제', '누구', '어떤', '설명',
            '정보', '알려줘', '뭔가요', '내용', '있어', '있나요',
        },
        QuestionType.CONVERSATIONAL: {
            '안녕', '고마워', '감사', '잘했어', '좋아', '최고', '멋져',
            '반가워', '잘가', '바이', '네', '응', '그래',
        },
    }

    # 분류 캐시
    _cache: Dict[str, ClassificationResult] = {}
    _cache_max_size: int = 500

    def __init__(self, brand_config: Dict):
        """
        Args:
            brand_config: 브랜드 설정
        """
        self.brand_config = brand_config
        self.brand_id = brand_config['brand']['id']
        self.enabled_features = set(brand_config.get('features', []))

        # 설정
        routing_config = brand_config.get('routing', {})
        self.config = RouterConfig(
            use_llm_fallback=routing_config.get('use_llm', False),
            min_confidence=routing_config.get('min_confidence', 0.3),
            cache_enabled=routing_config.get('cache', True),
        )

        # LLM 클라이언트 (lazy load)
        self._llm = None

    @property
    def llm(self):
        """LLM 클라이언트 lazy loading"""
        if self._llm is None:
            from app.services.shared.llm import get_llm_client
            self._llm = get_llm_client()
        return self._llm

    def route(self, context: QueryContext) -> str:
        """
        질문 라우팅

        Args:
            context: 쿼리 컨텍스트

        Returns:
            question_type 문자열
        """
        question = context.question

        # 1. 캐시 확인
        if self.config.cache_enabled:
            cached = self._get_cached(question)
            if cached:
                self._apply_result(context, cached)
                logger.info(f"Router cache hit: {cached.question_type.value}")
                return cached.question_type.value

        # 2. 분류 실행
        result = self._classify(question)

        # 3. 기능 활성화 확인
        if result.question_type.value not in self.enabled_features:
            if result.question_type != QuestionType.CONVERSATIONAL:
                logger.warning(
                    f"Type '{result.question_type.value}' not enabled, "
                    f"falling back to 'conversational'"
                )
            result.question_type = QuestionType.CONVERSATIONAL
            result.confidence = max(result.confidence * 0.5, 0.3)

        # 4. 컨텍스트 업데이트
        self._apply_result(context, result)

        # 5. 캐싱
        if self.config.cache_enabled:
            self._cache_result(question, result)

        logger.info(
            f"Routed to: {result.question_type.value} "
            f"(confidence={result.confidence:.2f}, method={result.method})"
        )

        return result.question_type.value

    def _classify(self, question: str) -> ClassificationResult:
        """다단계 분류"""
        question_lower = question.lower()

        # 1. 엔티티 추출
        entities = EntityExtractor.extract(question)

        # 2. 의도 감지
        intent = IntentDetector.detect(question)

        # 3. 패턴 기반 분류
        pattern_type, pattern_conf, matched_patterns = PatternMatcher.match(question)

        if pattern_type and pattern_conf >= self.config.min_confidence:
            return ClassificationResult(
                question_type=pattern_type,
                confidence=pattern_conf,
                intent=intent,
                entities=entities,
                matched_patterns=matched_patterns,
                method="pattern",
            )

        # 4. 키워드 기반 분류
        keyword_type, keyword_conf = self._classify_by_keywords(question_lower)

        if keyword_conf >= self.config.min_confidence:
            return ClassificationResult(
                question_type=keyword_type,
                confidence=keyword_conf,
                intent=intent,
                entities=entities,
                method="keyword",
            )

        # 5. LLM 폴백 (선택적)
        if self.config.use_llm_fallback:
            llm_type = self._classify_by_llm(question)
            if llm_type:
                return ClassificationResult(
                    question_type=llm_type,
                    confidence=0.7,
                    intent=intent,
                    entities=entities,
                    method="llm",
                )

        # 6. 기본값
        return ClassificationResult(
            question_type=QuestionType.CONVERSATIONAL,
            confidence=0.3,
            intent=intent,
            entities=entities,
            method="default",
        )

    def _classify_by_keywords(self, question: str) -> Tuple[QuestionType, float]:
        """키워드 기반 분류"""
        scores = {}

        for qtype, keywords in self.TYPE_KEYWORDS.items():
            # 활성화되지 않은 기능은 스킵
            if qtype.value not in self.enabled_features and qtype != QuestionType.CONVERSATIONAL:
                continue

            score = 0
            for keyword in keywords:
                if keyword in question:
                    score += 1

            if score > 0:
                scores[qtype] = score

        if not scores:
            return QuestionType.CONVERSATIONAL, 0.2

        # 최고 점수 타입 선택
        best_type = max(scores.items(), key=lambda x: x[1])
        qtype = best_type[0]
        match_count = best_type[1]

        # 신뢰도 계산
        confidence = min(0.3 + match_count * 0.15, 0.85)

        return qtype, confidence

    def _classify_by_llm(self, question: str) -> Optional[QuestionType]:
        """LLM 기반 분류"""
        try:
            available_types = [
                qtype.value for qtype in QuestionType
                if qtype.value in self.enabled_features or qtype == QuestionType.CONVERSATIONAL
            ]

            if not available_types:
                return None

            prompt = f"""Classify the following question into one of these types.

Question: {question}

Available types:
{', '.join(available_types)}

Rules:
- Respond with ONLY the type name, nothing else
- If unsure, use 'conversational'

Type:"""

            response = self.llm.invoke(
                prompt=prompt,
                model_variant="mini",
                temperature=0
            ).strip().lower()

            for qtype in QuestionType:
                if qtype.value in response:
                    return qtype

            return None

        except Exception as e:
            logger.error(f"LLM classification error: {e}")
            return None

    def _apply_result(self, context: QueryContext, result: ClassificationResult):
        """분류 결과를 컨텍스트에 적용"""
        context.set_question_type(result.question_type, result.confidence)
        context.intent = result.intent.value
        context.entities = result.entities
        context.add_metadata('classification_method', result.method)
        context.add_metadata('matched_patterns', result.matched_patterns)

    def _get_cache_key(self, question: str) -> str:
        """캐시 키 생성"""
        return hashlib.md5(f"{self.brand_id}:{question.lower()}".encode()).hexdigest()[:16]

    def _get_cached(self, question: str) -> Optional[ClassificationResult]:
        """캐시 조회"""
        key = self._get_cache_key(question)
        return self._cache.get(key)

    def _cache_result(self, question: str, result: ClassificationResult):
        """결과 캐싱"""
        if len(self._cache) >= self._cache_max_size:
            # 가장 오래된 항목 제거
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        key = self._get_cache_key(question)
        self._cache[key] = result

    @classmethod
    def clear_cache(cls):
        """캐시 초기화"""
        cls._cache.clear()
        logger.info("Router cache cleared")

    def get_stats(self) -> Dict:
        """통계 조회"""
        return {
            'cache_size': len(self._cache),
            'cache_max_size': self._cache_max_size,
            'enabled_features': list(self.enabled_features),
            'config': {
                'use_llm_fallback': self.config.use_llm_fallback,
                'min_confidence': self.config.min_confidence,
                'cache_enabled': self.config.cache_enabled,
            }
        }
