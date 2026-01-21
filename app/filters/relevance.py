"""
Relevance Filter - Production Grade v2.0
관련성 검증 시스템

Features:
- 질문-응답 관련성 분석
- 의미적 유사도 계산
- 토픽 일관성 검증
- 브랜드 컨텍스트 일치
- 회피 응답 감지
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime
import re
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class RelevanceLevel(Enum):
    """관련성 수준"""
    HIGHLY_RELEVANT = "highly_relevant"     # 매우 관련됨
    RELEVANT = "relevant"                   # 관련됨
    PARTIALLY_RELEVANT = "partially_relevant"  # 부분 관련
    TANGENTIAL = "tangential"               # 간접 관련
    IRRELEVANT = "irrelevant"               # 무관함


class RelevanceType(Enum):
    """관련성 유형"""
    KEYWORD = "keyword"             # 키워드 매칭
    SEMANTIC = "semantic"           # 의미적 유사도
    TOPIC = "topic"                 # 토픽 일치
    INTENT = "intent"               # 의도 부합
    CONTEXT = "context"             # 컨텍스트 일치


class ResponseType(Enum):
    """응답 유형"""
    DIRECT_ANSWER = "direct_answer"     # 직접 답변
    EXPLANATION = "explanation"         # 설명
    CLARIFICATION = "clarification"     # 명확화 요청
    EVASIVE = "evasive"                 # 회피성
    OFF_TOPIC = "off_topic"             # 주제 이탈


class IssueSeverity(Enum):
    """이슈 심각도"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class RelevanceIssue:
    """관련성 이슈"""
    relevance_type: RelevanceType
    severity: IssueSeverity
    message: str
    score_impact: float = 0.0
    suggestion: Optional[str] = None


@dataclass
class RelevanceScore:
    """관련성 점수"""
    relevance_type: RelevanceType
    score: float  # 0.0 ~ 1.0
    weight: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RelevanceConfig:
    """관련성 설정"""
    # 최소 키워드 길이
    min_keyword_length: int = 2

    # 관련성 임계값
    min_relevance_score: float = 0.4
    high_relevance_threshold: float = 0.7

    # 회피 표현 (한국어)
    evasive_phrases_ko: List[str] = field(default_factory=lambda: [
        '정보가 부족',
        '알 수 없',
        '확실하지 않',
        '죄송합니다',
        '답변드리기 어렵',
        '정확한 정보를 찾을 수 없',
        '확인이 필요',
        '판단하기 어렵',
        '말씀드리기 어렵',
        '정보가 없',
    ])

    # 회피 표현 (영어)
    evasive_phrases_en: List[str] = field(default_factory=lambda: [
        "i don't know",
        "i'm not sure",
        "i cannot",
        "unable to",
        "no information",
        "not available",
        "cannot determine",
        "unclear",
    ])

    # 주제 이탈 표현
    off_topic_indicators: List[str] = field(default_factory=lambda: [
        '다른 주제',
        '그건 그렇고',
        '그나저나',
        '참고로',
        '여담이지만',
        'by the way',
        'anyway',
        'speaking of',
    ])

    # 가중치
    weights: Dict[RelevanceType, float] = field(default_factory=lambda: {
        RelevanceType.KEYWORD: 1.0,
        RelevanceType.SEMANTIC: 1.5,
        RelevanceType.TOPIC: 1.2,
        RelevanceType.INTENT: 1.3,
        RelevanceType.CONTEXT: 1.0,
    })

    # 언어 설정
    language: str = "ko"


@dataclass
class RelevanceResult:
    """관련성 검증 결과"""
    valid: bool
    score: float
    level: RelevanceLevel
    response_type: ResponseType
    issues: List[RelevanceIssue]
    warnings: List[str]
    scores: Dict[RelevanceType, RelevanceScore]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "score": self.score,
            "level": self.level.value,
            "response_type": self.response_type.value,
            "issues": [
                {
                    "type": issue.relevance_type.value,
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "score_impact": issue.score_impact,
                    "suggestion": issue.suggestion,
                }
                for issue in self.issues
            ],
            "warnings": self.warnings,
            "scores": {
                rtype.value: {
                    "score": rs.score,
                    "weight": rs.weight,
                    "details": rs.details,
                }
                for rtype, rs in self.scores.items()
            },
            "metadata": self.metadata,
        }


# ============================================================================
# Relevance Analyzers
# ============================================================================

class BaseRelevanceAnalyzer(ABC):
    """관련성 분석기 베이스 클래스"""

    @abstractmethod
    def analyze(
        self,
        question: str,
        response: str,
        context: Dict[str, Any],
        config: RelevanceConfig
    ) -> RelevanceScore:
        """관련성 분석"""
        pass

    @abstractmethod
    def get_issues(
        self,
        question: str,
        response: str,
        context: Dict[str, Any],
        config: RelevanceConfig
    ) -> List[RelevanceIssue]:
        """이슈 추출"""
        pass


class KeywordAnalyzer(BaseRelevanceAnalyzer):
    """키워드 기반 관련성 분석기"""

    def analyze(
        self,
        question: str,
        response: str,
        context: Dict[str, Any],
        config: RelevanceConfig
    ) -> RelevanceScore:
        # 키워드 추출
        question_keywords = self._extract_keywords(question, config)
        response_lower = response.lower()

        if not question_keywords:
            return RelevanceScore(
                relevance_type=RelevanceType.KEYWORD,
                score=0.7,  # 키워드 없으면 기본값
                weight=config.weights.get(RelevanceType.KEYWORD, 1.0),
                details={"keywords": [], "matches": 0, "reason": "no_keywords"}
            )

        # 키워드 매칭
        matches = sum(1 for kw in question_keywords if kw.lower() in response_lower)
        match_ratio = matches / len(question_keywords)

        # 점수 계산
        if match_ratio >= 0.5:
            score = 0.7 + 0.3 * match_ratio
        elif match_ratio >= 0.2:
            score = 0.4 + 0.6 * match_ratio
        else:
            score = match_ratio * 2

        return RelevanceScore(
            relevance_type=RelevanceType.KEYWORD,
            score=min(1.0, score),
            weight=config.weights.get(RelevanceType.KEYWORD, 1.0),
            details={
                "keywords": list(question_keywords),
                "matches": matches,
                "match_ratio": round(match_ratio, 3),
            }
        )

    def get_issues(
        self,
        question: str,
        response: str,
        context: Dict[str, Any],
        config: RelevanceConfig
    ) -> List[RelevanceIssue]:
        issues = []
        question_keywords = self._extract_keywords(question, config)
        response_lower = response.lower()

        if question_keywords:
            matches = sum(1 for kw in question_keywords if kw.lower() in response_lower)
            match_ratio = matches / len(question_keywords)

            if match_ratio < 0.2:
                issues.append(RelevanceIssue(
                    relevance_type=RelevanceType.KEYWORD,
                    severity=IssueSeverity.MEDIUM,
                    message="질문의 주요 키워드가 응답에 거의 없습니다",
                    score_impact=-0.2,
                    suggestion="질문에서 언급된 핵심 용어를 응답에 포함하세요",
                ))

        return issues

    def _extract_keywords(self, text: str, config: RelevanceConfig) -> Set[str]:
        """텍스트에서 키워드 추출"""
        # 불용어 (한국어)
        stopwords_ko = {
            '이', '그', '저', '것', '수', '등', '및', '또',
            '의', '가', '이', '은', '는', '을', '를', '에', '와', '과',
            '도', '로', '으로', '에서', '까지', '부터',
            '무엇', '어떤', '어떻게', '왜', '언제', '어디',
        }

        # 불용어 (영어)
        stopwords_en = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may',
            'might', 'must', 'shall', 'can', 'need', 'dare',
            'what', 'which', 'who', 'whom', 'whose', 'where',
            'when', 'why', 'how', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me',
            'him', 'her', 'us', 'them', 'my', 'your', 'his',
            'its', 'our', 'their', 'and', 'or', 'but', 'if',
            'because', 'as', 'until', 'while', 'of', 'at', 'by',
            'for', 'with', 'about', 'against', 'between', 'into',
            'through', 'during', 'before', 'after', 'above', 'below',
        }

        stopwords = stopwords_ko | stopwords_en

        # 단어 추출
        if config.language == "ko":
            words = re.findall(r'[가-힣]+|[a-zA-Z]+', text)
        else:
            words = re.findall(r'[a-zA-Z]+', text.lower())

        # 필터링
        keywords = {
            w for w in words
            if len(w) >= config.min_keyword_length and w.lower() not in stopwords
        }

        return keywords


class SemanticAnalyzer(BaseRelevanceAnalyzer):
    """의미적 유사도 분석기"""

    def analyze(
        self,
        question: str,
        response: str,
        context: Dict[str, Any],
        config: RelevanceConfig
    ) -> RelevanceScore:
        # 간단한 의미적 유사도 (N-gram 기반)
        question_ngrams = self._get_ngrams(question, n=2)
        response_ngrams = self._get_ngrams(response, n=2)

        if not question_ngrams or not response_ngrams:
            return RelevanceScore(
                relevance_type=RelevanceType.SEMANTIC,
                score=0.5,
                weight=config.weights.get(RelevanceType.SEMANTIC, 1.5),
                details={"reason": "insufficient_text"}
            )

        # Jaccard 유사도
        intersection = question_ngrams & response_ngrams
        union = question_ngrams | response_ngrams
        jaccard = len(intersection) / len(union) if union else 0

        # 점수 조정 (Jaccard는 보통 낮으므로 스케일링)
        score = min(1.0, jaccard * 5)

        return RelevanceScore(
            relevance_type=RelevanceType.SEMANTIC,
            score=score,
            weight=config.weights.get(RelevanceType.SEMANTIC, 1.5),
            details={
                "jaccard_similarity": round(jaccard, 4),
                "common_ngrams": len(intersection),
            }
        )

    def get_issues(
        self,
        question: str,
        response: str,
        context: Dict[str, Any],
        config: RelevanceConfig
    ) -> List[RelevanceIssue]:
        # 의미적 분석에서는 직접적인 이슈 감지보다
        # 점수 기반 판단에 의존
        return []

    def _get_ngrams(self, text: str, n: int = 2) -> Set[tuple]:
        """N-gram 추출"""
        words = re.findall(r'[가-힣]+|[a-zA-Z]+', text.lower())
        if len(words) < n:
            return set()
        return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)}


class TopicAnalyzer(BaseRelevanceAnalyzer):
    """토픽 일관성 분석기"""

    # 토픽 카테고리 키워드
    TOPIC_CATEGORIES = {
        'product': ['제품', '상품', '품목', 'product', 'item', '가격', '재고', '구매'],
        'service': ['서비스', '지원', '도움', 'service', 'support', 'help'],
        'inquiry': ['문의', '질문', '궁금', 'question', 'inquiry', 'ask'],
        'complaint': ['불만', '문제', '오류', 'complaint', 'problem', 'error', 'issue'],
        'feedback': ['피드백', '의견', '제안', 'feedback', 'opinion', 'suggestion'],
        'order': ['주문', '배송', '결제', 'order', 'delivery', 'payment'],
        'account': ['계정', '로그인', '비밀번호', 'account', 'login', 'password'],
        'brand': ['브랜드', '회사', '기업', 'brand', 'company', 'corporation'],
    }

    def analyze(
        self,
        question: str,
        response: str,
        context: Dict[str, Any],
        config: RelevanceConfig
    ) -> RelevanceScore:
        question_topics = self._detect_topics(question)
        response_topics = self._detect_topics(response)

        if not question_topics:
            return RelevanceScore(
                relevance_type=RelevanceType.TOPIC,
                score=0.7,
                weight=config.weights.get(RelevanceType.TOPIC, 1.2),
                details={"reason": "no_topic_detected"}
            )

        # 토픽 일치율
        common_topics = question_topics & response_topics
        topic_match_ratio = len(common_topics) / len(question_topics) if question_topics else 0

        score = 0.5 + 0.5 * topic_match_ratio

        return RelevanceScore(
            relevance_type=RelevanceType.TOPIC,
            score=score,
            weight=config.weights.get(RelevanceType.TOPIC, 1.2),
            details={
                "question_topics": list(question_topics),
                "response_topics": list(response_topics),
                "common_topics": list(common_topics),
                "match_ratio": round(topic_match_ratio, 3),
            }
        )

    def get_issues(
        self,
        question: str,
        response: str,
        context: Dict[str, Any],
        config: RelevanceConfig
    ) -> List[RelevanceIssue]:
        issues = []
        question_topics = self._detect_topics(question)
        response_topics = self._detect_topics(response)

        if question_topics and not (question_topics & response_topics):
            issues.append(RelevanceIssue(
                relevance_type=RelevanceType.TOPIC,
                severity=IssueSeverity.MEDIUM,
                message=f"질문 토픽({', '.join(question_topics)})과 응답 토픽이 다릅니다",
                score_impact=-0.15,
                suggestion="질문의 주제에 맞는 응답을 제공하세요",
            ))

        return issues

    def _detect_topics(self, text: str) -> Set[str]:
        """텍스트에서 토픽 감지"""
        text_lower = text.lower()
        detected = set()

        for topic, keywords in self.TOPIC_CATEGORIES.items():
            if any(kw in text_lower for kw in keywords):
                detected.add(topic)

        return detected


class IntentAnalyzer(BaseRelevanceAnalyzer):
    """의도 부합 분석기"""

    # 질문 의도 패턴
    INTENT_PATTERNS = {
        'information': [
            r'무엇|뭐|어떤|what|which',
            r'알려|설명|정보|info|tell|explain',
        ],
        'how_to': [
            r'어떻게|방법|how to|how do',
        ],
        'reason': [
            r'왜|이유|why|reason',
        ],
        'comparison': [
            r'비교|차이|vs|versus|compare|differ',
        ],
        'confirmation': [
            r'맞|인가요|인지|is it|does it|can it',
        ],
        'recommendation': [
            r'추천|권장|best|recommend|suggest',
        ],
        'location': [
            r'어디|위치|where|location',
        ],
        'time': [
            r'언제|시간|when|time',
        ],
        'quantity': [
            r'몇|얼마|how many|how much',
        ],
    }

    def analyze(
        self,
        question: str,
        response: str,
        context: Dict[str, Any],
        config: RelevanceConfig
    ) -> RelevanceScore:
        question_intent = self._detect_intent(question)
        response_satisfies = self._check_intent_satisfaction(question_intent, response)

        score = 0.8 if response_satisfies else 0.4

        return RelevanceScore(
            relevance_type=RelevanceType.INTENT,
            score=score,
            weight=config.weights.get(RelevanceType.INTENT, 1.3),
            details={
                "detected_intent": question_intent,
                "intent_satisfied": response_satisfies,
            }
        )

    def get_issues(
        self,
        question: str,
        response: str,
        context: Dict[str, Any],
        config: RelevanceConfig
    ) -> List[RelevanceIssue]:
        issues = []
        question_intent = self._detect_intent(question)
        response_satisfies = self._check_intent_satisfaction(question_intent, response)

        if question_intent and not response_satisfies:
            issues.append(RelevanceIssue(
                relevance_type=RelevanceType.INTENT,
                severity=IssueSeverity.MEDIUM,
                message=f"'{question_intent}' 유형의 질문에 대한 적절한 응답 형식이 아닙니다",
                score_impact=-0.15,
                suggestion="질문 유형에 맞는 형식으로 응답하세요",
            ))

        return issues

    def _detect_intent(self, question: str) -> Optional[str]:
        """질문 의도 감지"""
        question_lower = question.lower()

        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    return intent

        return None

    def _check_intent_satisfaction(self, intent: Optional[str], response: str) -> bool:
        """의도 충족 여부 확인"""
        if not intent:
            return True

        response_lower = response.lower()

        satisfaction_indicators = {
            'information': [r'\w+입니다', r'\w+은\s', r'\w+는\s', r'is\s', r'are\s'],
            'how_to': [r'먼저|다음|마지막|step|first|then|finally', r'\d+\.\s'],
            'reason': [r'때문|이유|왜냐|because|since|due to'],
            'comparison': [r'반면|차이|vs|while|whereas|compared'],
            'confirmation': [r'네|예|맞|아니|yes|no|correct|incorrect'],
            'recommendation': [r'추천|권장|suggest|recommend|best'],
            'location': [r'위치|장소|에서|at|in|location'],
            'time': [r'\d+시|\d+분|년|월|일|am|pm|o\'clock'],
            'quantity': [r'\d+개|\d+명|\d+원|\d+%'],
        }

        patterns = satisfaction_indicators.get(intent, [])
        return any(re.search(p, response_lower) for p in patterns)


class ContextAnalyzer(BaseRelevanceAnalyzer):
    """컨텍스트 일치 분석기"""

    def analyze(
        self,
        question: str,
        response: str,
        context: Dict[str, Any],
        config: RelevanceConfig
    ) -> RelevanceScore:
        brand_id = context.get('brand_id', '')
        brand_name = context.get('brand_name', '')
        retrieval_results = context.get('retrieval_results', {})

        score = 0.7  # 기본 점수

        # 브랜드 언급 확인
        if brand_name and brand_name.lower() in response.lower():
            score += 0.15

        # 검색 결과 기반 응답 확인
        if retrieval_results:
            source_keywords = self._extract_source_keywords(retrieval_results)
            response_lower = response.lower()
            matches = sum(1 for kw in source_keywords if kw.lower() in response_lower)
            if source_keywords:
                source_match_ratio = matches / len(source_keywords)
                score += 0.15 * source_match_ratio

        return RelevanceScore(
            relevance_type=RelevanceType.CONTEXT,
            score=min(1.0, score),
            weight=config.weights.get(RelevanceType.CONTEXT, 1.0),
            details={
                "brand_id": brand_id,
                "brand_mentioned": brand_name.lower() in response.lower() if brand_name else False,
                "has_retrieval_context": bool(retrieval_results),
            }
        )

    def get_issues(
        self,
        question: str,
        response: str,
        context: Dict[str, Any],
        config: RelevanceConfig
    ) -> List[RelevanceIssue]:
        issues = []
        brand_name = context.get('brand_name', '')

        # 다른 브랜드 언급 감지 (컨텍스트에 브랜드가 있는 경우)
        if brand_name:
            # 경쟁 브랜드 언급 감지 로직 (확장 가능)
            pass

        return issues

    def _extract_source_keywords(self, sources: Dict[str, Any]) -> Set[str]:
        """출처에서 키워드 추출"""
        keywords = set()

        def extract_recursive(obj: Any) -> None:
            if isinstance(obj, str):
                words = re.findall(r'[가-힣]{2,}|[a-zA-Z]{3,}', obj)
                keywords.update(w.lower() for w in words)
            elif isinstance(obj, dict):
                for v in obj.values():
                    extract_recursive(v)
            elif isinstance(obj, list):
                for item in obj:
                    extract_recursive(item)

        extract_recursive(sources)
        return keywords


# ============================================================================
# Response Type Detector
# ============================================================================

class ResponseTypeDetector:
    """응답 유형 감지기"""

    def detect(self, response: str, config: RelevanceConfig) -> ResponseType:
        """응답 유형 감지"""
        response_lower = response.lower()

        # 회피성 응답 감지
        evasive_phrases = (
            config.evasive_phrases_ko if config.language == "ko"
            else config.evasive_phrases_en
        )
        evasive_count = sum(1 for phrase in evasive_phrases if phrase in response_lower)
        if evasive_count >= 2 or (evasive_count == 1 and len(response) < 100):
            return ResponseType.EVASIVE

        # 주제 이탈 감지
        off_topic_count = sum(
            1 for indicator in config.off_topic_indicators
            if indicator in response_lower
        )
        if off_topic_count >= 2:
            return ResponseType.OFF_TOPIC

        # 명확화 요청 감지
        clarification_patterns = [
            r'무슨\s*말씀|정확히|구체적으로|어떤\s*것을',
            r'could you clarify|what do you mean|please specify',
        ]
        for pattern in clarification_patterns:
            if re.search(pattern, response_lower):
                return ResponseType.CLARIFICATION

        # 설명 감지
        explanation_indicators = [
            r'즉|다시\s*말해|예를\s*들어|왜냐하면',
            r'in other words|for example|because|that is',
            r'\d+\.\s+',  # 번호 매기기
        ]
        explanation_count = sum(
            1 for pattern in explanation_indicators
            if re.search(pattern, response_lower)
        )
        if explanation_count >= 2:
            return ResponseType.EXPLANATION

        return ResponseType.DIRECT_ANSWER


# ============================================================================
# Main Relevance Filter
# ============================================================================

class RelevanceFilter:
    """
    관련성 필터 - Production Grade v2.0

    다차원 분석:
    - 키워드 매칭 (Keyword)
    - 의미적 유사도 (Semantic)
    - 토픽 일관성 (Topic)
    - 의도 부합 (Intent)
    - 컨텍스트 일치 (Context)
    """

    def __init__(self, config: Optional[RelevanceConfig] = None):
        self.config = config or RelevanceConfig()
        self.analyzers: Dict[RelevanceType, BaseRelevanceAnalyzer] = {
            RelevanceType.KEYWORD: KeywordAnalyzer(),
            RelevanceType.SEMANTIC: SemanticAnalyzer(),
            RelevanceType.TOPIC: TopicAnalyzer(),
            RelevanceType.INTENT: IntentAnalyzer(),
            RelevanceType.CONTEXT: ContextAnalyzer(),
        }
        self.response_detector = ResponseTypeDetector()
        self._custom_analyzers: List[Callable[[str, str, Dict[str, Any]], List[RelevanceIssue]]] = []

    def register_analyzer(
        self,
        analyzer: Callable[[str, str, Dict[str, Any]], List[RelevanceIssue]]
    ) -> None:
        """커스텀 분석기 등록"""
        self._custom_analyzers.append(analyzer)

    def validate(
        self,
        response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> RelevanceResult:
        """
        관련성 검증 수행

        Args:
            response: 검증할 응답 텍스트
            context: 컨텍스트 (question, brand_id 등)

        Returns:
            RelevanceResult: 검증 결과
        """
        context = context or {}
        question = context.get('question', '')

        # 빈 응답 처리
        if not response or not response.strip():
            return RelevanceResult(
                valid=False,
                score=0.0,
                level=RelevanceLevel.IRRELEVANT,
                response_type=ResponseType.EVASIVE,
                issues=[RelevanceIssue(
                    relevance_type=RelevanceType.KEYWORD,
                    severity=IssueSeverity.CRITICAL,
                    message="응답이 비어 있습니다",
                )],
                warnings=[],
                scores={},
            )

        # 응답 유형 감지
        response_type = self.response_detector.detect(response, self.config)

        # 각 분석기 실행
        scores: Dict[RelevanceType, RelevanceScore] = {}
        all_issues: List[RelevanceIssue] = []

        for rtype, analyzer in self.analyzers.items():
            try:
                score = analyzer.analyze(question, response, context, self.config)
                scores[rtype] = score

                issues = analyzer.get_issues(question, response, context, self.config)
                all_issues.extend(issues)
            except Exception as e:
                logger.error(f"분석 오류 [{rtype.value}]: {e}")

        # 커스텀 분석기 실행
        for custom_analyzer in self._custom_analyzers:
            try:
                custom_issues = custom_analyzer(question, response, context)
                all_issues.extend(custom_issues)
            except Exception as e:
                logger.error(f"커스텀 분석기 오류: {e}")

        # 응답 유형에 따른 패널티
        type_penalty = {
            ResponseType.DIRECT_ANSWER: 0.0,
            ResponseType.EXPLANATION: 0.0,
            ResponseType.CLARIFICATION: 0.1,
            ResponseType.EVASIVE: 0.3,
            ResponseType.OFF_TOPIC: 0.4,
        }
        penalty = type_penalty.get(response_type, 0.0)

        # 가중 평균 점수 계산
        total_weight = sum(s.weight for s in scores.values())
        weighted_score = sum(
            s.score * s.weight for s in scores.values()
        ) / total_weight if total_weight > 0 else 0.0

        # 이슈 기반 감점
        for issue in all_issues:
            weighted_score += issue.score_impact

        # 최종 점수
        final_score = max(0.0, min(1.0, weighted_score - penalty))

        # 관련성 수준 결정
        level = self._determine_level(final_score)

        # 유효성 판정
        critical_issues = [i for i in all_issues if i.severity == IssueSeverity.CRITICAL]
        valid = (
            len(critical_issues) == 0 and
            final_score >= self.config.min_relevance_score and
            response_type not in (ResponseType.EVASIVE, ResponseType.OFF_TOPIC)
        )

        # 경고 생성
        warnings = []
        if response_type == ResponseType.EVASIVE:
            warnings.append("회피성 응답이 감지되었습니다")
        if response_type == ResponseType.OFF_TOPIC:
            warnings.append("주제 이탈이 감지되었습니다")
        for issue in all_issues:
            if issue.severity in (IssueSeverity.LOW, IssueSeverity.MEDIUM):
                warnings.append(f"[{issue.relevance_type.value}] {issue.message}")

        return RelevanceResult(
            valid=valid,
            score=round(final_score, 3),
            level=level,
            response_type=response_type,
            issues=all_issues,
            warnings=warnings,
            scores=scores,
            metadata={
                "analyzed_at": datetime.now().isoformat(),
                "question_length": len(question),
                "response_length": len(response),
                "penalty_applied": penalty,
            }
        )

    def _determine_level(self, score: float) -> RelevanceLevel:
        """관련성 수준 결정"""
        if score >= 0.85:
            return RelevanceLevel.HIGHLY_RELEVANT
        elif score >= 0.7:
            return RelevanceLevel.RELEVANT
        elif score >= 0.5:
            return RelevanceLevel.PARTIALLY_RELEVANT
        elif score >= 0.3:
            return RelevanceLevel.TANGENTIAL
        else:
            return RelevanceLevel.IRRELEVANT

    @staticmethod
    def quick_validate(response: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        빠른 검증 (정적 메서드, 하위 호환성)

        Args:
            response: 검증할 응답
            context: 컨텍스트

        Returns:
            검증 결과 딕셔너리
        """
        filter_instance = RelevanceFilter()
        result = filter_instance.validate(response, context or {})

        # 하위 호환성을 위한 간단한 형식
        return {
            'valid': result.valid,
            'issues': [i.message for i in result.issues if i.severity == IssueSeverity.CRITICAL],
            'warnings': result.warnings,
            'score': result.score,
        }
