"""
Trust Filter - Production Grade v2.0
신뢰성 검증 시스템

Features:
- 환각(Hallucination) 감지
- 출처 검증
- 사실 일관성 체크
- 데이터 근거 검증
- 신뢰도 점수 계산
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Set, Tuple, Callable
from datetime import datetime
import re
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class TrustLevel(Enum):
    """신뢰 수준"""
    VERIFIED = "verified"           # 검증됨
    HIGH = "high"                   # 높음
    MODERATE = "moderate"           # 보통
    LOW = "low"                     # 낮음
    UNVERIFIED = "unverified"       # 미검증


class HallucinationType(Enum):
    """환각 유형"""
    FABRICATED_FACT = "fabricated_fact"     # 지어낸 사실
    WRONG_NUMBER = "wrong_number"           # 잘못된 수치
    NONEXISTENT_SOURCE = "nonexistent_source"  # 존재하지 않는 출처
    TEMPORAL_ERROR = "temporal_error"       # 시간 오류
    ENTITY_CONFUSION = "entity_confusion"   # 엔티티 혼동


class TrustIssueSeverity(Enum):
    """신뢰성 이슈 심각도"""
    CRITICAL = "critical"   # 치명적
    HIGH = "high"           # 높음
    MEDIUM = "medium"       # 중간
    LOW = "low"             # 낮음


class VerificationType(Enum):
    """검증 유형"""
    SOURCE_CHECK = "source_check"           # 출처 확인
    FACT_CHECK = "fact_check"               # 사실 확인
    CONSISTENCY_CHECK = "consistency_check"  # 일관성 확인
    TEMPORAL_CHECK = "temporal_check"        # 시간 확인
    NUMERIC_CHECK = "numeric_check"          # 수치 확인


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TrustIssue:
    """신뢰성 이슈"""
    issue_type: HallucinationType
    severity: TrustIssueSeverity
    message: str
    evidence: Optional[str] = None
    location: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class VerificationResult:
    """검증 결과"""
    verification_type: VerificationType
    passed: bool
    confidence: float  # 0.0 ~ 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    issues: List[TrustIssue] = field(default_factory=list)


@dataclass
class TrustConfig:
    """신뢰성 설정"""
    # 수치 관련 키워드
    numeric_keywords: List[str] = field(default_factory=lambda: [
        '개', '명', '원', '₩', '%', '퍼센트', '배', '번', '회',
        '달러', '$', '유로', '€', '엔', '¥', '위안',
        '건', '조', '억', '만', '천', '백',
    ])

    # 시간 관련 키워드
    temporal_keywords: List[str] = field(default_factory=lambda: [
        '년', '월', '일', '시', '분', '초',
        '오늘', '어제', '내일', '지난', '다음',
        '최근', '현재', '과거', '미래',
    ])

    # 출처 관련 키워드
    source_keywords: List[str] = field(default_factory=lambda: [
        '에 따르면', '에서 발표', '에 의하면', '에서 보도',
        '연구 결과', '조사 결과', '통계', '보고서',
        '발표했다', '밝혔다', '전했다', '보도했다',
    ])

    # 확신 표현 (환각 위험 증가)
    certainty_phrases: List[str] = field(default_factory=lambda: [
        '확실히', '분명히', '틀림없이', '명백히',
        '반드시', '절대적으로', '100%', '항상',
    ])

    # 가중치
    weights: Dict[VerificationType, float] = field(default_factory=lambda: {
        VerificationType.SOURCE_CHECK: 1.5,
        VerificationType.FACT_CHECK: 1.3,
        VerificationType.CONSISTENCY_CHECK: 1.2,
        VerificationType.TEMPORAL_CHECK: 1.0,
        VerificationType.NUMERIC_CHECK: 1.4,
    })

    # 임계값
    min_trust_score: float = 0.5
    hallucination_threshold: float = 0.3


@dataclass
class TrustResult:
    """신뢰성 검증 결과"""
    valid: bool
    score: float
    level: TrustLevel
    issues: List[TrustIssue]
    verifications: Dict[VerificationType, VerificationResult]
    hallucination_risk: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "score": self.score,
            "level": self.level.value,
            "issues": [
                {
                    "type": issue.issue_type.value,
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "evidence": issue.evidence,
                    "location": issue.location,
                    "suggestion": issue.suggestion,
                }
                for issue in self.issues
            ],
            "verifications": {
                vtype.value: {
                    "passed": vr.passed,
                    "confidence": vr.confidence,
                    "details": vr.details,
                }
                for vtype, vr in self.verifications.items()
            },
            "hallucination_risk": self.hallucination_risk,
            "metadata": self.metadata,
        }


# ============================================================================
# Verification Checkers
# ============================================================================

class BaseChecker(ABC):
    """검증기 베이스 클래스"""

    @abstractmethod
    def check(
        self,
        response: str,
        context: Dict[str, Any],
        config: TrustConfig
    ) -> VerificationResult:
        """검증 수행"""
        pass


class SourceChecker(BaseChecker):
    """출처 검증기"""

    def check(
        self,
        response: str,
        context: Dict[str, Any],
        config: TrustConfig
    ) -> VerificationResult:
        issues: List[TrustIssue] = []
        retrieval_results = context.get('retrieval_results', {})

        # 검색 결과 유무 확인
        has_sources = bool(retrieval_results) and sum(
            len(v) if isinstance(v, list) else 1
            for v in retrieval_results.values()
        ) > 0

        # 출처 인용 표현 감지
        source_mentions = []
        for keyword in config.source_keywords:
            if keyword in response:
                source_mentions.append(keyword)

        # 출처 없이 구체적 정보 제공 시 경고
        specific_indicators = config.numeric_keywords + ['재고', '좋아요', '팔로워', '구독자']
        has_specific_data = any(ind in response for ind in specific_indicators)

        if not has_sources and has_specific_data:
            issues.append(TrustIssue(
                issue_type=HallucinationType.FABRICATED_FACT,
                severity=TrustIssueSeverity.HIGH,
                message="검색 결과 없이 구체적인 데이터를 제공하고 있습니다",
                suggestion="데이터 출처를 확인하거나 불확실성을 명시하세요",
            ))

        # 출처 언급은 있지만 실제 데이터가 없는 경우
        if source_mentions and not has_sources:
            issues.append(TrustIssue(
                issue_type=HallucinationType.NONEXISTENT_SOURCE,
                severity=TrustIssueSeverity.MEDIUM,
                message=f"출처 표현 '{source_mentions[0]}'이 사용되었으나 검색 데이터가 없습니다",
                evidence=f"사용된 표현: {', '.join(source_mentions)}",
                suggestion="실제 검색 데이터를 기반으로 응답하세요",
            ))

        # 점수 계산
        if issues:
            critical_count = sum(1 for i in issues if i.severity == TrustIssueSeverity.CRITICAL)
            high_count = sum(1 for i in issues if i.severity == TrustIssueSeverity.HIGH)
            confidence = max(0.0, 1.0 - critical_count * 0.4 - high_count * 0.2)
        else:
            confidence = 1.0 if has_sources else 0.7

        return VerificationResult(
            verification_type=VerificationType.SOURCE_CHECK,
            passed=len([i for i in issues if i.severity in (TrustIssueSeverity.CRITICAL, TrustIssueSeverity.HIGH)]) == 0,
            confidence=confidence,
            details={
                "has_sources": has_sources,
                "source_mentions": source_mentions,
                "has_specific_data": has_specific_data,
            },
            issues=issues,
        )


class FactChecker(BaseChecker):
    """사실 검증기"""

    # 일반적으로 잘못 알려진 사실 패턴
    COMMON_MISCONCEPTIONS = [
        (r'만리장성.{0,10}달에서.{0,10}보인다', "만리장성은 달에서 보이지 않습니다"),
        (r'금붕어.{0,10}기억.{0,3}3초', "금붕어 기억력 3초는 잘못된 속설입니다"),
    ]

    def check(
        self,
        response: str,
        context: Dict[str, Any],
        config: TrustConfig
    ) -> VerificationResult:
        issues: List[TrustIssue] = []

        # 잘못된 사실 패턴 체크
        for pattern, message in self.COMMON_MISCONCEPTIONS:
            if re.search(pattern, response, re.IGNORECASE):
                issues.append(TrustIssue(
                    issue_type=HallucinationType.FABRICATED_FACT,
                    severity=TrustIssueSeverity.HIGH,
                    message=message,
                    suggestion="사실 관계를 재확인하세요",
                ))

        # 과도한 확신 표현 체크
        certainty_count = sum(1 for phrase in config.certainty_phrases if phrase in response)
        if certainty_count >= 3:
            issues.append(TrustIssue(
                issue_type=HallucinationType.FABRICATED_FACT,
                severity=TrustIssueSeverity.LOW,
                message="과도한 확신 표현이 사용되었습니다",
                evidence=f"확신 표현 {certainty_count}개 감지",
                suggestion="적절한 불확실성 표현을 사용하세요",
            ))

        confidence = max(0.0, 1.0 - len(issues) * 0.2)

        return VerificationResult(
            verification_type=VerificationType.FACT_CHECK,
            passed=len([i for i in issues if i.severity in (TrustIssueSeverity.CRITICAL, TrustIssueSeverity.HIGH)]) == 0,
            confidence=confidence,
            details={
                "misconception_matches": len([i for i in issues if i.issue_type == HallucinationType.FABRICATED_FACT]),
                "certainty_expression_count": certainty_count,
            },
            issues=issues,
        )


class ConsistencyChecker(BaseChecker):
    """일관성 검증기"""

    def check(
        self,
        response: str,
        context: Dict[str, Any],
        config: TrustConfig
    ) -> VerificationResult:
        issues: List[TrustIssue] = []

        # 같은 응답 내 모순 체크
        contradictions = self._find_contradictions(response)
        for contradiction in contradictions:
            issues.append(TrustIssue(
                issue_type=HallucinationType.ENTITY_CONFUSION,
                severity=TrustIssueSeverity.MEDIUM,
                message="응답 내 모순이 감지되었습니다",
                evidence=contradiction,
                suggestion="일관된 정보를 제공하세요",
            ))

        # 이전 컨텍스트와의 일관성 (대화 기록이 있는 경우)
        previous_response = context.get('previous_response', '')
        if previous_response:
            inconsistencies = self._check_context_consistency(response, previous_response)
            for inconsistency in inconsistencies:
                issues.append(TrustIssue(
                    issue_type=HallucinationType.ENTITY_CONFUSION,
                    severity=TrustIssueSeverity.MEDIUM,
                    message="이전 응답과 불일치합니다",
                    evidence=inconsistency,
                    suggestion="이전 맥락과 일관성을 유지하세요",
                ))

        confidence = max(0.0, 1.0 - len(issues) * 0.25)

        return VerificationResult(
            verification_type=VerificationType.CONSISTENCY_CHECK,
            passed=len(issues) == 0,
            confidence=confidence,
            details={
                "internal_contradictions": len(contradictions),
                "context_inconsistencies": len(issues) - len(contradictions),
            },
            issues=issues,
        )

    def _find_contradictions(self, text: str) -> List[str]:
        """내부 모순 감지"""
        contradictions = []

        # 부정 패턴 감지
        affirmative_negative_pairs = [
            (r'있습니다', r'없습니다'),
            (r'맞습니다', r'틀립니다'),
            (r'가능합니다', r'불가능합니다'),
            (r'증가', r'감소'),
        ]

        sentences = re.split(r'[.!?]\s*', text)
        for aff, neg in affirmative_negative_pairs:
            aff_sentences = [s for s in sentences if re.search(aff, s)]
            neg_sentences = [s for s in sentences if re.search(neg, s)]

            # 동일 주제에 대한 긍정/부정이 모두 있으면 모순
            if aff_sentences and neg_sentences:
                # 간단한 키워드 기반 주제 매칭
                for aff_s in aff_sentences:
                    for neg_s in neg_sentences:
                        common_words = set(aff_s.split()) & set(neg_s.split())
                        if len(common_words) >= 2:  # 공통 단어가 2개 이상이면 같은 주제로 판단
                            contradictions.append(f"'{aff_s[:30]}...' vs '{neg_s[:30]}...'")

        return contradictions[:3]  # 최대 3개까지만

    def _check_context_consistency(self, current: str, previous: str) -> List[str]:
        """컨텍스트 일관성 체크"""
        # 숫자 일관성 체크
        current_numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', current)
        previous_numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', previous)

        inconsistencies = []

        # 같은 컨텍스트에서 급격한 숫자 변화 감지
        # (실제 구현에서는 더 정교한 엔티티 매칭 필요)

        return inconsistencies


class TemporalChecker(BaseChecker):
    """시간 검증기"""

    def check(
        self,
        response: str,
        context: Dict[str, Any],
        config: TrustConfig
    ) -> VerificationResult:
        issues: List[TrustIssue] = []

        # 미래 날짜가 과거형으로 서술된 경우
        future_dates = self._extract_future_dates(response)
        past_tense_indicators = ['했다', '였다', '했습니다', '이었습니다', '발표했다', '보도했다']

        for date_info in future_dates:
            for indicator in past_tense_indicators:
                if indicator in response:
                    # 미래 날짜 주변에 과거형이 있는지 확인
                    date_pos = response.find(date_info['text'])
                    indicator_pos = response.find(indicator)
                    if abs(date_pos - indicator_pos) < 100:  # 100자 이내
                        issues.append(TrustIssue(
                            issue_type=HallucinationType.TEMPORAL_ERROR,
                            severity=TrustIssueSeverity.HIGH,
                            message=f"미래 날짜 '{date_info['text']}'에 대해 과거형이 사용되었습니다",
                            suggestion="시제를 수정하세요",
                        ))
                        break

        # 현재 날짜 기준 검증
        current_year = datetime.now().year
        year_mentions = re.findall(r'(20\d{2})년', response)
        for year_str in year_mentions:
            year = int(year_str)
            if year > current_year + 5:
                issues.append(TrustIssue(
                    issue_type=HallucinationType.TEMPORAL_ERROR,
                    severity=TrustIssueSeverity.MEDIUM,
                    message=f"먼 미래 연도({year}년)가 언급되었습니다",
                    suggestion="날짜 정보를 확인하세요",
                ))

        confidence = max(0.0, 1.0 - len(issues) * 0.3)

        return VerificationResult(
            verification_type=VerificationType.TEMPORAL_CHECK,
            passed=len([i for i in issues if i.severity in (TrustIssueSeverity.CRITICAL, TrustIssueSeverity.HIGH)]) == 0,
            confidence=confidence,
            details={
                "temporal_issues": len(issues),
                "year_mentions": year_mentions,
            },
            issues=issues,
        )

    def _extract_future_dates(self, text: str) -> List[Dict[str, Any]]:
        """미래 날짜 추출"""
        future_dates = []
        current_date = datetime.now()

        # 년월일 패턴
        date_patterns = [
            r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일',
            r'(\d{4})[./-](\d{1,2})[./-](\d{1,2})',
        ]

        for pattern in date_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
                    date = datetime(year, month, day)
                    if date > current_date:
                        future_dates.append({
                            'text': match.group(0),
                            'date': date,
                        })
                except (ValueError, IndexError):
                    pass

        return future_dates


class NumericChecker(BaseChecker):
    """수치 검증기"""

    # 비현실적인 수치 패턴
    UNREALISTIC_PATTERNS = [
        (r'100\s*%\s*이상', "100% 초과는 일반적으로 불가능합니다"),
        (r'(\d{10,})\s*명', "인구수가 비현실적입니다"),
        (r'마이너스\s*(\d+)\s*%\s*성장', "마이너스 성장률 표현을 확인하세요"),
    ]

    def check(
        self,
        response: str,
        context: Dict[str, Any],
        config: TrustConfig
    ) -> VerificationResult:
        issues: List[TrustIssue] = []

        # 비현실적 수치 패턴 체크
        for pattern, message in self.UNREALISTIC_PATTERNS:
            if re.search(pattern, response):
                issues.append(TrustIssue(
                    issue_type=HallucinationType.WRONG_NUMBER,
                    severity=TrustIssueSeverity.MEDIUM,
                    message=message,
                    suggestion="수치를 재확인하세요",
                ))

        # 검색 결과의 수치와 비교
        retrieval_results = context.get('retrieval_results', {})
        if retrieval_results:
            response_numbers = self._extract_numbers(response)
            source_numbers = self._extract_numbers_from_sources(retrieval_results)

            # 출처에 없는 큰 수치 감지
            for num in response_numbers:
                if num > 1000 and num not in source_numbers:
                    # 비슷한 수치도 없는지 확인 (10% 오차 허용)
                    has_similar = any(
                        abs(num - src_num) / max(num, 1) < 0.1
                        for src_num in source_numbers
                    )
                    if not has_similar and source_numbers:
                        issues.append(TrustIssue(
                            issue_type=HallucinationType.WRONG_NUMBER,
                            severity=TrustIssueSeverity.LOW,
                            message=f"수치 {num:,}이 출처에서 확인되지 않습니다",
                            suggestion="수치의 출처를 명시하거나 확인하세요",
                        ))

        confidence = max(0.0, 1.0 - len(issues) * 0.2)

        return VerificationResult(
            verification_type=VerificationType.NUMERIC_CHECK,
            passed=len([i for i in issues if i.severity in (TrustIssueSeverity.CRITICAL, TrustIssueSeverity.HIGH)]) == 0,
            confidence=confidence,
            details={
                "numeric_issues": len(issues),
            },
            issues=issues,
        )

    def _extract_numbers(self, text: str) -> Set[float]:
        """텍스트에서 숫자 추출"""
        numbers = set()

        # 쉼표가 있는 숫자
        for match in re.finditer(r'(\d{1,3}(?:,\d{3})+)', text):
            try:
                numbers.add(float(match.group(1).replace(',', '')))
            except ValueError:
                pass

        # 일반 숫자
        for match in re.finditer(r'(?<![,\d])(\d+(?:\.\d+)?)(?![,\d])', text):
            try:
                numbers.add(float(match.group(1)))
            except ValueError:
                pass

        return numbers

    def _extract_numbers_from_sources(self, sources: Dict[str, Any]) -> Set[float]:
        """출처 데이터에서 숫자 추출"""
        numbers = set()

        def extract_recursive(obj: Any) -> None:
            if isinstance(obj, (int, float)):
                numbers.add(float(obj))
            elif isinstance(obj, str):
                numbers.update(self._extract_numbers(obj))
            elif isinstance(obj, dict):
                for v in obj.values():
                    extract_recursive(v)
            elif isinstance(obj, list):
                for item in obj:
                    extract_recursive(item)

        extract_recursive(sources)
        return numbers


# ============================================================================
# Main Trust Filter
# ============================================================================

class TrustFilter:
    """
    신뢰성 필터 - Production Grade v2.0

    다중 검증:
    - 출처 검증 (Source Check)
    - 사실 검증 (Fact Check)
    - 일관성 검증 (Consistency Check)
    - 시간 검증 (Temporal Check)
    - 수치 검증 (Numeric Check)
    """

    def __init__(self, config: Optional[TrustConfig] = None):
        self.config = config or TrustConfig()
        self.checkers: Dict[VerificationType, BaseChecker] = {
            VerificationType.SOURCE_CHECK: SourceChecker(),
            VerificationType.FACT_CHECK: FactChecker(),
            VerificationType.CONSISTENCY_CHECK: ConsistencyChecker(),
            VerificationType.TEMPORAL_CHECK: TemporalChecker(),
            VerificationType.NUMERIC_CHECK: NumericChecker(),
        }
        self._custom_checkers: List[Callable[[str, Dict[str, Any]], List[TrustIssue]]] = []

    def register_checker(
        self,
        checker: Callable[[str, Dict[str, Any]], List[TrustIssue]]
    ) -> None:
        """커스텀 검증기 등록"""
        self._custom_checkers.append(checker)

    def validate(
        self,
        response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> TrustResult:
        """
        신뢰성 검증 수행

        Args:
            response: 검증할 응답 텍스트
            context: 추가 컨텍스트 (retrieval_results 등)

        Returns:
            TrustResult: 검증 결과
        """
        context = context or {}

        # 빈 응답 처리
        if not response or not response.strip():
            return TrustResult(
                valid=True,  # 빈 응답은 환각이 아님
                score=1.0,
                level=TrustLevel.VERIFIED,
                issues=[],
                verifications={},
                hallucination_risk=0.0,
            )

        # 각 검증기 실행
        verifications: Dict[VerificationType, VerificationResult] = {}
        all_issues: List[TrustIssue] = []

        for vtype, checker in self.checkers.items():
            try:
                result = checker.check(response, context, self.config)
                verifications[vtype] = result
                all_issues.extend(result.issues)
            except Exception as e:
                logger.error(f"검증 오류 [{vtype.value}]: {e}")

        # 커스텀 검증기 실행
        for custom_checker in self._custom_checkers:
            try:
                custom_issues = custom_checker(response, context)
                all_issues.extend(custom_issues)
            except Exception as e:
                logger.error(f"커스텀 검증기 오류: {e}")

        # 가중 평균 점수 계산
        total_weight = sum(
            self.config.weights.get(vtype, 1.0)
            for vtype in verifications.keys()
        )
        weighted_score = sum(
            vr.confidence * self.config.weights.get(vtype, 1.0)
            for vtype, vr in verifications.items()
        ) / total_weight if total_weight > 0 else 0.0

        # 환각 위험도 계산
        critical_count = sum(1 for i in all_issues if i.severity == TrustIssueSeverity.CRITICAL)
        high_count = sum(1 for i in all_issues if i.severity == TrustIssueSeverity.HIGH)
        medium_count = sum(1 for i in all_issues if i.severity == TrustIssueSeverity.MEDIUM)

        hallucination_risk = min(1.0, critical_count * 0.4 + high_count * 0.2 + medium_count * 0.1)

        # 최종 점수
        final_score = max(0.0, min(1.0, weighted_score - hallucination_risk * 0.5))

        # 신뢰 수준 결정
        level = self._determine_level(final_score, hallucination_risk)

        # 유효성 판정
        valid = (
            critical_count == 0 and
            hallucination_risk < self.config.hallucination_threshold and
            final_score >= self.config.min_trust_score
        )

        return TrustResult(
            valid=valid,
            score=round(final_score, 3),
            level=level,
            issues=all_issues,
            verifications=verifications,
            hallucination_risk=round(hallucination_risk, 3),
            metadata={
                "analyzed_at": datetime.now().isoformat(),
                "text_length": len(response),
                "issue_counts": {
                    "critical": critical_count,
                    "high": high_count,
                    "medium": medium_count,
                    "low": sum(1 for i in all_issues if i.severity == TrustIssueSeverity.LOW),
                }
            }
        )

    def _determine_level(self, score: float, hallucination_risk: float) -> TrustLevel:
        """신뢰 수준 결정"""
        if score >= 0.9 and hallucination_risk < 0.1:
            return TrustLevel.VERIFIED
        elif score >= 0.7 and hallucination_risk < 0.2:
            return TrustLevel.HIGH
        elif score >= 0.5 and hallucination_risk < 0.4:
            return TrustLevel.MODERATE
        elif score >= 0.3:
            return TrustLevel.LOW
        else:
            return TrustLevel.UNVERIFIED

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
        filter_instance = TrustFilter()
        result = filter_instance.validate(response, context or {})

        # 하위 호환성을 위한 간단한 형식
        return {
            'valid': result.valid,
            'issues': [i.message for i in result.issues if i.severity in (TrustIssueSeverity.CRITICAL, TrustIssueSeverity.HIGH)],
            'score': result.score,
        }
