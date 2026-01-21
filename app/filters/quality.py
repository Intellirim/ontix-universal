"""
Quality Filter - Production Grade v2.0
품질 검증 시스템

Features:
- 다층 품질 점수 계산
- 언어별 품질 기준
- 구조적 완결성 검증
- 가독성 분석
- 자동 품질 개선 제안
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime
import re
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class QualityLevel(Enum):
    """품질 수준"""
    EXCELLENT = "excellent"      # 90-100
    GOOD = "good"               # 70-89
    ACCEPTABLE = "acceptable"   # 50-69
    POOR = "poor"               # 30-49
    UNACCEPTABLE = "unacceptable"  # 0-29


class QualityDimension(Enum):
    """품질 측정 차원"""
    LENGTH = "length"
    COMPLETENESS = "completeness"
    READABILITY = "readability"
    STRUCTURE = "structure"
    LANGUAGE = "language"
    FORMATTING = "formatting"


class IssueSeverity(Enum):
    """이슈 심각도"""
    CRITICAL = "critical"   # 즉시 수정 필요
    MAJOR = "major"         # 수정 권장
    MINOR = "minor"         # 개선 가능
    INFO = "info"           # 참고 사항


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class QualityIssue:
    """품질 이슈"""
    dimension: QualityDimension
    severity: IssueSeverity
    message: str
    location: Optional[str] = None
    suggestion: Optional[str] = None
    auto_fixable: bool = False


@dataclass
class DimensionScore:
    """차원별 점수"""
    dimension: QualityDimension
    score: float  # 0.0 ~ 1.0
    weight: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityConfig:
    """품질 설정"""
    min_length: int = 10
    max_length: int = 5000
    optimal_min_length: int = 50
    optimal_max_length: int = 2000
    min_sentence_count: int = 1
    max_sentence_length: int = 200
    language: str = "ko"
    weights: Dict[QualityDimension, float] = field(default_factory=lambda: {
        QualityDimension.LENGTH: 1.0,
        QualityDimension.COMPLETENESS: 1.5,
        QualityDimension.READABILITY: 1.2,
        QualityDimension.STRUCTURE: 1.0,
        QualityDimension.LANGUAGE: 1.0,
        QualityDimension.FORMATTING: 0.8,
    })


@dataclass
class QualityResult:
    """품질 검증 결과"""
    valid: bool
    score: float
    level: QualityLevel
    issues: List[QualityIssue]
    warnings: List[str]
    dimension_scores: Dict[QualityDimension, DimensionScore]
    suggestions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "score": self.score,
            "level": self.level.value,
            "issues": [
                {
                    "dimension": issue.dimension.value,
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "location": issue.location,
                    "suggestion": issue.suggestion,
                    "auto_fixable": issue.auto_fixable,
                }
                for issue in self.issues
            ],
            "warnings": self.warnings,
            "dimension_scores": {
                dim.value: {
                    "score": ds.score,
                    "weight": ds.weight,
                    "details": ds.details,
                }
                for dim, ds in self.dimension_scores.items()
            },
            "suggestions": self.suggestions,
            "metadata": self.metadata,
        }


# ============================================================================
# Quality Analyzers
# ============================================================================

class BaseAnalyzer(ABC):
    """분석기 베이스 클래스"""

    @abstractmethod
    def analyze(
        self,
        text: str,
        config: QualityConfig,
        context: Dict[str, Any]
    ) -> DimensionScore:
        """텍스트 분석"""
        pass

    @abstractmethod
    def get_issues(
        self,
        text: str,
        config: QualityConfig,
        context: Dict[str, Any]
    ) -> List[QualityIssue]:
        """이슈 추출"""
        pass


class LengthAnalyzer(BaseAnalyzer):
    """길이 분석기"""

    def analyze(
        self,
        text: str,
        config: QualityConfig,
        context: Dict[str, Any]
    ) -> DimensionScore:
        length = len(text)

        if length < config.min_length:
            score = 0.0
        elif length < config.optimal_min_length:
            score = 0.3 + 0.3 * (length - config.min_length) / (config.optimal_min_length - config.min_length)
        elif length <= config.optimal_max_length:
            score = 1.0
        elif length <= config.max_length:
            score = 1.0 - 0.3 * (length - config.optimal_max_length) / (config.max_length - config.optimal_max_length)
        else:
            score = 0.5

        return DimensionScore(
            dimension=QualityDimension.LENGTH,
            score=max(0.0, min(1.0, score)),
            weight=config.weights.get(QualityDimension.LENGTH, 1.0),
            details={
                "length": length,
                "min": config.min_length,
                "max": config.max_length,
                "optimal_range": (config.optimal_min_length, config.optimal_max_length),
            }
        )

    def get_issues(
        self,
        text: str,
        config: QualityConfig,
        context: Dict[str, Any]
    ) -> List[QualityIssue]:
        issues = []
        length = len(text)

        if length < config.min_length:
            issues.append(QualityIssue(
                dimension=QualityDimension.LENGTH,
                severity=IssueSeverity.CRITICAL,
                message=f"텍스트가 너무 짧습니다 ({length}자 < {config.min_length}자)",
                suggestion="더 상세한 내용을 추가하세요",
            ))
        elif length < config.optimal_min_length:
            issues.append(QualityIssue(
                dimension=QualityDimension.LENGTH,
                severity=IssueSeverity.MINOR,
                message=f"텍스트가 다소 짧습니다 ({length}자)",
                suggestion=f"최소 {config.optimal_min_length}자 이상 작성을 권장합니다",
            ))

        if length > config.max_length:
            issues.append(QualityIssue(
                dimension=QualityDimension.LENGTH,
                severity=IssueSeverity.MAJOR,
                message=f"텍스트가 너무 깁니다 ({length}자 > {config.max_length}자)",
                suggestion="핵심 내용만 유지하고 불필요한 부분을 제거하세요",
            ))

        return issues


class CompletenessAnalyzer(BaseAnalyzer):
    """완결성 분석기"""

    INCOMPLETE_PATTERNS_KO = [
        (r'\.{3,}$', "말줄임표로 끝남"),
        (r'등$', "'등'으로 끝남"),
        (r'그리고$', "'그리고'로 끝남"),
        (r'또한$', "'또한'으로 끝남"),
        (r'및$', "'및'으로 끝남"),
        (r'그래서$', "'그래서'로 끝남"),
    ]

    INCOMPLETE_PATTERNS_EN = [
        (r'\.{3,}$', "Ends with ellipsis"),
        (r'\band\s*$', "Ends with 'and'"),
        (r'\bor\s*$', "Ends with 'or'"),
        (r'\bbut\s*$', "Ends with 'but'"),
        (r'\balso\s*$', "Ends with 'also'"),
    ]

    def analyze(
        self,
        text: str,
        config: QualityConfig,
        context: Dict[str, Any]
    ) -> DimensionScore:
        issues = self._check_completeness(text, config.language)

        # 문장 구조 분석
        sentences = self._split_sentences(text, config.language)
        has_proper_ending = self._has_proper_ending(text, config.language)
        has_structure = len(sentences) >= config.min_sentence_count

        score = 1.0
        if issues:
            score -= 0.2 * len(issues)
        if not has_proper_ending:
            score -= 0.3
        if not has_structure:
            score -= 0.2

        return DimensionScore(
            dimension=QualityDimension.COMPLETENESS,
            score=max(0.0, min(1.0, score)),
            weight=config.weights.get(QualityDimension.COMPLETENESS, 1.5),
            details={
                "sentence_count": len(sentences),
                "has_proper_ending": has_proper_ending,
                "incomplete_patterns_found": len(issues),
            }
        )

    def get_issues(
        self,
        text: str,
        config: QualityConfig,
        context: Dict[str, Any]
    ) -> List[QualityIssue]:
        issues = []

        for pattern, message in self._get_patterns(config.language):
            if re.search(pattern, text.strip()):
                issues.append(QualityIssue(
                    dimension=QualityDimension.COMPLETENESS,
                    severity=IssueSeverity.MAJOR,
                    message=f"불완전한 문장: {message}",
                    suggestion="문장을 완성하세요",
                ))

        if not self._has_proper_ending(text, config.language):
            issues.append(QualityIssue(
                dimension=QualityDimension.COMPLETENESS,
                severity=IssueSeverity.MINOR,
                message="적절한 종결 표현이 없습니다",
                suggestion="마침표나 종결어미로 문장을 마무리하세요",
            ))

        return issues

    def _get_patterns(self, language: str) -> List[tuple]:
        if language == "ko":
            return self.INCOMPLETE_PATTERNS_KO
        return self.INCOMPLETE_PATTERNS_EN

    def _split_sentences(self, text: str, language: str) -> List[str]:
        if language == "ko":
            sentences = re.split(r'[.!?。]\s*', text)
        else:
            sentences = re.split(r'[.!?]\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _has_proper_ending(self, text: str, language: str) -> bool:
        text = text.strip()
        if not text:
            return False

        if language == "ko":
            # 한국어: 마침표, 종결어미 등
            return bool(re.search(r'[.!?。다요죠습니다까네세요]$', text))
        else:
            return bool(re.search(r'[.!?]$', text))

    def _check_completeness(self, text: str, language: str) -> List[str]:
        issues = []
        for pattern, message in self._get_patterns(language):
            if re.search(pattern, text.strip()):
                issues.append(message)
        return issues


class ReadabilityAnalyzer(BaseAnalyzer):
    """가독성 분석기"""

    def analyze(
        self,
        text: str,
        config: QualityConfig,
        context: Dict[str, Any]
    ) -> DimensionScore:
        sentences = self._split_sentences(text)
        words = self._count_words(text, config.language)

        # 평균 문장 길이
        avg_sentence_length = len(text) / max(1, len(sentences))

        # 문장 길이 점수
        if avg_sentence_length < 30:
            length_score = 0.8
        elif avg_sentence_length < 80:
            length_score = 1.0
        elif avg_sentence_length < 150:
            length_score = 0.7
        else:
            length_score = 0.4

        # 단어 다양성
        unique_words = set(self._tokenize(text, config.language))
        diversity = len(unique_words) / max(1, words) if words > 0 else 0
        diversity_score = min(1.0, diversity * 2)

        # 문단 구조
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        structure_score = 1.0 if len(paragraphs) > 1 or len(text) < 200 else 0.8

        score = (length_score * 0.4 + diversity_score * 0.3 + structure_score * 0.3)

        return DimensionScore(
            dimension=QualityDimension.READABILITY,
            score=score,
            weight=config.weights.get(QualityDimension.READABILITY, 1.2),
            details={
                "avg_sentence_length": avg_sentence_length,
                "word_count": words,
                "unique_words": len(unique_words),
                "paragraph_count": len(paragraphs),
                "diversity_ratio": diversity,
            }
        )

    def get_issues(
        self,
        text: str,
        config: QualityConfig,
        context: Dict[str, Any]
    ) -> List[QualityIssue]:
        issues = []
        sentences = self._split_sentences(text)

        for i, sentence in enumerate(sentences):
            if len(sentence) > config.max_sentence_length:
                issues.append(QualityIssue(
                    dimension=QualityDimension.READABILITY,
                    severity=IssueSeverity.MINOR,
                    message=f"문장이 너무 깁니다 ({len(sentence)}자)",
                    location=f"문장 {i+1}",
                    suggestion="문장을 나누어 가독성을 높이세요",
                ))

        return issues

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'[.!?。]\s*', text)
        return [s.strip() for s in sentences if s.strip()]

    def _count_words(self, text: str, language: str) -> int:
        if language == "ko":
            # 한국어: 공백 기준 + 형태소 추정
            return len(text.split())
        return len(text.split())

    def _tokenize(self, text: str, language: str) -> List[str]:
        if language == "ko":
            # 간단한 토큰화 (공백 + 조사 제거)
            words = re.findall(r'[가-힣]+', text)
            return [w for w in words if len(w) > 1]
        return text.lower().split()


class StructureAnalyzer(BaseAnalyzer):
    """구조 분석기"""

    def analyze(
        self,
        text: str,
        config: QualityConfig,
        context: Dict[str, Any]
    ) -> DimensionScore:
        # 리스트/불릿 포인트 감지
        has_list = bool(re.search(r'(?:^|\n)\s*[-•*]\s+', text))
        has_numbered_list = bool(re.search(r'(?:^|\n)\s*\d+[.)]\s+', text))

        # 헤더 감지
        has_headers = bool(re.search(r'(?:^|\n)#+\s+', text)) or bool(re.search(r'(?:^|\n).+:\s*$', text, re.MULTILINE))

        # 문단 구조
        paragraphs = [p for p in text.split('\n\n') if p.strip()]

        # 점수 계산
        score = 0.7  # 기본 점수

        if has_list or has_numbered_list:
            score += 0.15
        if has_headers:
            score += 0.1
        if len(paragraphs) > 1:
            score += 0.05

        return DimensionScore(
            dimension=QualityDimension.STRUCTURE,
            score=min(1.0, score),
            weight=config.weights.get(QualityDimension.STRUCTURE, 1.0),
            details={
                "has_list": has_list,
                "has_numbered_list": has_numbered_list,
                "has_headers": has_headers,
                "paragraph_count": len(paragraphs),
            }
        )

    def get_issues(
        self,
        text: str,
        config: QualityConfig,
        context: Dict[str, Any]
    ) -> List[QualityIssue]:
        issues = []

        # 긴 텍스트에 구조가 없는 경우
        if len(text) > 500:
            paragraphs = [p for p in text.split('\n\n') if p.strip()]
            has_structure = (
                bool(re.search(r'(?:^|\n)\s*[-•*]\s+', text)) or
                bool(re.search(r'(?:^|\n)\s*\d+[.)]\s+', text)) or
                len(paragraphs) > 2
            )

            if not has_structure:
                issues.append(QualityIssue(
                    dimension=QualityDimension.STRUCTURE,
                    severity=IssueSeverity.MINOR,
                    message="긴 텍스트에 구조가 부족합니다",
                    suggestion="리스트, 문단 나누기, 또는 소제목을 사용하세요",
                ))

        return issues


class LanguageAnalyzer(BaseAnalyzer):
    """언어 품질 분석기"""

    ERROR_PATTERNS_KO = [
        (r'되{2,}', "중복된 '되'"),
        (r'요{2,}', "중복된 종결어미"),
        (r'ㅋ{5,}|ㅎ{5,}|ㅠ{5,}', "과도한 이모티콘"),
        (r'\.{4,}', "과도한 마침표"),
        (r'!{3,}', "과도한 느낌표"),
        (r'\?{3,}', "과도한 물음표"),
    ]

    ERROR_KEYWORDS = [
        'error', 'exception', 'failed', 'undefined', 'null',
        '실패', '오류', '에러', '예외',
    ]

    def analyze(
        self,
        text: str,
        config: QualityConfig,
        context: Dict[str, Any]
    ) -> DimensionScore:
        issues = self._check_errors(text, config.language)
        has_error_keywords = self._has_error_keywords(text)

        score = 1.0
        score -= 0.1 * len(issues)
        if has_error_keywords:
            score -= 0.3

        return DimensionScore(
            dimension=QualityDimension.LANGUAGE,
            score=max(0.0, min(1.0, score)),
            weight=config.weights.get(QualityDimension.LANGUAGE, 1.0),
            details={
                "error_patterns_found": len(issues),
                "has_error_keywords": has_error_keywords,
            }
        )

    def get_issues(
        self,
        text: str,
        config: QualityConfig,
        context: Dict[str, Any]
    ) -> List[QualityIssue]:
        issues = []

        for pattern, message in self.ERROR_PATTERNS_KO:
            if re.search(pattern, text):
                issues.append(QualityIssue(
                    dimension=QualityDimension.LANGUAGE,
                    severity=IssueSeverity.MINOR,
                    message=f"언어 품질 이슈: {message}",
                    suggestion="정제된 표현을 사용하세요",
                    auto_fixable=True,
                ))

        if self._has_error_keywords(text):
            issues.append(QualityIssue(
                dimension=QualityDimension.LANGUAGE,
                severity=IssueSeverity.CRITICAL,
                message="에러 메시지가 응답에 포함되어 있습니다",
                suggestion="에러 내용을 제거하고 사용자 친화적인 메시지로 변경하세요",
            ))

        return issues

    def _check_errors(self, text: str, language: str) -> List[str]:
        issues = []
        for pattern, message in self.ERROR_PATTERNS_KO:
            if re.search(pattern, text):
                issues.append(message)
        return issues

    def _has_error_keywords(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.ERROR_KEYWORDS)


class FormattingAnalyzer(BaseAnalyzer):
    """포맷팅 분석기"""

    def analyze(
        self,
        text: str,
        config: QualityConfig,
        context: Dict[str, Any]
    ) -> DimensionScore:
        # 일관된 포맷팅 검사
        issues = self._check_formatting(text)

        # 마크다운 포맷 검사
        has_markdown = bool(re.search(r'[*_`#\[\]]', text))
        markdown_balanced = self._check_markdown_balance(text) if has_markdown else True

        score = 1.0
        score -= 0.1 * len(issues)
        if has_markdown and not markdown_balanced:
            score -= 0.2

        return DimensionScore(
            dimension=QualityDimension.FORMATTING,
            score=max(0.0, min(1.0, score)),
            weight=config.weights.get(QualityDimension.FORMATTING, 0.8),
            details={
                "has_markdown": has_markdown,
                "markdown_balanced": markdown_balanced,
                "formatting_issues": len(issues),
            }
        )

    def get_issues(
        self,
        text: str,
        config: QualityConfig,
        context: Dict[str, Any]
    ) -> List[QualityIssue]:
        issues = []

        # 마크다운 밸런스 검사
        if not self._check_markdown_balance(text):
            issues.append(QualityIssue(
                dimension=QualityDimension.FORMATTING,
                severity=IssueSeverity.MINOR,
                message="마크다운 포맷이 올바르지 않습니다",
                suggestion="여닫는 기호를 확인하세요",
                auto_fixable=True,
            ))

        # 불필요한 공백
        if re.search(r'  +', text):
            issues.append(QualityIssue(
                dimension=QualityDimension.FORMATTING,
                severity=IssueSeverity.INFO,
                message="불필요한 연속 공백이 있습니다",
                auto_fixable=True,
            ))

        return issues

    def _check_formatting(self, text: str) -> List[str]:
        issues = []

        # 연속 공백
        if re.search(r'  +', text):
            issues.append("consecutive_spaces")

        # 연속 줄바꿈
        if re.search(r'\n{4,}', text):
            issues.append("excessive_newlines")

        return issues

    def _check_markdown_balance(self, text: str) -> bool:
        # 간단한 마크다운 밸런스 검사
        for char in ['*', '_', '`']:
            count = text.count(char)
            if count % 2 != 0:
                return False
        return True


# ============================================================================
# Main Quality Filter
# ============================================================================

class QualityFilter:
    """
    품질 필터 - Production Grade v2.0

    다차원 품질 분석:
    - 길이 (Length)
    - 완결성 (Completeness)
    - 가독성 (Readability)
    - 구조 (Structure)
    - 언어 (Language)
    - 포맷팅 (Formatting)
    """

    def __init__(self, config: Optional[QualityConfig] = None):
        self.config = config or QualityConfig()
        self.analyzers: Dict[QualityDimension, BaseAnalyzer] = {
            QualityDimension.LENGTH: LengthAnalyzer(),
            QualityDimension.COMPLETENESS: CompletenessAnalyzer(),
            QualityDimension.READABILITY: ReadabilityAnalyzer(),
            QualityDimension.STRUCTURE: StructureAnalyzer(),
            QualityDimension.LANGUAGE: LanguageAnalyzer(),
            QualityDimension.FORMATTING: FormattingAnalyzer(),
        }
        self._custom_validators: List[Callable[[str, Dict[str, Any]], List[QualityIssue]]] = []

    def register_validator(
        self,
        validator: Callable[[str, Dict[str, Any]], List[QualityIssue]]
    ) -> None:
        """커스텀 검증기 등록"""
        self._custom_validators.append(validator)

    def validate(
        self,
        response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> QualityResult:
        """
        품질 검증 수행

        Args:
            response: 검증할 응답 텍스트
            context: 추가 컨텍스트

        Returns:
            QualityResult: 검증 결과
        """
        context = context or {}

        # 빈 응답 처리
        if not response or not response.strip():
            return QualityResult(
                valid=False,
                score=0.0,
                level=QualityLevel.UNACCEPTABLE,
                issues=[QualityIssue(
                    dimension=QualityDimension.LENGTH,
                    severity=IssueSeverity.CRITICAL,
                    message="응답이 비어 있습니다",
                )],
                warnings=[],
                dimension_scores={},
                suggestions=["응답을 작성하세요"],
            )

        # 각 차원별 분석
        dimension_scores: Dict[QualityDimension, DimensionScore] = {}
        all_issues: List[QualityIssue] = []

        for dimension, analyzer in self.analyzers.items():
            try:
                score = analyzer.analyze(response, self.config, context)
                dimension_scores[dimension] = score

                issues = analyzer.get_issues(response, self.config, context)
                all_issues.extend(issues)
            except Exception as e:
                logger.error(f"분석 오류 [{dimension.value}]: {e}")

        # 커스텀 검증기 실행
        for validator in self._custom_validators:
            try:
                custom_issues = validator(response, context)
                all_issues.extend(custom_issues)
            except Exception as e:
                logger.error(f"커스텀 검증기 오류: {e}")

        # 가중 평균 점수 계산
        total_weight = sum(ds.weight for ds in dimension_scores.values())
        weighted_score = sum(
            ds.score * ds.weight for ds in dimension_scores.values()
        ) / total_weight if total_weight > 0 else 0.0

        # 심각한 이슈가 있으면 점수 감소
        critical_count = sum(1 for i in all_issues if i.severity == IssueSeverity.CRITICAL)
        major_count = sum(1 for i in all_issues if i.severity == IssueSeverity.MAJOR)

        final_score = weighted_score
        final_score -= critical_count * 0.2
        final_score -= major_count * 0.1
        final_score = max(0.0, min(1.0, final_score))

        # 품질 레벨 결정
        level = self._determine_level(final_score)

        # 유효성 판정
        valid = (
            critical_count == 0 and
            final_score >= 0.5
        )

        # 경고 및 제안 생성
        warnings = [
            f"[{issue.dimension.value}] {issue.message}"
            for issue in all_issues
            if issue.severity in (IssueSeverity.MINOR, IssueSeverity.INFO)
        ]

        suggestions = list(set(
            issue.suggestion
            for issue in all_issues
            if issue.suggestion
        ))

        return QualityResult(
            valid=valid,
            score=round(final_score, 3),
            level=level,
            issues=all_issues,
            warnings=warnings,
            dimension_scores=dimension_scores,
            suggestions=suggestions,
            metadata={
                "analyzed_at": datetime.now().isoformat(),
                "text_length": len(response),
                "config": {
                    "language": self.config.language,
                    "min_length": self.config.min_length,
                    "max_length": self.config.max_length,
                }
            }
        )

    def _determine_level(self, score: float) -> QualityLevel:
        """점수 기반 품질 레벨 결정"""
        if score >= 0.9:
            return QualityLevel.EXCELLENT
        elif score >= 0.7:
            return QualityLevel.GOOD
        elif score >= 0.5:
            return QualityLevel.ACCEPTABLE
        elif score >= 0.3:
            return QualityLevel.POOR
        else:
            return QualityLevel.UNACCEPTABLE

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
        filter_instance = QualityFilter()
        result = filter_instance.validate(response, context or {})

        # 하위 호환성을 위한 간단한 형식
        return {
            'valid': result.valid,
            'issues': [i.message for i in result.issues if i.severity == IssueSeverity.CRITICAL],
            'warnings': result.warnings,
            'score': result.score,
        }
