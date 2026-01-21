"""
Pytest Configuration and Fixtures
ontix-universal 테스트 공통 설정

Features:
- 공통 fixture 정의
- 테스트 환경 설정
- Mock 객체 제공
"""

import pytest
from typing import Dict, Any
from unittest.mock import MagicMock, patch


# ============================================================
# Brand Config Fixtures
# ============================================================

@pytest.fixture
def sample_brand_config() -> Dict[str, Any]:
    """테스트용 브랜드 설정"""
    return {
        'brand_id': 'test_brand',
        'brand_name': 'Test Brand',
        'brand_description': '테스트용 브랜드입니다.',
        'brand_industry': 'E-commerce',
        'filters': {
            'enabled': True,
            'quality': {
                'enabled': True,
                'min_score': 0.5,
            },
            'trust': {
                'enabled': True,
                'min_score': 0.5,
                'hallucination_threshold': 0.3,
            },
            'relevance': {
                'enabled': True,
                'min_score': 0.5,
            },
            'validation': {
                'enabled': True,
                'pass_threshold': 0.7,
            },
        },
        'advisor': {
            'quality_filter_enabled': True,
            'trust_filter_enabled': True,
            'relevance_filter_enabled': True,
            'validation_filter_enabled': True,
            'min_quality_score': 0.5,
            'min_trust_score': 0.5,
            'min_relevance_score': 0.5,
        },
        'retrieval': {
            'vector': {
                'top_k': 10,
                'min_score': 0.5,
            },
            'hybrid': {
                'top_k': 10,
            },
        },
    }


@pytest.fixture
def sample_context() -> Dict[str, Any]:
    """테스트용 컨텍스트"""
    return {
        'question': '테스트 질문입니다',
        'brand_id': 'test_brand',
        'retrieval_results': {
            'vector_search': [
                {'content': '테스트 내용 1', 'score': 0.9},
                {'content': '테스트 내용 2', 'score': 0.8},
            ],
        },
    }


@pytest.fixture
def sample_response() -> str:
    """테스트용 응답"""
    return """
    테스트 응답입니다. 이 응답은 사용자의 질문에 대한 답변을 제공합니다.
    충분한 길이와 내용을 포함하여 품질 테스트를 통과할 수 있도록 작성되었습니다.
    또한 신뢰할 수 있는 정보를 기반으로 작성되었습니다.
    """


@pytest.fixture
def short_response() -> str:
    """짧은 응답 (품질 테스트 실패용)"""
    return "짧은 응답"


@pytest.fixture
def hallucination_response() -> str:
    """환각 포함 응답 (신뢰성 테스트용)"""
    return """
    이 제품은 100% 효과가 보장됩니다. 모든 사용자가 반드시 만족합니다.
    절대적으로 안전하며 어떤 부작용도 없습니다.
    과학적으로 완벽하게 검증되었습니다.
    """


# ============================================================
# Filter Fixtures
# ============================================================

@pytest.fixture
def quality_filter():
    """QualityFilter 인스턴스"""
    from app.filters.quality import QualityFilter, QualityConfig
    return QualityFilter(config=QualityConfig(language="ko", min_length=20))


@pytest.fixture
def trust_filter():
    """TrustFilter 인스턴스"""
    from app.filters.trust import TrustFilter, TrustConfig
    return TrustFilter(config=TrustConfig(min_trust_score=0.5, hallucination_threshold=0.3))


@pytest.fixture
def relevance_filter():
    """RelevanceFilter 인스턴스"""
    from app.filters.relevance import RelevanceFilter, RelevanceConfig
    return RelevanceFilter(config=RelevanceConfig(min_relevance_score=0.5))


@pytest.fixture
def validation_filter():
    """ValidationFilter 인스턴스"""
    from app.filters.validation import ValidationFilter, ValidationConfig
    return ValidationFilter(config=ValidationConfig())


# ============================================================
# Mock Fixtures
# ============================================================

@pytest.fixture
def mock_neo4j_driver():
    """Neo4j 드라이버 Mock"""
    with patch('neo4j.GraphDatabase.driver') as mock_driver:
        mock_session = MagicMock()
        mock_driver.return_value.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.return_value.session.return_value.__exit__ = MagicMock(return_value=None)
        yield mock_driver


@pytest.fixture
def mock_openai_client():
    """OpenAI 클라이언트 Mock"""
    with patch('openai.OpenAI') as mock_client:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Mock response"))]
        mock_client.return_value.chat.completions.create.return_value = mock_response
        yield mock_client


@pytest.fixture
def mock_redis_client():
    """Redis 클라이언트 Mock"""
    with patch('redis.Redis') as mock_redis:
        mock_redis.return_value.get.return_value = None
        mock_redis.return_value.set.return_value = True
        yield mock_redis


# ============================================================
# Integration Test Fixtures
# ============================================================

@pytest.fixture
def sample_raw_instagram_data() -> Dict[str, Any]:
    """테스트용 Instagram 원시 데이터"""
    return {
        "id": "test_content_001",
        "shortCode": "ABC123",
        "type": "Reel",
        "caption": "테스트 콘텐츠입니다 #테스트 #복싱",
        "url": "https://www.instagram.com/p/ABC123/",
        "likesCount": 100,
        "commentsCount": 10,
        "viewCount": 5000,
        "timestamp": "2025-01-01T12:00:00.000Z",
        "ownerUsername": "test_user",
        "ownerId": "user_001",
        "ownerFullName": "Test User",
        "displayUrl": "https://example.com/image.jpg",
        "videoUrl": "https://example.com/video.mp4",
        "dimensionsWidth": 1080,
        "dimensionsHeight": 1920,
        "firstComment": "첫 번째 댓글입니다",
        "hashtags": ["테스트", "복싱"],
        "mentions": [],
        "locationName": "서울",
        "latestComments": [
            {
                "id": "comment_001",
                "text": "좋은 콘텐츠입니다!",
                "ownerUsername": "commenter1",
                "ownerProfilePicUrl": "https://example.com/avatar1.jpg",
                "timestamp": "2025-01-01T13:00:00.000Z",
                "repliesCount": 2,
                "likesCount": 5,
                "owner": {
                    "id": "commenter_001",
                    "is_verified": False,
                    "username": "commenter1",
                }
            },
            {
                "id": "comment_002",
                "text": "정말 유용해요!",
                "ownerUsername": "commenter2",
                "timestamp": "2025-01-01T14:00:00.000Z",
                "repliesCount": 0,
                "likesCount": 3,
                "owner": {
                    "id": "commenter_002",
                    "is_verified": True,
                    "username": "commenter2",
                }
            }
        ]
    }


@pytest.fixture
def sample_content_dto():
    """테스트용 ContentDTO"""
    from app.data_pipeline.domain.models import (
        ContentDTO, ContentType, PlatformType, ActorDTO
    )

    actor = ActorDTO(
        platform=PlatformType.INSTAGRAM,
        actor_id="user_001",
        username="test_user",
        display_name="Test User",
        metadata={"owner_id": "user_001"},
    )

    return ContentDTO(
        platform=PlatformType.INSTAGRAM,
        content_id="test_content_001",
        content_type=ContentType.REEL,
        author=actor,
        text="테스트 콘텐츠입니다 #테스트 #복싱",
        url="https://www.instagram.com/p/ABC123/",
        like_count=100,
        comment_count=10,
        view_count=5000,
        hashtags=["테스트", "복싱"],
        location="서울",
        metadata={
            "short_code": "ABC123",
            "type": "Reel",
            "first_comment": "첫 번째 댓글입니다",
            "dimensions": {"width": 1080, "height": 1920},
        }
    )


@pytest.fixture
def sample_interaction_dto():
    """테스트용 InteractionDTO"""
    from app.data_pipeline.domain.models import (
        InteractionDTO, PlatformType, ActorDTO
    )
    from datetime import datetime

    actor = ActorDTO(
        platform=PlatformType.INSTAGRAM,
        actor_id="commenter_001",
        username="commenter1",
    )

    return InteractionDTO(
        platform=PlatformType.INSTAGRAM,
        interaction_id="comment_001",
        content_id="test_content_001",
        author=actor,
        text="좋은 콘텐츠입니다!",
        created_at=datetime(2025, 1, 1, 13, 0, 0),
        like_count=5,
        metadata={
            "replies_count": 2,
            "sentiment": "positive",
        }
    )
