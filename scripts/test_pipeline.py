"""
파이프라인 테스트 스크립트
샘플 데이터로 파이프라인 각 단계를 테스트

Usage:
    python scripts/test_pipeline.py
"""
import asyncio
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.data_pipeline.domain.models import (
    PlatformType,
    ContentType,
    ActorDTO,
    ContentDTO,
    InteractionDTO,
)
from app.data_pipeline.adapters import (
    InstagramAdapter,
    YouTubeAdapter,
    TikTokAdapter,
    TwitterAdapter,
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# 샘플 Instagram 데이터 (Apify 크롤러 형식)
SAMPLE_INSTAGRAM_DATA = {
    "id": "3123456789012345678",
    "type": "Image",
    "url": "https://www.instagram.com/p/ABC123/",
    "caption": "Amazing sunset at the beach! #sunset #beach #travel #photography",
    "timestamp": "2024-01-15T18:30:00.000Z",
    "displayUrl": "https://instagram.fxyz1-1.fna.fbcdn.net/v/t51.2885-15/123456789.jpg",
    "ownerId": "123456789",
    "ownerUsername": "travel_photographer",
    "ownerFullName": "John Doe",
    "likesCount": 1234,
    "commentsCount": 56,
    "videoViewCount": 0,
    "locationName": "Santa Monica Beach",
    "alt": "A beautiful sunset over the ocean",
    "dimensionsWidth": 1080,
    "dimensionsHeight": 1350,
    "latestComments": [
        {
            "id": "17890123456789012",
            "text": "Beautiful shot! 📸",
            "ownerUsername": "photo_lover",
            "timestamp": 1705342800,
            "likesCount": 10,
        },
        {
            "id": "17890123456789013",
            "text": "Where was this taken?",
            "ownerUsername": "curious_traveler",
            "timestamp": 1705343400,
            "likesCount": 3,
        },
    ],
}


async def test_adapter():
    """어댑터 테스트"""
    logger.info("=" * 50)
    logger.info("Testing Instagram Adapter")
    logger.info("=" * 50)

    adapter = InstagramAdapter()

    # 데이터 유효성 검증
    is_valid = adapter.validate_raw_data(SAMPLE_INSTAGRAM_DATA)
    logger.info(f"Data validation: {is_valid}")

    if not is_valid:
        logger.error("Invalid sample data")
        return False

    # 데이터 변환
    transformed = adapter.transform(SAMPLE_INSTAGRAM_DATA)

    # Actor 출력
    actor: ActorDTO = transformed["actor"]
    logger.info(f"\n📱 Actor:")
    logger.info(f"  - Platform: {actor.platform.value}")
    logger.info(f"  - Username: {actor.username}")
    logger.info(f"  - Display Name: {actor.display_name}")
    logger.info(f"  - Profile URL: {actor.profile_url}")

    # Content 출력
    content: ContentDTO = transformed["content"]
    logger.info(f"\n📝 Content:")
    logger.info(f"  - ID: {content.content_id}")
    logger.info(f"  - Type: {content.content_type.value}")
    logger.info(f"  - Text: {content.text[:100]}...")
    logger.info(f"  - URL: {content.url}")
    logger.info(f"  - Created: {content.created_at}")
    logger.info(f"  - Likes: {content.like_count}")
    logger.info(f"  - Comments: {content.comment_count}")
    logger.info(f"  - Hashtags: {', '.join(content.hashtags)}")
    logger.info(f"  - Location: {content.location}")

    # Interactions 출력
    interactions: list[InteractionDTO] = transformed["interactions"]
    logger.info(f"\n💬 Interactions ({len(interactions)}):")
    for i, interaction in enumerate(interactions, 1):
        logger.info(f"  {i}. @{interaction.author.username}: {interaction.text}")
        logger.info(f"     Likes: {interaction.like_count}")

    logger.info("\n✅ Adapter test passed!")
    return True


async def test_json_serialization():
    """JSON 직렬화 테스트"""
    logger.info("=" * 50)
    logger.info("Testing JSON Serialization")
    logger.info("=" * 50)

    from dataclasses import asdict
    from enum import Enum

    adapter = InstagramAdapter()
    transformed = adapter.transform(SAMPLE_INSTAGRAM_DATA)

    # DTO를 딕셔너리로 변환
    def dto_to_dict(obj):
        """DTO를 JSON 직렬화 가능한 딕셔너리로 변환"""
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: dto_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [dto_to_dict(item) for item in obj]
        elif hasattr(obj, "__dataclass_fields__"):
            # dataclass인 경우
            return {k: dto_to_dict(v) for k, v in asdict(obj).items()}
        else:
            return obj

    # 직렬화
    serialized = {
        "actor": dto_to_dict(transformed["actor"]),
        "content": dto_to_dict(transformed["content"]),
        "interactions": [dto_to_dict(i) for i in transformed["interactions"]],
    }

    # JSON 출력
    json_str = json.dumps(serialized, indent=2, ensure_ascii=False)
    logger.info(f"\n📄 Serialized JSON:\n{json_str[:500]}...\n")

    logger.info("✅ JSON serialization test passed!")
    return True


async def main():
    """메인 테스트 함수"""
    logger.info("🚀 Starting pipeline tests\n")

    tests = [
        ("Adapter Test", test_adapter),
        ("JSON Serialization Test", test_json_serialization),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
            logger.info("")
        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            results.append((test_name, False))
            logger.info("")

    # 결과 요약
    logger.info("=" * 50)
    logger.info("Test Results Summary")
    logger.info("=" * 50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} - {test_name}")

    logger.info(f"\n{passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed!")
        sys.exit(0)
    else:
        logger.error("❌ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
