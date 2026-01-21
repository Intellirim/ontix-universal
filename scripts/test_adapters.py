#!/usr/bin/env python3
"""
ONTIX Universal - Adapter Unit Tests
어댑터 단위 테스트 스크립트

Usage:
    python scripts/test_adapters.py
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from typing import List, Tuple, Any

# Import adapters
from app.data_pipeline.adapters import (
    InstagramAdapter,
    YouTubeAdapter,
    TikTokAdapter,
    TwitterAdapter,
)
from app.data_pipeline.domain.models import (
    PlatformType,
    ContentType,
    ActorDTO,
    ContentDTO,
    InteractionDTO,
)


# =============================================================================
# SAMPLE DATA (실제 Apify Actor 출력 형식)
# =============================================================================

TWITTER_SAMPLE = {
    "url": "https://x.com/roundproxynews/status/2011040295245451577",
    "twitterUrl": "https://twitter.com/roundproxynews/status/2011040295245451577",
    "id": "2011040295245451577",
    "text": "Want to automatically extract data from websites #webscraping @testuser https://t.co/abc123",
    "retweetCount": 10,
    "replyCount": 5,
    "likeCount": 100,
    "quoteCount": 2,
    "createdAt": "Tue Jan 13 11:39:04 +0000 2026",
    "bookmarkCount": 3,
    "isRetweet": False,
    "isQuote": False,
}

YOUTUBE_SAMPLE = {
    "title": "Crawlee, the web scraping library #coding",
    "id": "g1Ll9OlFwEQ",
    "url": "https://www.youtube.com/watch?v=g1Ll9OlFwEQ",
    "viewCount": 11974,
    "date": "2022-08-22T15:26:13.000Z",
    "likes": 154,
    "channelName": "Apify",
    "channelUrl": "https://www.youtube.com/@Apify",
    "numberOfSubscribers": 11600,
    "duration": "00:03:14",
}

YOUTUBE_SHORTS_SAMPLE = {
    "title": "Quick tip! #shorts",
    "id": "shortVideoId",
    "url": "https://www.youtube.com/shorts/shortVideoId",
    "viewCount": 50000,
    "date": "2024-01-01T00:00:00.000Z",
    "likes": 1000,
    "channelName": "TestChannel",
    "channelUrl": "https://www.youtube.com/@TestChannel",
    "numberOfSubscribers": 5000,
    "duration": "00:00:45",  # 45 seconds
}

TIKTOK_SAMPLE = {
    "authorMeta.avatar": "https://example.com/avatar.jpg",
    "authorMeta.name": "themalachibarton",
    "authorMeta.nickname": "Malachi",
    "text": "Certified #fyp @Mk check this out!",
    "diggCount": 3100000,
    "shareCount": 52800,
    "playCount": 27500000,
    "commentCount": 7320,
    "collectCount": 256837,
    "videoMeta.duration": 15,
    "musicMeta.musicName": "Not Like Us",
    "musicMeta.musicAuthor": "Kendrick Lamar",
    "musicMeta.musicOriginal": False,
    "createTimeISO": "2024-05-19T01:44:18.000Z",
    "webVideoUrl": "https://www.tiktok.com/@themalachibarton/video/7370520570070338859",
}

INSTAGRAM_SAMPLE = {
    "id": "3029218572739856426",
    "type": "Sidecar",
    "shortCode": "CoJ8j4PLMgq",
    "caption": "Running up that hill #fitness @friend was great!",
    "hashtags": ["fitness"],
    "mentions": ["friend"],
    "url": "https://www.instagram.com/p/CoJ8j4PLMgq/",
    "commentsCount": 61,
    "likesCount": 500,
    "videoViewCount": 1000,
    "firstComment": "Cute",
    "ownerUsername": "testuser",
    "ownerId": "123456789",
    "displayUrl": "https://example.com/image.jpg",
    "timestamp": "2025-01-01T12:00:00.000Z",
    "latestComments": [
        {
            "id": "18073818227593872",
            "text": "Amazing!",
            "ownerUsername": "therealklopp",
            "ownerProfilePicUrl": "https://example.com/profile.jpg",
            "timestamp": "2025-12-22T17:24:48.000Z",
            "likesCount": 5,
            "repliesCount": 2,
            "replies": [],
            "owner": {
                "id": "9169100897",
                "username": "therealklopp",
                "is_verified": True,
                "profile_pic_url": "https://example.com/profile.jpg",
            },
        },
        {
            "id": "18073818227593873",
            "text": "Great work!",
            "ownerUsername": "anotheruser",
            "timestamp": "2025-12-22T18:00:00.000Z",
            "likesCount": 2,
            "repliesCount": 0,
            "owner": {"id": "111222333", "username": "anotheruser"},
        },
    ],
}

INSTAGRAM_REEL_SAMPLE = {
    "id": "reel123",
    "type": "Reel",
    "shortCode": "ReelCode",
    "caption": "Check out this reel!",
    "url": "https://www.instagram.com/reel/ReelCode/",
    "commentsCount": 10,
    "likesCount": 200,
    "videoViewCount": 5000,
}


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_twitter_adapter() -> List[Tuple[str, bool, str]]:
    """Twitter 어댑터 테스트"""
    results = []
    adapter = TwitterAdapter()

    # 1. Validation
    try:
        assert adapter.validate_raw_data(TWITTER_SAMPLE) == True
        assert adapter.validate_raw_data({}) == False
        results.append(("Twitter: validate_raw_data", True, ""))
    except AssertionError as e:
        results.append(("Twitter: validate_raw_data", False, str(e)))

    # 2. Parse Actor
    try:
        actor = adapter.parse_actor(TWITTER_SAMPLE)
        assert actor.platform == PlatformType.TWITTER
        assert actor.username == "roundproxynews"
        assert "twitter.com/roundproxynews" in actor.profile_url
        results.append(("Twitter: parse_actor", True, f"username={actor.username}"))
    except AssertionError as e:
        results.append(("Twitter: parse_actor", False, str(e)))

    # 3. Parse Content
    try:
        content = adapter.parse_content(TWITTER_SAMPLE)
        assert content.content_id == "2011040295245451577"
        assert content.content_type == ContentType.TWEET
        assert content.like_count == 100
        assert content.comment_count == 5  # replyCount
        assert content.share_count == 12  # retweetCount + quoteCount
        assert "webscraping" in content.hashtags
        assert "testuser" in content.mentions
        assert content.created_at is not None
        results.append(("Twitter: parse_content", True, f"likes={content.like_count}"))
    except AssertionError as e:
        results.append(("Twitter: parse_content", False, str(e)))

    # 4. Parse Interactions (should be empty)
    try:
        interactions = adapter.parse_interactions(TWITTER_SAMPLE)
        assert len(interactions) == 0
        results.append(("Twitter: parse_interactions", True, "empty as expected"))
    except AssertionError as e:
        results.append(("Twitter: parse_interactions", False, str(e)))

    # 5. Transform
    try:
        result = adapter.transform(TWITTER_SAMPLE)
        assert "actor" in result
        assert "content" in result
        assert "interactions" in result
        results.append(("Twitter: transform", True, ""))
    except Exception as e:
        results.append(("Twitter: transform", False, str(e)))

    return results


def test_youtube_adapter() -> List[Tuple[str, bool, str]]:
    """YouTube 어댑터 테스트"""
    results = []
    adapter = YouTubeAdapter()

    # 1. Validation
    try:
        assert adapter.validate_raw_data(YOUTUBE_SAMPLE) == True
        assert adapter.validate_raw_data({}) == False
        results.append(("YouTube: validate_raw_data", True, ""))
    except AssertionError as e:
        results.append(("YouTube: validate_raw_data", False, str(e)))

    # 2. Parse Actor
    try:
        actor = adapter.parse_actor(YOUTUBE_SAMPLE)
        assert actor.platform == PlatformType.YOUTUBE
        assert actor.username == "Apify"
        assert actor.follower_count == 11600
        results.append(("YouTube: parse_actor", True, f"subs={actor.follower_count}"))
    except AssertionError as e:
        results.append(("YouTube: parse_actor", False, str(e)))

    # 3. Parse Content (regular video)
    try:
        content = adapter.parse_content(YOUTUBE_SAMPLE)
        assert content.content_id == "g1Ll9OlFwEQ"
        assert content.content_type == ContentType.VIDEO
        assert content.view_count == 11974
        assert content.like_count == 154
        assert content.metadata.get("duration_seconds") == 194  # 3:14
        results.append(("YouTube: parse_content (video)", True, f"views={content.view_count}"))
    except AssertionError as e:
        results.append(("YouTube: parse_content (video)", False, str(e)))

    # 4. Parse Content (Shorts - under 60 seconds)
    try:
        content = adapter.parse_content(YOUTUBE_SHORTS_SAMPLE)
        assert content.content_type == ContentType.SHORT
        assert content.metadata.get("duration_seconds") == 45
        results.append(("YouTube: parse_content (short)", True, f"type={content.content_type.value}"))
    except AssertionError as e:
        results.append(("YouTube: parse_content (short)", False, str(e)))

    # 5. Parse Interactions (should be empty)
    try:
        interactions = adapter.parse_interactions(YOUTUBE_SAMPLE)
        assert len(interactions) == 0
        results.append(("YouTube: parse_interactions", True, "empty as expected"))
    except AssertionError as e:
        results.append(("YouTube: parse_interactions", False, str(e)))

    return results


def test_tiktok_adapter() -> List[Tuple[str, bool, str]]:
    """TikTok 어댑터 테스트"""
    results = []
    adapter = TikTokAdapter()

    # 1. Validation
    try:
        assert adapter.validate_raw_data(TIKTOK_SAMPLE) == True
        assert adapter.validate_raw_data({}) == False
        results.append(("TikTok: validate_raw_data", True, ""))
    except AssertionError as e:
        results.append(("TikTok: validate_raw_data", False, str(e)))

    # 2. Parse Actor (flat 구조)
    try:
        actor = adapter.parse_actor(TIKTOK_SAMPLE)
        assert actor.platform == PlatformType.TIKTOK
        assert actor.username == "themalachibarton"
        assert actor.avatar_url == "https://example.com/avatar.jpg"
        results.append(("TikTok: parse_actor", True, f"user={actor.username}"))
    except AssertionError as e:
        results.append(("TikTok: parse_actor", False, str(e)))

    # 3. Parse Content (flat 구조)
    try:
        content = adapter.parse_content(TIKTOK_SAMPLE)
        assert content.content_id == "7370520570070338859"
        assert content.like_count == 3100000  # diggCount
        assert content.view_count == 27500000  # playCount
        assert content.share_count == 52800
        assert content.comment_count == 7320
        assert "fyp" in content.hashtags
        assert "Mk" in content.mentions
        assert content.metadata.get("music_name") == "Not Like Us"
        results.append(("TikTok: parse_content", True, f"plays={content.view_count}"))
    except AssertionError as e:
        results.append(("TikTok: parse_content", False, str(e)))

    # 4. Parse Interactions (should be empty)
    try:
        interactions = adapter.parse_interactions(TIKTOK_SAMPLE)
        assert len(interactions) == 0
        results.append(("TikTok: parse_interactions", True, "empty as expected"))
    except AssertionError as e:
        results.append(("TikTok: parse_interactions", False, str(e)))

    return results


def test_instagram_adapter() -> List[Tuple[str, bool, str]]:
    """Instagram 어댑터 테스트"""
    results = []
    adapter = InstagramAdapter()

    # 1. Validation
    try:
        assert adapter.validate_raw_data(INSTAGRAM_SAMPLE) == True
        assert adapter.validate_raw_data({}) == False
        results.append(("Instagram: validate_raw_data", True, ""))
    except AssertionError as e:
        results.append(("Instagram: validate_raw_data", False, str(e)))

    # 2. Parse Actor
    try:
        actor = adapter.parse_actor(INSTAGRAM_SAMPLE)
        assert actor.platform == PlatformType.INSTAGRAM
        assert actor.username == "testuser"
        assert actor.actor_id == "123456789"  # ownerId
        results.append(("Instagram: parse_actor", True, f"user={actor.username}"))
    except AssertionError as e:
        results.append(("Instagram: parse_actor", False, str(e)))

    # 3. Parse Content (Sidecar/Post)
    try:
        content = adapter.parse_content(INSTAGRAM_SAMPLE)
        assert content.content_id == "3029218572739856426"
        assert content.content_type == ContentType.POST  # Sidecar -> POST
        assert content.like_count == 500
        assert content.comment_count == 61
        assert "fitness" in content.hashtags
        assert "friend" in content.mentions
        assert content.created_at is not None
        results.append(("Instagram: parse_content (post)", True, f"likes={content.like_count}"))
    except AssertionError as e:
        results.append(("Instagram: parse_content (post)", False, str(e)))

    # 4. Parse Content (Reel)
    try:
        content = adapter.parse_content(INSTAGRAM_REEL_SAMPLE)
        assert content.content_type == ContentType.REEL
        results.append(("Instagram: parse_content (reel)", True, f"type={content.content_type.value}"))
    except AssertionError as e:
        results.append(("Instagram: parse_content (reel)", False, str(e)))

    # 5. Parse Interactions (댓글)
    try:
        interactions = adapter.parse_interactions(INSTAGRAM_SAMPLE)
        assert len(interactions) == 2
        assert interactions[0].text == "Amazing!"
        assert interactions[0].author.username == "therealklopp"
        assert interactions[0].author.verified == True
        assert interactions[0].like_count == 5
        assert interactions[1].text == "Great work!"
        results.append(("Instagram: parse_interactions", True, f"count={len(interactions)}"))
    except AssertionError as e:
        results.append(("Instagram: parse_interactions", False, str(e)))

    return results


def print_results(all_results: List[Tuple[str, bool, str]]) -> Tuple[int, int]:
    """결과 출력"""
    passed = 0
    failed = 0

    print("\n" + "=" * 70)
    print("ADAPTER TEST RESULTS")
    print("=" * 70)

    for name, success, detail in all_results:
        status = "PASS" if success else "FAIL"
        if success:
            passed += 1
            print(f"  [{status}] {name} - {detail}")
        else:
            failed += 1
            print(f"  [{status}] {name}")
            if detail:
                print(f"         Error: {detail}")

    print("=" * 70)
    print(f"Total: {passed + failed} | Passed: {passed} | Failed: {failed}")
    print("=" * 70)

    return passed, failed


def main():
    """메인 실행"""
    print("\nONTIX Universal - Adapter Unit Tests")
    print("Testing adapters with actual Apify Actor data formats...")

    all_results = []

    # Run all tests
    all_results.extend(test_twitter_adapter())
    all_results.extend(test_youtube_adapter())
    all_results.extend(test_tiktok_adapter())
    all_results.extend(test_instagram_adapter())

    # Print results
    passed, failed = print_results(all_results)

    # Exit code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
