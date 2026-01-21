"""
SNS 데이터 동기화 CLI
범용 SNS 데이터 수집 및 지식그래프 생성

Usage:
    # Instagram
    python scripts/sync_sns.py --platform instagram --actor-id apify/instagram-scraper --username <username>
    python scripts/sync_sns.py --platform instagram --actor-id apify/instagram-scraper --hashtag <hashtag>

    # YouTube
    python scripts/sync_sns.py --platform youtube --actor-id streamers/youtube-scraper --video-id <video_id>
    python scripts/sync_sns.py --platform youtube --actor-id streamers/youtube-scraper --channel-id <channel_id>
    python scripts/sync_sns.py --platform youtube --actor-id streamers/youtube-scraper --search-query <query>

    # TikTok
    python scripts/sync_sns.py --platform tiktok --actor-id clockworks/tiktok-scraper --tiktok-username <username>
    python scripts/sync_sns.py --platform tiktok --actor-id clockworks/tiktok-scraper --tiktok-hashtag <hashtag>
    python scripts/sync_sns.py --platform tiktok --actor-id clockworks/tiktok-scraper --tiktok-video-url <url>

    # Twitter
    python scripts/sync_sns.py --platform twitter --actor-id quacker/twitter-scraper --twitter-username <username>
    python scripts/sync_sns.py --platform twitter --actor-id quacker/twitter-scraper --twitter-search <query>
    python scripts/sync_sns.py --platform twitter --actor-id quacker/twitter-scraper --tweet-ids <id1> <id2>
"""
import asyncio
import argparse
import logging
import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.data_pipeline.pipeline import SNSDataPipeline
from app.data_pipeline.domain.models import PlatformType

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="SNS 데이터 수집 및 지식그래프 생성"
    )

    parser.add_argument(
        "--platform",
        type=str,
        required=True,
        choices=["instagram", "youtube", "tiktok", "twitter"],
        help="SNS 플랫폼",
    )

    parser.add_argument(
        "--actor-id",
        type=str,
        required=True,
        help="Apify Actor ID (예: apify/instagram-scraper)",
    )

    # Instagram 옵션
    parser.add_argument(
        "--username",
        type=str,
        help="Instagram 사용자명",
    )

    parser.add_argument(
        "--hashtag",
        type=str,
        help="Instagram 해시태그",
    )

    # YouTube 옵션
    parser.add_argument(
        "--video-id",
        type=str,
        help="YouTube 비디오 ID",
    )

    parser.add_argument(
        "--channel-id",
        type=str,
        help="YouTube 채널 ID",
    )

    parser.add_argument(
        "--search-query",
        type=str,
        help="YouTube 검색어",
    )

    # TikTok 옵션
    parser.add_argument(
        "--tiktok-username",
        type=str,
        help="TikTok 사용자명",
    )

    parser.add_argument(
        "--tiktok-hashtag",
        type=str,
        help="TikTok 해시태그",
    )

    parser.add_argument(
        "--tiktok-video-url",
        type=str,
        help="TikTok 비디오 URL",
    )

    # Twitter 옵션
    parser.add_argument(
        "--twitter-username",
        type=str,
        help="Twitter 사용자명",
    )

    parser.add_argument(
        "--twitter-search",
        type=str,
        help="Twitter 검색어",
    )

    parser.add_argument(
        "--tweet-ids",
        type=str,
        nargs="+",
        help="Twitter 트윗 ID 목록",
    )

    parser.add_argument(
        "--include-replies",
        action="store_true",
        help="답글 포함 (Twitter)",
    )

    # 공통 옵션
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="수집할 최대 게시물 수 (기본값: 10)",
    )

    parser.add_argument(
        "--apify-token",
        type=str,
        default=os.getenv("APIFY_API_TOKEN"),
        help="Apify API 토큰 (환경변수: APIFY_API_TOKEN)",
    )

    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j URI (환경변수: NEO4J_URI)",
    )

    parser.add_argument(
        "--neo4j-username",
        type=str,
        default=os.getenv("NEO4J_USERNAME", "neo4j"),
        help="Neo4j 사용자명 (환경변수: NEO4J_USERNAME)",
    )

    parser.add_argument(
        "--neo4j-password",
        type=str,
        default=os.getenv("NEO4J_PASSWORD"),
        help="Neo4j 비밀번호 (환경변수: NEO4J_PASSWORD)",
    )

    return parser.parse_args()


def build_actor_input(args) -> dict:
    """플랫폼별 Actor 입력 생성"""
    platform = args.platform

    if platform == "instagram":
        input_data = {
            "resultsLimit": args.limit,
        }

        if args.username:
            input_data["username"] = [args.username]
        elif args.hashtag:
            input_data["hashtags"] = [args.hashtag]
        else:
            raise ValueError("Instagram requires --username or --hashtag")

        return input_data

    elif platform == "youtube":
        input_data = {
            "maxResults": args.limit,
        }

        if args.video_id:
            input_data["videoIds"] = [args.video_id]
        elif args.channel_id:
            input_data["channelIds"] = [args.channel_id]
        elif args.search_query:
            input_data["searchKeywords"] = args.search_query
        else:
            raise ValueError("YouTube requires --video-id, --channel-id, or --search-query")

        return input_data

    elif platform == "tiktok":
        input_data = {
            "resultsPerPage": args.limit,
        }

        if args.tiktok_username:
            input_data["profiles"] = [args.tiktok_username]
        elif args.tiktok_hashtag:
            input_data["hashtags"] = [args.tiktok_hashtag]
        elif args.tiktok_video_url:
            input_data["postURLs"] = [args.tiktok_video_url]
        else:
            raise ValueError("TikTok requires --tiktok-username, --tiktok-hashtag, or --tiktok-video-url")

        return input_data

    elif platform == "twitter":
        input_data = {
            "maxTweets": args.limit,
            "includeReplies": args.include_replies,
        }

        if args.twitter_username:
            input_data["handles"] = [args.twitter_username]
        elif args.twitter_search:
            input_data["searchTerms"] = [args.twitter_search]
        elif args.tweet_ids:
            input_data["tweetIDs"] = args.tweet_ids
        else:
            raise ValueError("Twitter requires --twitter-username, --twitter-search, or --tweet-ids")

        return input_data

    else:
        raise ValueError(f"Unsupported platform: {platform}")


async def main():
    """메인 실행 함수"""
    args = parse_args()

    # 필수 설정 확인
    if not args.apify_token:
        logger.error("APIFY_API_TOKEN is required")
        sys.exit(1)

    if not args.neo4j_password:
        logger.error("NEO4J_PASSWORD is required")
        sys.exit(1)

    # 플랫폼 타입 변환
    platform = PlatformType(args.platform)

    # Actor 입력 생성
    actor_input = build_actor_input(args)

    logger.info(f"Starting SNS data sync for {platform.value}")
    logger.info(f"Actor: {args.actor_id}")
    logger.info(f"Input: {actor_input}")

    # 파이프라인 초기화
    pipeline = SNSDataPipeline(
        apify_token=args.apify_token,
        neo4j_uri=args.neo4j_uri,
        neo4j_username=args.neo4j_username,
        neo4j_password=args.neo4j_password,
    )

    try:
        # 파이프라인 실행
        success = await pipeline.run(
            platform=platform,
            actor_id=args.actor_id,
            **actor_input,
        )

        if success:
            logger.info("✅ SNS data sync completed successfully")
            sys.exit(0)
        else:
            logger.error("❌ SNS data sync failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)

    finally:
        await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())
