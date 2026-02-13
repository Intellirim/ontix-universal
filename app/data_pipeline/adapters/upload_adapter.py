"""
Upload Adapter
CSV/JSON 파일 업로드 데이터를 공통 포맷으로 변환

지원 소스 타입:
- generic: 텍스트 기반 범용 데이터 (리뷰, 피드백, 설문 등)
- reviews: 고객 리뷰 (텍스트 + 평점)
- social_export: SNS 분석 내보내기 (Meta Business Suite, TikTok Analytics 등)
- platform_native: Apify 호환 JSON (기존 플랫폼 어댑터로 라우팅)

CSV 자동 컬럼 매핑:
- text/content/body/review/comment/caption → 텍스트
- author/username/user/name/reviewer → 작성자
- date/created_at/timestamp/time/posted_at → 날짜
- likes/like_count/hearts/reactions → 좋아요
- rating/score/stars → 평점 (metadata)
- url/link/source_url → URL
- location/place/city → 위치
"""
import csv
import io
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from .base import BaseSNSAdapter
from ..domain.models import (
    ActorDTO,
    ContentDTO,
    InteractionDTO,
    PlatformType,
    ContentType,
)

logger = logging.getLogger(__name__)

# 자동 컬럼 매핑 테이블
COLUMN_MAPPINGS = {
    "text": ["text", "content", "body", "review", "comment", "caption", "message",
             "description", "feedback", "note", "review_text", "comment_text",
             "post_text", "post_content", "review_body"],
    "author": ["author", "username", "user", "name", "reviewer", "commenter",
               "user_name", "author_name", "display_name", "screen_name",
               "reviewer_name", "posted_by"],
    "date": ["date", "created_at", "timestamp", "time", "posted_at",
             "review_date", "comment_date", "created", "datetime",
             "published_at", "post_date"],
    "likes": ["likes", "like_count", "hearts", "reactions", "upvotes",
              "likes_count", "favorite_count", "thumbs_up"],
    "comments": ["comments", "comment_count", "replies", "reply_count",
                 "comments_count", "num_comments"],
    "shares": ["shares", "share_count", "retweets", "reposts",
               "shares_count", "num_shares"],
    "views": ["views", "view_count", "impressions", "plays",
              "views_count", "num_views", "reach"],
    "rating": ["rating", "score", "stars", "grade", "review_rating",
               "star_rating", "review_score"],
    "url": ["url", "link", "source_url", "post_url", "permalink",
            "review_url", "original_url"],
    "location": ["location", "place", "city", "region", "store",
                 "branch", "outlet", "venue"],
    "id": ["id", "post_id", "review_id", "comment_id", "content_id",
           "item_id", "entry_id"],
    "hashtags": ["hashtags", "tags", "keywords", "labels", "categories"],
    "source": ["source", "platform", "channel", "medium", "origin",
               "source_platform"],
    "sentiment": ["sentiment", "polarity", "mood", "emotion", "tone"],
}


def _detect_column(headers: List[str], field_type: str) -> Optional[str]:
    """CSV 헤더에서 필드 타입에 맞는 컬럼명 자동 감지"""
    candidates = COLUMN_MAPPINGS.get(field_type, [])
    headers_lower = {h.lower().strip(): h for h in headers}

    for candidate in candidates:
        if candidate in headers_lower:
            return headers_lower[candidate]

    return None


def _generate_id(text: str, index: int) -> str:
    """텍스트 기반 고유 ID 생성"""
    content = f"{text[:100]}_{index}" if text else f"upload_{index}"
    return hashlib.md5(content.encode()).hexdigest()[:16]


def _parse_date(value: Any) -> Optional[datetime]:
    """다양한 날짜 형식 파싱"""
    if not value:
        return None
    if isinstance(value, datetime):
        return value

    date_str = str(value).strip()
    formats = [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    # ISO format fallback
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _parse_int(value: Any, default: int = 0) -> int:
    """안전한 정수 파싱"""
    if value is None:
        return default
    try:
        return int(float(str(value).replace(",", "")))
    except (ValueError, TypeError):
        return default


def _parse_list(value: Any) -> List[str]:
    """문자열 또는 리스트를 리스트로 변환"""
    if not value:
        return []
    if isinstance(value, list):
        return value
    # "tag1, tag2, tag3" or "tag1;tag2;tag3"
    text = str(value)
    if "," in text:
        return [t.strip() for t in text.split(",") if t.strip()]
    if ";" in text:
        return [t.strip() for t in text.split(";") if t.strip()]
    return [text.strip()] if text.strip() else []


def parse_csv_file(content: bytes, encoding: str = "utf-8") -> List[Dict[str, Any]]:
    """CSV 파일을 딕셔너리 리스트로 파싱"""
    # BOM 처리
    if content.startswith(b'\xef\xbb\xbf'):
        content = content[3:]

    # 인코딩 시도
    for enc in [encoding, "utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1"]:
        try:
            text = content.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    else:
        raise ValueError("Unable to decode CSV file. Supported encodings: UTF-8, CP949, EUC-KR")

    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)

    if not rows:
        raise ValueError("CSV file is empty or has no data rows")

    return rows


def parse_json_file(content: bytes) -> List[Dict[str, Any]]:
    """JSON 파일을 딕셔너리 리스트로 파싱"""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    # 단일 객체 → 리스트 변환
    if isinstance(data, dict):
        # {"results": [...]} 같은 래퍼 처리
        for key in ["results", "data", "items", "records", "rows", "reviews", "posts"]:
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("JSON must be an array or object")


class UploadAdapter(BaseSNSAdapter):
    """범용 업로드 데이터 어댑터"""

    # UPLOAD이 PlatformType에 없으므로 None으로 초기화, 런타임에 설정
    def __init__(self, platform: PlatformType = PlatformType.INSTAGRAM):
        super().__init__(platform)
        self._column_map: Dict[str, Optional[str]] = {}

    def set_column_map(self, headers: List[str]) -> Dict[str, Optional[str]]:
        """CSV 헤더에서 자동 컬럼 매핑 감지"""
        self._column_map = {}
        for field_type in COLUMN_MAPPINGS:
            self._column_map[field_type] = _detect_column(headers, field_type)

        detected = {k: v for k, v in self._column_map.items() if v}
        logger.info(f"Auto-detected column mappings: {detected}")

        if not self._column_map.get("text"):
            logger.warning("No text column detected. Will use first string column.")
            # fallback: 첫 번째 컬럼을 text로 사용
            if headers:
                self._column_map["text"] = headers[0]

        return self._column_map

    def _get_field(self, raw_data: Dict[str, Any], field_type: str, default: Any = None) -> Any:
        """매핑된 컬럼에서 값 추출"""
        col = self._column_map.get(field_type)
        if col and col in raw_data:
            return raw_data[col]
        return default

    def validate_raw_data(self, raw_data: Dict[str, Any]) -> bool:
        """업로드 데이터 유효성 검증 — 텍스트 필드 존재 확인"""
        text = self._get_field(raw_data, "text")
        if text and str(text).strip():
            return True
        # fallback: 아무 문자열 값이 있는지 확인
        for v in raw_data.values():
            if isinstance(v, str) and len(v.strip()) > 5:
                return True
        return False

    def parse_actor(self, raw_data: Dict[str, Any]) -> ActorDTO:
        """업로드 데이터에서 ActorDTO 추출"""
        author = self._get_field(raw_data, "author", "unknown")
        author_str = str(author).strip() or "unknown"

        return ActorDTO(
            platform=self.platform,
            actor_id=hashlib.md5(author_str.encode()).hexdigest()[:12],
            username=author_str,
            display_name=author_str if author_str != "unknown" else None,
            metadata={
                "source": "upload",
                "original_source": self._get_field(raw_data, "source", ""),
            },
        )

    def parse_content(self, raw_data: Dict[str, Any]) -> ContentDTO:
        """업로드 데이터에서 ContentDTO 추출"""
        text = str(self._get_field(raw_data, "text", "")).strip()

        # ID 결정
        content_id = self._get_field(raw_data, "id")
        if not content_id:
            content_id = _generate_id(text, id(raw_data))
        content_id = f"upload_{content_id}"

        # 메타데이터
        metadata = {"source": "upload"}
        rating = self._get_field(raw_data, "rating")
        if rating is not None:
            try:
                metadata["rating"] = float(str(rating))
            except (ValueError, TypeError):
                pass

        sentiment = self._get_field(raw_data, "sentiment")
        if sentiment:
            metadata["sentiment"] = str(sentiment)

        source = self._get_field(raw_data, "source")
        if source:
            metadata["original_platform"] = str(source)

        # 원본 행에서 매핑되지 않은 필드도 metadata에 보존
        mapped_cols = set(v for v in self._column_map.values() if v)
        for k, v in raw_data.items():
            if k not in mapped_cols and v is not None and str(v).strip():
                metadata[f"extra_{k}"] = str(v)

        actor = self.parse_actor(raw_data)

        return ContentDTO(
            platform=self.platform,
            content_id=str(content_id),
            content_type=ContentType.POST,
            author=actor,
            text=text,
            url=str(self._get_field(raw_data, "url", "")),
            created_at=_parse_date(self._get_field(raw_data, "date")),
            like_count=_parse_int(self._get_field(raw_data, "likes")),
            comment_count=_parse_int(self._get_field(raw_data, "comments")),
            share_count=_parse_int(self._get_field(raw_data, "shares")),
            view_count=_parse_int(self._get_field(raw_data, "views")),
            hashtags=_parse_list(self._get_field(raw_data, "hashtags")),
            location=self._get_field(raw_data, "location"),
            metadata=metadata,
        )

    def parse_interactions(self, raw_data: Dict[str, Any]) -> List[InteractionDTO]:
        """업로드 데이터에서 InteractionDTO 추출 (단일 행 = 단일 컨텐츠, 인터랙션 없음)"""
        return []

    def convert_rows(
        self,
        rows: List[Dict[str, Any]],
        column_overrides: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        CSV/JSON 행 리스트를 파이프라인 raw_data 형식으로 변환.

        이 메서드의 출력은 pipeline.run(raw_data=...) 에 직접 전달 가능.
        UploadAdapter가 TRANSFORM 단계에서 ContentDTO로 변환함.

        Args:
            rows: 파싱된 CSV/JSON 행 리스트
            column_overrides: 수동 컬럼 매핑 (e.g., {"text": "review_body"})

        Returns:
            파이프라인 호환 raw_data 리스트
        """
        if not rows:
            return []

        # 헤더 감지
        headers = list(rows[0].keys())
        self.set_column_map(headers)

        # 수동 오버라이드 적용
        if column_overrides:
            for field_type, col_name in column_overrides.items():
                if col_name in headers:
                    self._column_map[field_type] = col_name

        # 각 행을 raw_data 형식으로 보존 (TRANSFORM에서 처리)
        # UploadAdapter.transform()이 이 데이터를 ContentDTO로 변환
        valid_rows = []
        skipped = 0
        for row in rows:
            if self.validate_raw_data(row):
                valid_rows.append(row)
            else:
                skipped += 1

        if skipped:
            logger.info(f"Skipped {skipped} invalid rows (no text content)")

        logger.info(f"Converted {len(valid_rows)} rows for pipeline processing")
        return valid_rows
