"""
Content Generation API
브랜드 맞춤 콘텐츠 생성 엔드포인트
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
import logging
import json

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/content")


# ============================================================
# Request/Response Models
# ============================================================

class ContentGenerationRequest(BaseModel):
    """콘텐츠 생성 요청"""
    brand_id: str = Field(..., description="브랜드 ID")
    platform: Literal["instagram", "twitter", "facebook", "linkedin", "blog"] = Field(..., description="타겟 플랫폼")
    content_type: Literal["post", "caption", "ad", "article"] = Field(..., description="콘텐츠 유형")
    topic: str = Field(..., description="주제/키워드")
    tone: Literal["professional", "friendly", "playful", "inspiring", "urgent"] = Field("professional", description="톤앤매너")
    reference_content_ids: Optional[List[str]] = Field(None, description="참조할 기존 콘텐츠 ID")


class GeneratedContentResponse(BaseModel):
    """생성된 콘텐츠 응답"""
    brand_id: str
    content: str
    hashtags: List[str]
    platform: str
    content_type: str
    metadata: dict


# ============================================================
# Platform Configurations
# ============================================================

PLATFORM_CONFIG = {
    "instagram": {"name": "Instagram", "max_length": 2200, "hashtag_limit": 30},
    "twitter": {"name": "Twitter/X", "max_length": 280, "hashtag_limit": 5},
    "facebook": {"name": "Facebook", "max_length": 63206, "hashtag_limit": 10},
    "linkedin": {"name": "LinkedIn", "max_length": 3000, "hashtag_limit": 5},
    "blog": {"name": "Blog", "max_length": 10000, "hashtag_limit": 10},
}

CONTENT_TYPES = {
    "post": "소셜 미디어 포스트",
    "caption": "이미지/영상 캡션",
    "ad": "광고 카피",
    "article": "블로그 아티클",
}

TONE_DESCRIPTIONS = {
    "professional": "전문적이고 신뢰감 있는",
    "friendly": "친근하고 따뜻한",
    "playful": "유머러스하고 재미있는",
    "inspiring": "영감을 주는",
    "urgent": "긴급하고 행동을 촉구하는",
}


# ============================================================
# API Endpoints
# ============================================================

@router.post("/generate", response_model=GeneratedContentResponse)
async def generate_content(request: ContentGenerationRequest):
    """
    브랜드 맞춤 콘텐츠 생성

    Neo4j에서 브랜드 정보, 기존 콘텐츠, 컨셉을 가져와
    GPT로 맥락에 맞는 콘텐츠를 생성합니다.
    """
    try:
        from app.services.shared.neo4j import get_neo4j_client
        from app.services.shared.llm import get_llm_client

        neo4j = get_neo4j_client()
        llm = get_llm_client()

        # 1. 브랜드 정보 조회
        brand_info = _get_brand_info(neo4j, request.brand_id)

        # 2. 브랜드의 기존 콘텐츠 스타일 분석
        existing_contents = _get_existing_contents(neo4j, request.brand_id, request.platform, limit=10)

        # 3. 브랜드 관련 컨셉/키워드 조회
        brand_concepts = _get_brand_concepts(neo4j, request.brand_id, limit=10)

        # 4. 트렌딩 해시태그 조회
        trending_hashtags = _get_trending_hashtags(neo4j, request.brand_id)

        # 5. 콘텐츠 생성 프롬프트 구성
        prompt = _build_generation_prompt(
            request=request,
            brand_info=brand_info,
            existing_contents=existing_contents,
            brand_concepts=brand_concepts,
            trending_hashtags=trending_hashtags
        )

        logger.info(f"Content generation prompt length: {len(prompt)} chars")
        logger.info(f"Reference contents: {len(existing_contents)}, Concepts: {len(brand_concepts)}")

        # 6. LLM 직접 호출로 콘텐츠 생성 (UniversalEngine 우회)
        system_prompt = """당신은 소셜 미디어 콘텐츠 전문 크리에이터입니다.
브랜드의 기존 콘텐츠 스타일을 분석하고, 그와 동일한 톤앤매너로 고품질 콘텐츠를 생성합니다.
주어진 참고 콘텐츠의 스타일(이모지 사용, 줄바꿈, 해시태그 배치, 문장 구조)을 최대한 유사하게 따라주세요.
콘텐츠만 출력하세요. 추가 설명이나 주석 없이 바로 사용할 수 있는 완성된 콘텐츠를 작성하세요."""

        # GPT-5-mini는 reasoning tokens + output tokens를 max_tokens에서 사용
        # reasoning이 길어지면 실제 output이 비어질 수 있으므로 충분히 크게 설정
        content_text = llm.invoke(
            prompt=prompt,
            system_prompt=system_prompt,
            model_variant="full",
            max_tokens=8000  # reasoning + output 포함
        )

        # 7. 응답에서 해시태그 추출
        hashtags = _extract_hashtags(content_text)

        logger.info(f"Generated content: {len(content_text)} chars, {len(hashtags)} hashtags")

        return GeneratedContentResponse(
            brand_id=request.brand_id,
            content=content_text,
            hashtags=hashtags,
            platform=request.platform,
            content_type=request.content_type,
            metadata={
                "brand_concepts_used": brand_concepts[:5],
                "reference_contents_count": len(existing_contents),
                "tone": request.tone,
                "trending_hashtags_available": len(trending_hashtags),
            }
        )

    except Exception as e:
        logger.error(f"Content generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{brand_id}/templates")
async def get_content_templates(brand_id: str, platform: Optional[str] = None):
    """
    브랜드별 콘텐츠 템플릿 조회

    기존 고성과 콘텐츠를 분석하여 템플릿을 제공합니다.
    """
    try:
        from app.services.shared.neo4j import get_neo4j_client

        neo4j = get_neo4j_client()

        # 기존 콘텐츠에서 패턴 추출
        query = """
        MATCH (c:Content)
        WHERE c.brand_id = $brand_id
        """ + (f"AND c.platform = $platform" if platform else "") + """
        RETURN c.id as id, c.text as text, c.platform as platform, c.metrics as metrics
        ORDER BY c.created_at DESC
        LIMIT 20
        """

        params = {"brand_id": brand_id}
        if platform:
            params["platform"] = platform

        contents = neo4j.query(query, params) or []

        templates = []
        for content in contents:
            text = content.get("text", "")
            if text and len(text) > 50:
                # 콘텐츠 구조 분석
                structure = _analyze_content_structure(text)
                templates.append({
                    "id": content.get("id"),
                    "platform": content.get("platform"),
                    "preview": text[:100] + "..." if len(text) > 100 else text,
                    "structure": structure,
                    "metrics": content.get("metrics"),
                })

        return {
            "brand_id": brand_id,
            "templates": templates,
            "total": len(templates)
        }

    except Exception as e:
        logger.error(f"Get templates failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{brand_id}/suggestions")
async def get_content_suggestions(brand_id: str):
    """
    콘텐츠 주제 추천

    브랜드의 컨셉과 트렌딩 토픽을 기반으로 콘텐츠 주제를 추천합니다.
    """
    try:
        from app.services.shared.neo4j import get_neo4j_client

        neo4j = get_neo4j_client()

        # 브랜드 컨셉 조회
        concepts = _get_brand_concepts(neo4j, brand_id, limit=15)

        # 트렌딩 해시태그 조회
        hashtags = _get_trending_hashtags(neo4j, brand_id)

        # 최근 콘텐츠 주제 분석
        recent_topics = _get_recent_topics(neo4j, brand_id)

        suggestions = []

        # 컨셉 기반 추천
        for concept in concepts[:5]:
            suggestions.append({
                "topic": concept,
                "type": "concept",
                "reason": f"브랜드 핵심 컨셉 '{concept}' 관련 콘텐츠"
            })

        # 트렌딩 기반 추천
        for tag in hashtags[:3]:
            suggestions.append({
                "topic": tag.replace("#", ""),
                "type": "trending",
                "reason": f"트렌딩 해시태그 '{tag}' 활용"
            })

        # 미다룬 주제 추천
        untouched = set(concepts) - set(recent_topics)
        for topic in list(untouched)[:3]:
            suggestions.append({
                "topic": topic,
                "type": "opportunity",
                "reason": f"아직 다루지 않은 컨셉 '{topic}'"
            })

        return {
            "brand_id": brand_id,
            "suggestions": suggestions,
            "concepts": concepts,
            "trending_hashtags": hashtags
        }

    except Exception as e:
        logger.error(f"Get suggestions failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Helper Functions
# ============================================================

def _get_brand_info(neo4j, brand_id: str) -> dict:
    """브랜드 정보 조회"""
    query = """
    MATCH (b:Brand {id: $brand_id})
    RETURN b.name as name, b.description as description, b.industry as industry
    """
    result = neo4j.query(query, {"brand_id": brand_id})
    if result:
        return result[0]
    return {"name": brand_id, "description": "", "industry": ""}


def _get_existing_contents(neo4j, brand_id: str, platform: str, limit: int = 10) -> List[str]:
    """기존 콘텐츠 조회 - 플랫폼 우선, 없으면 전체에서"""
    # 먼저 해당 플랫폼 콘텐츠 조회
    query = """
    MATCH (c:Content)
    WHERE c.brand_id = $brand_id AND c.platform = $platform
    RETURN c.text as text
    ORDER BY c.created_at DESC
    LIMIT $limit
    """
    results = neo4j.query(query, {"brand_id": brand_id, "platform": platform, "limit": limit}) or []
    contents = [r.get("text", "") for r in results if r.get("text")]

    # 플랫폼별 콘텐츠가 부족하면 전체에서 가져옴
    if len(contents) < 3:
        query_all = """
        MATCH (c:Content)
        WHERE c.brand_id = $brand_id
        RETURN c.text as text
        ORDER BY c.created_at DESC
        LIMIT $limit
        """
        all_results = neo4j.query(query_all, {"brand_id": brand_id, "limit": limit}) or []
        all_contents = [r.get("text", "") for r in all_results if r.get("text")]
        # 중복 제거하며 추가
        for content in all_contents:
            if content not in contents:
                contents.append(content)
            if len(contents) >= limit:
                break

    return contents


def _get_brand_concepts(neo4j, brand_id: str, limit: int = 10) -> List[str]:
    """브랜드 관련 컨셉 조회"""
    query = """
    MATCH (b:Brand {id: $brand_id})-[:HAS_CONCEPT]->(c:Concept)
    RETURN c.name as name
    LIMIT $limit
    """
    results = neo4j.query(query, {"brand_id": brand_id, "limit": limit}) or []
    return [r.get("name", "") for r in results if r.get("name")]


def _get_trending_hashtags(neo4j, brand_id: str) -> List[str]:
    """트렌딩 해시태그 조회"""
    import re

    query = """
    MATCH (c:Content)
    WHERE c.brand_id = $brand_id
    RETURN c.text as text
    ORDER BY c.created_at DESC
    LIMIT 20
    """
    results = neo4j.query(query, {"brand_id": brand_id}) or []

    hashtag_counts = {}
    for r in results:
        text = r.get("text", "") or ""
        hashtags = re.findall(r'#(\w+)', text)
        for tag in hashtags:
            hashtag_counts[tag] = hashtag_counts.get(tag, 0) + 1

    sorted_tags = sorted(hashtag_counts.items(), key=lambda x: x[1], reverse=True)
    return [f"#{tag}" for tag, _ in sorted_tags[:10]]


def _get_recent_topics(neo4j, brand_id: str) -> List[str]:
    """최근 다룬 주제 조회"""
    query = """
    MATCH (c:Content)-[:MENTIONS_CONCEPT]->(concept:Concept)
    WHERE c.brand_id = $brand_id
    RETURN DISTINCT concept.name as name
    LIMIT 20
    """
    results = neo4j.query(query, {"brand_id": brand_id}) or []
    return [r.get("name", "") for r in results if r.get("name")]


PLATFORM_GUIDELINES = {
    "instagram": """
### Instagram 콘텐츠 가이드라인
- 첫 문장에 강력한 훅(Hook)을 사용하여 스크롤을 멈추게 하세요
- 이모지를 적절히 활용하여 시각적 재미를 더하세요 (문단 시작, 포인트 강조)
- 줄바꿈을 충분히 활용하여 가독성을 높이세요
- 캐러셀(슬라이드) 형식을 고려한 구조화된 정보 제공
- CTA(Call-to-Action): "저장해두세요", "댓글로 알려주세요", "팔로우하고 더 알아보세요"
- 해시태그는 본문 끝에 모아서 배치 (최대 30개, 권장 10-15개)
- 숫자, 리스트, 단계별 설명이 효과적
""",
    "twitter": """
### Twitter/X 콘텐츠 가이드라인
- 280자 제한! 핵심만 임팩트 있게 전달
- 첫 문장이 전부입니다 - 강렬한 주장이나 질문으로 시작
- 스레드(Thread) 형식: 1/n으로 시작하면 연속 트윗 가능
- 짧고 강한 문장, 불필요한 조사 생략
- 해시태그는 1-3개만 (본문에 자연스럽게 삽입)
- 리트윗/인용을 유도하는 공유할 만한 인사이트
""",
    "facebook": """
### Facebook 콘텐츠 가이드라인
- 스토리텔링 중심: 개인적 경험이나 사례로 시작
- 중간 길이 선호 (150-300자 권장)
- 질문으로 끝내 댓글 유도
- 이모지 사용은 적당히 (인스타그램보다 절제)
- 해시태그는 3-5개 정도
- 공유 가치가 있는 정보성 콘텐츠
""",
    "linkedin": """
### LinkedIn 콘텐츠 가이드라인
- 전문적이고 비즈니스 톤 유지
- 첫 3줄이 "더보기" 전에 보이므로 핵심을 앞에 배치
- 개인적 인사이트 + 업계 트렌드 조합
- 숫자와 데이터로 신뢰감 부여
- CTA: "어떻게 생각하시나요?", "여러분의 경험도 공유해주세요"
- 해시태그 3-5개 (관련 업계/직무 해시태그)
- 문단 사이 줄바꿈으로 가독성 확보
""",
    "blog": """
### 블로그 콘텐츠 가이드라인
- SEO를 고려한 제목과 소제목 (H2, H3 태그 구조)
- 도입부에서 독자의 문제/관심사 언급
- 본론은 구체적 정보, 단계별 가이드, 예시 포함
- 결론에서 핵심 요약 + 다음 행동 제안
- 내부/외부 링크 활용 가능
- 1000-2000자 분량 권장
- 소제목마다 명확한 주제
"""
}


def _build_generation_prompt(
    request: ContentGenerationRequest,
    brand_info: dict,
    existing_contents: List[str],
    brand_concepts: List[str],
    trending_hashtags: List[str]
) -> str:
    """콘텐츠 생성 프롬프트 구성"""

    platform_config = PLATFORM_CONFIG[request.platform]
    platform_guide = PLATFORM_GUIDELINES.get(request.platform, "")

    prompt = f"""당신은 '{brand_info.get('name', request.brand_id)}' 브랜드의 전문 콘텐츠 크리에이터입니다.
브랜드의 기존 콘텐츠 스타일을 완벽하게 학습하여, 동일한 톤앤매너로 고품질 콘텐츠를 생성해야 합니다.

═══════════════════════════════════════════════════════════
📌 브랜드 정보
═══════════════════════════════════════════════════════════
• 브랜드명: {brand_info.get('name', request.brand_id)}
• 브랜드 미션: {brand_info.get('description', '정보 없음')}
• 산업/카테고리: {brand_info.get('industry', '정보 없음')}
• 핵심 키워드: {', '.join(brand_concepts[:10]) if brand_concepts else '역노화, 안티에이징, 과학적 건강관리'}

═══════════════════════════════════════════════════════════
🎯 콘텐츠 요청사항
═══════════════════════════════════════════════════════════
• 플랫폼: {platform_config['name']}
• 콘텐츠 유형: {CONTENT_TYPES[request.content_type]}
• 톤앤매너: {TONE_DESCRIPTIONS[request.tone]}
• 주제/키워드: {request.topic}
• 최대 길이: {platform_config['max_length']}자
• 해시태그: {platform_config['hashtag_limit']}개 이내

{platform_guide}

═══════════════════════════════════════════════════════════
📚 브랜드 기존 콘텐츠 (스타일 참고용)
═══════════════════════════════════════════════════════════
아래는 이 브랜드가 실제로 발행한 {platform_config['name']} 콘텐츠입니다.
이 스타일, 어투, 이모지 사용법, 구조를 최대한 유사하게 따라주세요:

"""

    if existing_contents:
        for i, content in enumerate(existing_contents[:5], 1):
            # 더 긴 콘텐츠 참조
            preview = content[:800] if len(content) > 800 else content
            prompt += f"---[기존 콘텐츠 {i}]---\n{preview}\n\n"
    else:
        prompt += "(기존 콘텐츠 없음 - 브랜드 미션에 맞게 새롭게 작성)\n"

    prompt += f"""
═══════════════════════════════════════════════════════════
🏷️ 추천 해시태그 (브랜드에서 자주 사용)
═══════════════════════════════════════════════════════════
{', '.join(trending_hashtags[:15]) if trending_hashtags else '#역노화 #안티에이징 #건강수명 #바이오해킹 #저속노화'}

═══════════════════════════════════════════════════════════
✍️ 작성 지침
═══════════════════════════════════════════════════════════
1. 반드시 위 '기존 콘텐츠'의 스타일과 어투를 그대로 따라하세요
2. 주제 '{request.topic}'에 대해 브랜드 관점에서 깊이 있는 정보를 제공하세요
3. 과학적 근거, 연구 결과, 구체적 수치를 포함하면 신뢰도가 높아집니다
4. {platform_config['name']} 플랫폼 특성에 맞는 형식을 사용하세요
5. 해시태그는 반드시 포함하되, 자연스럽게 배치하세요
6. 독자가 행동하도록 유도하는 CTA(Call-to-Action)를 포함하세요

═══════════════════════════════════════════════════════════
📝 이제 '{request.topic}' 주제로 {platform_config['name']} {CONTENT_TYPES[request.content_type]}를 작성해주세요.
콘텐츠만 출력하세요. 추가 설명 없이 바로 사용할 수 있는 완성된 콘텐츠를 작성하세요.
═══════════════════════════════════════════════════════════
"""

    return prompt


def _extract_hashtags(text: str) -> List[str]:
    """텍스트에서 해시태그 추출"""
    import re
    hashtags = re.findall(r'#[\w가-힣]+', text)
    return list(set(hashtags))


def _analyze_content_structure(text: str) -> dict:
    """콘텐츠 구조 분석"""
    import re

    lines = text.split('\n')
    has_emoji = bool(re.search(r'[\U0001F300-\U0001F9FF]', text))
    has_hashtags = bool(re.search(r'#\w+', text))
    has_numbers = bool(re.search(r'\d+[.)]', text))
    has_links = bool(re.search(r'https?://', text))

    return {
        "line_count": len(lines),
        "char_count": len(text),
        "has_emoji": has_emoji,
        "has_hashtags": has_hashtags,
        "has_numbered_list": has_numbers,
        "has_links": has_links,
    }
