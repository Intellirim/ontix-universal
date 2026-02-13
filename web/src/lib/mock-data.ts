// ============================================
// ONTIX Dashboard Mock Data
// ============================================

import type {
  Channel,
  OntixMode,
  ChannelOverview,
  OntixAnswer,
  EngagementDataPoint,
  GraphNode,
  GraphEdge,
  ActionItem,
  Grade,
} from "@/types/dashboard";

// Generate engagement timeline data
function generateEngagementTimeline(days: number = 7): EngagementDataPoint[] {
  const data: EngagementDataPoint[] = [];
  const now = new Date();

  for (let i = days; i >= 0; i--) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);

    // Create realistic fluctuation pattern
    const baseScore = 65 + Math.sin(i * 0.5) * 15;
    const noise = (Math.random() - 0.5) * 20;

    data.push({
      timestamp: date.toISOString(),
      score: Math.round(Math.max(0, Math.min(100, baseScore + noise))),
      likes: Math.round(1200 + Math.random() * 800),
      comments: Math.round(80 + Math.random() * 120),
      saves: Math.round(150 + Math.random() * 100),
      clicks: Math.round(400 + Math.random() * 300),
    });
  }

  return data;
}

// Generate knowledge graph nodes
function generateGraphNodes(): GraphNode[] {
  const nodes: GraphNode[] = [
    // Central content nodes
    { id: "c1", type: "content", label: "신제품 런칭", x: 0.5, y: 0.5, size: 1.2 },
    { id: "c2", type: "content", label: "튜토리얼", x: 0.3, y: 0.4, size: 1.0 },
    { id: "c3", type: "content", label: "후기 리뷰", x: 0.7, y: 0.35, size: 0.9 },

    // Hashtag nodes
    { id: "h1", type: "hashtag", label: "#브랜드명", x: 0.2, y: 0.6, size: 0.8 },
    { id: "h2", type: "hashtag", label: "#신상품", x: 0.4, y: 0.7, size: 0.7 },
    { id: "h3", type: "hashtag", label: "#추천템", x: 0.6, y: 0.65, size: 0.75 },

    // Audience nodes
    { id: "a1", type: "audience", label: "2030 여성", x: 0.15, y: 0.3, size: 0.85 },
    { id: "a2", type: "audience", label: "인플루언서", x: 0.8, y: 0.55, size: 0.7 },

    // Topic nodes
    { id: "t1", type: "topic", label: "스킨케어", x: 0.35, y: 0.25, size: 0.65 },
    { id: "t2", type: "topic", label: "트렌드", x: 0.65, y: 0.8, size: 0.6 },

    // Product nodes
    { id: "p1", type: "product", label: "베스트셀러", x: 0.85, y: 0.25, size: 0.9 },
    { id: "p2", type: "product", label: "신제품A", x: 0.25, y: 0.85, size: 0.8 },
  ];

  return nodes;
}

// Generate knowledge graph edges
function generateGraphEdges(): GraphEdge[] {
  return [
    { source: "c1", target: "h1", weight: 0.9 },
    { source: "c1", target: "h2", weight: 0.85 },
    { source: "c1", target: "a1", weight: 0.7 },
    { source: "c1", target: "p1", weight: 0.95 },

    { source: "c2", target: "t1", weight: 0.8 },
    { source: "c2", target: "a1", weight: 0.75 },
    { source: "c2", target: "h1", weight: 0.6 },

    { source: "c3", target: "a2", weight: 0.85 },
    { source: "c3", target: "h3", weight: 0.7 },
    { source: "c3", target: "p1", weight: 0.8 },

    { source: "h2", target: "t2", weight: 0.65 },
    { source: "h3", target: "p2", weight: 0.7 },

    { source: "a1", target: "t1", weight: 0.55 },
    { source: "a2", target: "t2", weight: 0.6 },

    { source: "p1", target: "p2", weight: 0.4 },
    { source: "t1", target: "t2", weight: 0.35 },
  ];
}

// Generate action recommendations
function generateActions(channel: Channel): ActionItem[] {
  const baseActions: Record<Channel, ActionItem[]> = {
    instagram: [
      {
        id: "act1",
        title: "릴스 콘텐츠 제작",
        description: "지난 주 릴스 도달이 42% 증가했습니다. 신제품 릴스를 추가 제작하세요.",
        priority: "high",
        type: "content",
      },
      {
        id: "act2",
        title: "댓글 응답률 개선",
        description: "미응답 댓글 23개가 있습니다. 24시간 내 응답을 권장합니다.",
        priority: "medium",
        type: "engagement",
      },
      {
        id: "act3",
        title: "해시태그 전략 수정",
        description: "#브랜드명 해시태그 노출이 저조합니다. 대안 해시태그를 검토하세요.",
        priority: "low",
        type: "analysis",
      },
    ],
    tiktok: [
      {
        id: "act1",
        title: "트렌드 챌린지 참여",
        description: "현재 인기 챌린지 '#신상템챌린지'에 참여하면 노출 증가가 예상됩니다.",
        priority: "high",
        type: "content",
      },
      {
        id: "act2",
        title: "듀엣 콘텐츠 기획",
        description: "인플루언서 콜라보 듀엣 영상으로 팔로워 확대를 노려보세요.",
        priority: "medium",
        type: "campaign",
      },
      {
        id: "act3",
        title: "업로드 시간 최적화",
        description: "분석 결과 오후 8-10시 업로드가 최적입니다.",
        priority: "low",
        type: "analysis",
      },
    ],
    twitter: [
      {
        id: "act1",
        title: "실시간 트렌드 대응",
        description: "브랜드 관련 키워드가 트렌딩 중입니다. 빠른 대응이 필요합니다.",
        priority: "high",
        type: "engagement",
      },
      {
        id: "act2",
        title: "스레드 콘텐츠 제작",
        description: "제품 스토리를 스레드 형식으로 풀어보세요. 저장률이 높습니다.",
        priority: "medium",
        type: "content",
      },
      {
        id: "act3",
        title: "멘션 모니터링 강화",
        description: "부정 멘션 3건이 감지되었습니다. 검토가 필요합니다.",
        priority: "high",
        type: "analysis",
      },
    ],
    youtube: [
      {
        id: "act1",
        title: "숏츠 시리즈 기획",
        description: "숏츠 조회수가 일반 영상 대비 3배 높습니다. 시리즈를 기획하세요.",
        priority: "high",
        type: "content",
      },
      {
        id: "act2",
        title: "커뮤니티 탭 활용",
        description: "커뮤니티 탭 활용으로 구독자 참여도를 높이세요.",
        priority: "medium",
        type: "engagement",
      },
      {
        id: "act3",
        title: "썸네일 A/B 테스트",
        description: "클릭률 개선을 위해 썸네일 테스트를 진행하세요.",
        priority: "low",
        type: "analysis",
      },
    ],
    web: [
      {
        id: "act1",
        title: "랜딩페이지 최적화",
        description: "이탈률이 65%로 높습니다. CTA 버튼 위치 조정을 권장합니다.",
        priority: "high",
        type: "analysis",
      },
      {
        id: "act2",
        title: "블로그 SEO 개선",
        description: "주요 키워드 검색 순위가 하락했습니다. 콘텐츠 업데이트가 필요합니다.",
        priority: "medium",
        type: "content",
      },
      {
        id: "act3",
        title: "뉴스레터 발송",
        description: "지난 뉴스레터 오픈율 28%. 이번 주 발송을 준비하세요.",
        priority: "low",
        type: "campaign",
      },
    ],
  };

  return baseActions[channel] || baseActions.instagram;
}

// Generate AI answer based on mode and channel
function generateAnswer(mode: OntixMode, channel: Channel): OntixAnswer {
  const answers: Record<OntixMode, Record<Channel, string>> = {
    advisor: {
      instagram: `이번 주 인스타그램 전략 핵심은 **릴스 중심 콘텐츠 확대**입니다.

지난 7일간 데이터를 분석한 결과, 릴스 콘텐츠의 도달률이 피드 대비 2.3배 높았고, 특히 15-30초 길이의 영상이 가장 높은 완료율(78%)을 기록했습니다.

**권장 액션:**
1. 신제품 언박싱 릴스 2개 추가 제작
2. 베스트셀러 사용법 숏폼 시리즈 기획
3. UGC 리그램 캠페인 진행으로 참여 유도

현재 팔로워 대비 참여율은 4.2%로 업계 평균(3.1%)을 상회하고 있으나, 저장률 개선의 여지가 있습니다. 정보성 콘텐츠를 늘려 저장 유도를 권장합니다.`,
      tiktok: `틱톡 채널의 이번 주 핵심 전략은 **트렌드 기반 빠른 대응**입니다.

현재 브랜드 관련 해시태그 '#뷰티템추천'이 급상승 중이며, 이 트렌드에 48시간 내 참여하면 예상 노출 30만 회 이상을 기대할 수 있습니다.

**권장 액션:**
1. 트렌딩 사운드 활용 제품 소개 영상
2. 듀엣/스티치 유도형 콘텐츠 제작
3. 라이브 커머스 테스트 진행`,
      twitter: `X(트위터) 채널에서는 **실시간 대화 참여**가 핵심입니다.

브랜드 멘션이 지난 주 대비 45% 증가했으며, 긍정 멘션 비율은 82%입니다. 다만 응답 시간이 평균 4시간으로, 1시간 이내 응답을 목표로 개선이 필요합니다.`,
      youtube: `유튜브 채널의 이번 주 전략은 **숏츠 + 커뮤니티 시너지**입니다.

숏츠 업로드 후 커뮤니티 탭에서 투표/질문을 연계하면 구독자 참여도가 2배 이상 증가하는 패턴이 확인되었습니다.`,
      web: `웹사이트의 이번 주 핵심은 **전환율 최적화**입니다.

현재 랜딩페이지 이탈률 65%, 장바구니 이탈률 78%로 개선이 시급합니다. A/B 테스트를 통해 CTA 버튼 색상과 위치 최적화를 권장합니다.`,
    },
    analytics: {
      instagram: `**인스타그램 주간 분석 리포트**

• 총 도달: 127,340 (+23% vs 전주)
• 참여율: 4.2% (업계 평균 3.1%)
• 팔로워 증가: +892명
• 최고 성과 콘텐츠: 신제품 릴스 (도달 45,200)

**인사이트:**
릴스 콘텐츠가 전체 도달의 58%를 차지하며, 특히 화요일/목요일 오후 7-9시 업로드 콘텐츠의 성과가 가장 높았습니다.`,
      tiktok: `**틱톡 주간 분석 리포트**

• 총 조회수: 892,000 (+67% vs 전주)
• 평균 시청 시간: 12.3초
• 팔로워 증가: +2,340명
• 바이럴 콘텐츠: 2개 (10만 뷰 이상)`,
      twitter: `**X(트위터) 주간 분석 리포트**

• 노출수: 234,500
• 참여율: 2.8%
• 멘션: 156건 (긍정 82%)
• 리포스트: 89건`,
      youtube: `**유튜브 주간 분석 리포트**

• 총 조회수: 45,200
• 평균 시청 시간: 4분 32초
• 구독자 증가: +234명
• 숏츠 조회수: 128,000`,
      web: `**웹사이트 주간 분석 리포트**

• 방문자: 12,450
• 평균 세션 시간: 2분 45초
• 이탈률: 65%
• 전환율: 2.3%`,
    },
    product: {
      instagram: `**인스타그램 추천 상품 전략**

분석 결과, 현재 인스타그램에서 가장 주목받는 상품은 **신제품 A**입니다.

관련 UGC 42건이 생성되었으며, 평균 좋아요 수는 브랜드 평균 대비 1.8배입니다. 이 상품을 중심으로 인플루언서 협업을 확대하는 것을 권장합니다.`,
      tiktok: `틱톡에서는 **가성비 라인** 제품이 가장 높은 반응을 얻고 있습니다.`,
      twitter: `X에서는 **한정판/콜라보 제품**에 대한 관심이 높습니다.`,
      youtube: `유튜브에서는 **사용법 튜토리얼**이 포함된 제품이 높은 전환율을 보입니다.`,
      web: `웹사이트에서는 **베스트셀러 번들**이 가장 높은 전환율을 기록 중입니다.`,
    },
    content: {
      instagram: `**인스타그램 콘텐츠 추천**

이번 주 제작 권장 콘텐츠:

1. **릴스**: "하루 스킨케어 루틴" (예상 도달 50,000+)
2. **캐러셀**: "제품 비교 가이드" (저장률 높음)
3. **스토리**: "Q&A 세션" (참여 유도)

최적 업로드 시간: 화/목 오후 8시`,
      tiktok: `**틱톡 콘텐츠 추천**

트렌딩 포맷 활용:
1. "Get Ready With Me" 스타일
2. 비포/애프터 변신 영상
3. 댓글 답변 영상`,
      twitter: `**X 콘텐츠 추천**

참여 유도 포맷:
1. 투표형 질문
2. 제품 스레드
3. 실시간 이벤트 대응`,
      youtube: `**유튜브 콘텐츠 추천**

제작 우선순위:
1. 숏츠: 제품 30초 리뷰
2. 롱폼: 상세 사용법 가이드
3. 커뮤니티: 구독자 투표`,
      web: `**웹 콘텐츠 추천**

블로그 포스트 주제:
1. "2024 뷰티 트렌드 가이드"
2. "제품 비교: A vs B"
3. "사용자 후기 모음"`,
    },
    onboarding: {
      instagram: `**인스타그램 온보딩 가이드**

ONTIX와 함께 인스타그램 채널을 최적화하는 방법을 안내드립니다.

**Step 1**: 비즈니스 계정 연동
**Step 2**: 30일 데이터 분석 완료 대기
**Step 3**: Knowledge Graph 생성 확인
**Step 4**: 첫 번째 인사이트 리포트 확인`,
      tiktok: `틱톡 채널 온보딩을 시작합니다. 계정 연동 후 14일간의 학습 기간이 필요합니다.`,
      twitter: `X 채널 온보딩을 시작합니다. API 연동 승인 후 분석이 시작됩니다.`,
      youtube: `유튜브 채널 온보딩을 시작합니다. YouTube Analytics 권한 부여가 필요합니다.`,
      web: `웹사이트 온보딩을 시작합니다. 트래킹 스크립트 설치가 필요합니다.`,
    },
    monitoring: {
      instagram: `**인스타그램 실시간 모니터링**

🟢 **현재 상태: 정상**

• 브랜드 멘션: 23건 (최근 24시간)
• 부정 멘션: 0건
• 위기 감지: 없음

**주목할 트렌드:**
"#신상품리뷰" 해시태그가 급상승 중 (+340%)`,
      tiktok: `**틱톡 실시간 모니터링**

🟢 현재 상태: 정상
바이럴 가능성 콘텐츠 1건 감지됨`,
      twitter: `**X 실시간 모니터링**

🟡 현재 상태: 주의
브랜드 관련 논쟁성 트윗 2건 감지`,
      youtube: `**유튜브 실시간 모니터링**

🟢 현재 상태: 정상
부정 댓글 0건`,
      web: `**웹사이트 실시간 모니터링**

🟢 현재 상태: 정상
서버 응답 시간: 평균 120ms`,
    },
  };

  const text = answers[mode]?.[channel] || "데이터를 분석 중입니다...";

  // Simulate grade based on mode/channel combination
  const gradeMap: Record<OntixMode, Grade> = {
    advisor: "A",
    analytics: "A",
    product: "B",
    content: "B",
    onboarding: "A",
    monitoring: "A",
  };

  return {
    mode,
    channel,
    grade: gradeMap[mode],
    text,
    filterStatus: gradeMap[mode] === "A" || gradeMap[mode] === "B" ? "passed" : "warning",
  };
}

// Main function to get channel overview
export function getChannelOverview(channel: Channel): ChannelOverview {
  return {
    channel,
    engagementTimeline: generateEngagementTimeline(7),
    activityNoiseSeed: Math.random() * 1000,
    graphNodes: generateGraphNodes(),
    graphEdges: generateGraphEdges(),
    topActions: generateActions(channel),
  };
}

// Get AI answer
export function getOntixAnswer(mode: OntixMode, channel: Channel): OntixAnswer {
  return generateAnswer(mode, channel);
}

// Get all channel overviews
export function getAllChannelOverviews(): Record<Channel, ChannelOverview> {
  return {
    instagram: getChannelOverview("instagram"),
    tiktok: getChannelOverview("tiktok"),
    twitter: getChannelOverview("twitter"),
    youtube: getChannelOverview("youtube"),
    web: getChannelOverview("web"),
  };
}
