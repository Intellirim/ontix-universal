"use client";

import { useState, useEffect } from "react";
import {
  Play,
  RefreshCw,
  Clock,
  CheckCircle2,
  XCircle,
  AlertCircle,
  Loader2,
  Instagram,
  Youtube,
  Music2,
  Twitter,
  Globe,
  Hash,
  User,
  Search,
  Link2,
  ChevronRight,
  Plus,
  X,
  Building2,
  Sparkles,
  ChevronDown,
} from "lucide-react";
import { usePipelineJobs, useRunPipeline, useBrands, useCreateBrand, usePipelineLogs } from "@/hooks/use-api";
import { useBrandContext } from "../layout";
import { cn } from "@/lib/utils";
import type { PipelinePlatform, PipelineTargetType } from "@/types";

const platforms: { id: PipelinePlatform; name: string; icon: typeof Instagram }[] = [
  { id: "instagram", name: "Instagram", icon: Instagram },
  { id: "youtube", name: "YouTube", icon: Youtube },
  { id: "tiktok", name: "TikTok", icon: Music2 },
  { id: "twitter", name: "Twitter/X", icon: Twitter },
  { id: "website", name: "Website", icon: Globe },
];

const targetTypes: Record<PipelinePlatform, { id: PipelineTargetType; label: string; placeholder: string; icon: typeof User }[]> = {
  instagram: [
    { id: "accounts", label: "계정", placeholder: "futurebiofficial", icon: User },
    { id: "hashtags", label: "해시태그", placeholder: "역노화", icon: Hash },
    { id: "search", label: "검색어", placeholder: "역노화 피부관리", icon: Search },
  ],
  youtube: [
    { id: "accounts", label: "채널", placeholder: "@futurebi", icon: User },
    { id: "search", label: "검색어", placeholder: "역노화 방법", icon: Search },
  ],
  tiktok: [
    { id: "accounts", label: "프로필", placeholder: "futurebi_official", icon: User },
    { id: "hashtags", label: "해시태그", placeholder: "antiaging", icon: Hash },
    { id: "search", label: "검색어", placeholder: "노화 방지 팁", icon: Search },
  ],
  twitter: [
    { id: "accounts", label: "핸들", placeholder: "futurebi_kr", icon: User },
    { id: "hashtags", label: "해시태그", placeholder: "역노화", icon: Hash },
    { id: "search", label: "검색어", placeholder: "longevity research", icon: Search },
  ],
  website: [
    { id: "urls", label: "URL", placeholder: "https://futurebi.io", icon: Link2 },
  ],
};

type TabType = "expand" | "new";

export default function PipelinePage() {
  const { selectedBrandId, setSelectedBrandId } = useBrandContext();
  const { data: brands, isLoading: brandsLoading, refetch: refetchBrands } = useBrands();
  const { data: jobsData, isLoading: jobsLoading, refetch } = usePipelineJobs(selectedBrandId);
  const runPipeline = useRunPipeline();
  const createBrand = useCreateBrand();

  const [activeTab, setActiveTab] = useState<TabType>("expand");
  const [localSelectedBrandId, setLocalSelectedBrandId] = useState<string>("");
  const [showBrandDropdown, setShowBrandDropdown] = useState(false);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);

  // Get the active job (running or pending) or selected job
  const activeJob = jobsData?.jobs?.find(j => j.status === "running" || j.status === "pending");
  const jobToWatch = selectedJobId || activeJob?.job_id || null;
  const isJobActive = jobToWatch ? ["running", "pending"].includes(jobsData?.jobs?.find(j => j.job_id === jobToWatch)?.status || "") : false;

  // Fetch logs for the watched job
  const { data: logsData } = usePipelineLogs(jobToWatch || "", isJobActive || !!selectedJobId);

  // New brand form state
  const [newBrandName, setNewBrandName] = useState("");
  const [newBrandIndustry, setNewBrandIndustry] = useState("");
  const [newBrandDescription, setNewBrandDescription] = useState("");

  const [selectedPlatform, setSelectedPlatform] = useState<PipelinePlatform>("instagram");
  const [targetType, setTargetType] = useState<PipelineTargetType>("accounts");
  const [targetInputs, setTargetInputs] = useState<string[]>([""]);
  const [maxItems, setMaxItems] = useState(50);
  const [dryRun, setDryRun] = useState(false);
  const [skipLlm, setSkipLlm] = useState(false);

  // Notification state
  const [notification, setNotification] = useState<{
    type: "success" | "error" | "loading";
    message: string;
  } | null>(null);

  // Auto-clear notification after 5 seconds
  useEffect(() => {
    if (notification && notification.type !== "loading") {
      const timer = setTimeout(() => setNotification(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [notification]);

  // Sync localSelectedBrandId with global context
  useEffect(() => {
    if (selectedBrandId && activeTab === "expand") {
      setLocalSelectedBrandId(selectedBrandId);
    }
  }, [selectedBrandId, activeTab]);

  // Platform 변경 시 target type 초기화
  useEffect(() => {
    const firstType = targetTypes[selectedPlatform][0];
    setTargetType(firstType.id);
    setTargetInputs([""]);
  }, [selectedPlatform]);

  // Target type 변경 시 입력 초기화
  useEffect(() => {
    setTargetInputs([""]);
  }, [targetType]);

  const addTargetInput = () => {
    setTargetInputs([...targetInputs, ""]);
  };

  const removeTargetInput = (index: number) => {
    if (targetInputs.length === 1) {
      setTargetInputs([""]);
    } else {
      setTargetInputs(targetInputs.filter((_, i) => i !== index));
    }
  };

  const updateTargetInput = (index: number, value: string) => {
    const updated = [...targetInputs];
    updated[index] = value;
    setTargetInputs(updated);
  };

  const handleRunForExistingBrand = async () => {
    const targets = targetInputs
      .map((t) => t.trim().replace(/^[@#]/, ""))
      .filter(Boolean);

    if (!localSelectedBrandId || targets.length === 0) return;

    setNotification({ type: "loading", message: "데이터 수집을 시작하는 중..." });

    try {
      const result = await runPipeline.mutateAsync({
        brand_id: localSelectedBrandId,
        platform: selectedPlatform,
        target_type: targetType,
        targets,
        max_items: maxItems,
        dry_run: dryRun,
        skip_crawl: false,
        skip_llm: skipLlm,
      });
      setTargetInputs([""]);
      setSelectedJobId(result.job_id); // Auto-select to show logs
      refetch();
      setNotification({
        type: "success",
        message: `데이터 수집이 시작되었습니다! (작업 ID: ${result.job_id.slice(0, 8)}...)`
      });
    } catch (error) {
      console.error("Failed to run pipeline:", error);
      setNotification({
        type: "error",
        message: error instanceof Error ? error.message : "데이터 수집 시작에 실패했습니다."
      });
    }
  };

  const handleCreateNewBrand = async () => {
    const targets = targetInputs
      .map((t) => t.trim().replace(/^[@#]/, ""))
      .filter(Boolean);

    if (!newBrandName.trim() || targets.length === 0) return;

    setNotification({ type: "loading", message: "브랜드 생성 및 데이터 수집을 시작하는 중..." });

    try {
      // Generate brand ID from name (lowercase letters, numbers, and hyphens only)
      const brandId = newBrandName
        .trim()
        .toLowerCase()
        .replace(/\s+/g, "-")
        .replace(/[^a-z0-9-]/g, "")
        .replace(/-+/g, "-")
        .replace(/^-|-$/g, "")
        .substring(0, 50) || `brand-${Date.now()}`;

      // 1. Create new brand
      const newBrand = await createBrand.mutateAsync({
        id: brandId,
        name: newBrandName.trim(),
        industry: newBrandIndustry.trim() || undefined,
        description: newBrandDescription.trim() || undefined,
      });

      // 2. Run pipeline for the new brand
      const pipelineResult = await runPipeline.mutateAsync({
        brand_id: newBrand.id,
        platform: selectedPlatform,
        target_type: targetType,
        targets,
        max_items: maxItems,
        dry_run: dryRun,
        skip_crawl: false,
        skip_llm: skipLlm,
      });
      setSelectedJobId(pipelineResult.job_id); // Auto-select to show logs

      // 3. Reset form and switch to expand tab
      setNewBrandName("");
      setNewBrandIndustry("");
      setNewBrandDescription("");
      setTargetInputs([""]);
      refetchBrands();
      setSelectedBrandId(newBrand.id);
      setActiveTab("expand");
      refetch();
      setNotification({
        type: "success",
        message: `브랜드 "${newBrand.name}"가 생성되고 데이터 수집이 시작되었습니다!`
      });
    } catch (error) {
      console.error("Failed to create brand and run pipeline:", error);
      setNotification({
        type: "error",
        message: error instanceof Error ? error.message : "브랜드 생성 또는 수집에 실패했습니다."
      });
    }
  };

  const getStatusConfig = (status: string) => {
    switch (status) {
      case "completed":
        return { icon: CheckCircle2, label: "완료", color: "text-emerald-400", bg: "bg-emerald-500/10 border-emerald-500/20" };
      case "failed":
        return { icon: XCircle, label: "실패", color: "text-red-400", bg: "bg-red-500/10 border-red-500/20" };
      case "running":
        return { icon: Loader2, label: "실행 중", color: "text-blue-400", bg: "bg-blue-500/10 border-blue-500/20", spin: true };
      case "pending":
        return { icon: Clock, label: "대기 중", color: "text-amber-400", bg: "bg-amber-500/10 border-amber-500/20" };
      default:
        return { icon: AlertCircle, label: status, color: "text-muted-foreground", bg: "bg-muted" };
    }
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleString("ko-KR", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const currentTargetConfig = targetTypes[selectedPlatform].find(t => t.id === targetType);
  const selectedBrand = brands?.find(b => b.id === localSelectedBrandId);

  const renderPlatformAndTargetForm = () => (
    <>
      {/* Platform Selection */}
      <div className="mb-5">
        <label className="block text-sm font-medium text-muted-foreground mb-2.5">
          플랫폼
        </label>
        <div className="grid grid-cols-3 sm:grid-cols-5 gap-2">
          {platforms.map((platform) => {
            const Icon = platform.icon;
            const isSelected = selectedPlatform === platform.id;
            return (
              <button
                key={platform.id}
                onClick={() => setSelectedPlatform(platform.id)}
                className={cn(
                  "flex flex-col items-center gap-1.5 sm:gap-2 p-2 sm:p-3 rounded-lg transition-all border",
                  isSelected
                    ? "bg-primary/10 border-primary/50 text-primary"
                    : "bg-muted/50 border-border hover:border-border/80 text-muted-foreground hover:text-foreground"
                )}
              >
                <Icon className="w-4 h-4 sm:w-5 sm:h-5" />
                <span className="text-[10px] sm:text-xs font-medium">{platform.name}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Target Type */}
      <div className="mb-5">
        <label className="block text-sm font-medium text-muted-foreground mb-2.5">
          수집 유형
        </label>
        <div className="flex gap-2 flex-wrap">
          {targetTypes[selectedPlatform].map((type) => {
            const Icon = type.icon;
            const isSelected = targetType === type.id;
            return (
              <button
                key={type.id}
                onClick={() => setTargetType(type.id)}
                className={cn(
                  "inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all border",
                  isSelected
                    ? "bg-primary/10 text-primary border-primary/50"
                    : "bg-muted/50 text-muted-foreground border-border hover:text-foreground hover:border-border/80"
                )}
              >
                <Icon className="w-4 h-4" />
                {type.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Targets Input */}
      <div className="mb-5">
        <div className="flex items-center justify-between mb-2.5">
          <label className="text-sm font-medium text-muted-foreground">
            대상 ({targetInputs.filter(t => t.trim()).length}개)
          </label>
          <button
            type="button"
            onClick={addTargetInput}
            className="inline-flex items-center gap-1 px-2 py-1 text-xs font-medium text-primary hover:text-primary/80 transition-colors"
          >
            <Plus className="w-3.5 h-3.5" />
            추가
          </button>
        </div>

        <div className="space-y-2 max-h-[200px] overflow-y-auto pr-1">
          {targetInputs.map((value, index) => (
            <div key={index} className="flex items-center gap-2">
              <div className="flex-1 relative">
                <div className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground/50 text-xs font-mono">
                  {index + 1}.
                </div>
                <input
                  type="text"
                  value={value}
                  onChange={(e) => updateTargetInput(index, e.target.value)}
                  placeholder={currentTargetConfig?.placeholder}
                  className="w-full pl-9 pr-3 py-2.5 rounded-lg bg-muted/50 border border-border text-foreground text-sm placeholder:text-muted-foreground/50 focus:border-primary/50 focus:outline-none focus:ring-1 focus:ring-primary/20 transition-all font-mono"
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      addTargetInput();
                    }
                  }}
                />
              </div>
              <button
                type="button"
                onClick={() => removeTargetInput(index)}
                className="p-2 rounded-lg text-muted-foreground/50 hover:text-red-400 hover:bg-red-500/10 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>

        <p className="text-xs text-muted-foreground mt-2">
          {targetType === "accounts" && "@ 기호 없이 사용자명만 입력해도 됩니다"}
          {targetType === "hashtags" && "# 기호 없이 태그만 입력해도 됩니다"}
          {targetType === "urls" && "https://로 시작하는 전체 URL을 입력하세요"}
          {targetType === "search" && "검색하고 싶은 키워드를 입력하세요"}
        </p>
      </div>

      {/* Max Items */}
      <div className="mb-5">
        <label className="block text-sm font-medium text-muted-foreground mb-2.5">
          최대 수집 항목
        </label>
        <div className="flex items-center gap-3">
          <input
            type="range"
            value={maxItems}
            onChange={(e) => setMaxItems(Number(e.target.value))}
            min={10}
            max={500}
            step={10}
            className="flex-1 h-2 bg-muted rounded-full appearance-none cursor-pointer accent-primary"
          />
          <div className="w-16 px-3 py-2 rounded-lg bg-muted/50 border border-border text-center">
            <span className="text-sm font-medium text-foreground">{maxItems}</span>
          </div>
        </div>
      </div>

      {/* Options */}
      <div className="mb-6 p-4 rounded-lg bg-muted/30 border border-border space-y-3">
        <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">
          옵션
        </div>

        <label className="flex items-center gap-3 cursor-pointer group">
          <div className={cn(
            "w-5 h-5 rounded border-2 flex items-center justify-center transition-all",
            dryRun
              ? "bg-primary border-primary"
              : "border-muted-foreground/30 group-hover:border-muted-foreground/50"
          )}>
            {dryRun && (
              <svg className="w-3 h-3 text-primary-foreground" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
              </svg>
            )}
          </div>
          <input
            type="checkbox"
            checked={dryRun}
            onChange={(e) => setDryRun(e.target.checked)}
            className="sr-only"
          />
          <div className="flex-1">
            <span className="text-sm text-foreground">테스트 모드</span>
            <p className="text-xs text-muted-foreground">저장하지 않고 테스트만 실행</p>
          </div>
        </label>

        <label className="flex items-center gap-3 cursor-pointer group">
          <div className={cn(
            "w-5 h-5 rounded border-2 flex items-center justify-center transition-all",
            skipLlm
              ? "bg-amber-500 border-amber-500"
              : "border-muted-foreground/30 group-hover:border-muted-foreground/50"
          )}>
            {skipLlm && (
              <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
              </svg>
            )}
          </div>
          <input
            type="checkbox"
            checked={skipLlm}
            onChange={(e) => setSkipLlm(e.target.checked)}
            className="sr-only"
          />
          <div className="flex-1">
            <span className="text-sm text-foreground">LLM 분석 건너뛰기</span>
            <p className="text-xs text-muted-foreground">AI 분석 없이 원본 저장</p>
          </div>
        </label>
      </div>
    </>
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-foreground">데이터 수집</h1>
          <p className="text-muted-foreground text-xs sm:text-sm mt-1">
            SNS 및 웹사이트에서 브랜드 관련 데이터를 수집합니다
          </p>
        </div>
        <button
          onClick={() => refetch()}
          disabled={jobsLoading}
          className="inline-flex items-center justify-center gap-2 px-4 py-2.5 bg-card border border-border text-muted-foreground text-sm font-medium rounded-lg hover:text-foreground hover:border-border/80 transition-colors w-full sm:w-auto"
        >
          <RefreshCw className={cn("w-4 h-4", jobsLoading && "animate-spin")} />
          새로고침
        </button>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">
        {/* Form Section */}
        <div className="xl:col-span-2">
          <div className="p-4 sm:p-6 rounded-xl bg-card border border-border">
            {/* Notification */}
            {notification && (
              <div className={cn(
                "mb-4 p-4 rounded-lg border flex items-center gap-3",
                notification.type === "success" && "bg-emerald-500/10 border-emerald-500/30 text-emerald-400",
                notification.type === "error" && "bg-red-500/10 border-red-500/30 text-red-400",
                notification.type === "loading" && "bg-blue-500/10 border-blue-500/30 text-blue-400"
              )}>
                {notification.type === "loading" && <Loader2 className="w-4 h-4 animate-spin" />}
                {notification.type === "success" && <CheckCircle2 className="w-4 h-4" />}
                {notification.type === "error" && <XCircle className="w-4 h-4" />}
                <span className="text-sm flex-1">{notification.message}</span>
                {notification.type !== "loading" && (
                  <button onClick={() => setNotification(null)} className="text-current/50 hover:text-current">
                    <X className="w-4 h-4" />
                  </button>
                )}
              </div>
            )}

            {/* Tabs */}
            <div className="flex gap-1 p-1 rounded-lg bg-muted/50 mb-6">
              <button
                onClick={() => setActiveTab("expand")}
                className={cn(
                  "flex-1 flex items-center justify-center gap-1.5 sm:gap-2 px-2 sm:px-4 py-2 sm:py-2.5 rounded-md text-xs sm:text-sm font-medium transition-all",
                  activeTab === "expand"
                    ? "bg-card text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                <Building2 className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                <span className="hidden xs:inline">기존 브랜드</span>
                <span className="xs:hidden">기존</span>
                <span className="hidden sm:inline"> 확장</span>
              </button>
              <button
                onClick={() => setActiveTab("new")}
                className={cn(
                  "flex-1 flex items-center justify-center gap-1.5 sm:gap-2 px-2 sm:px-4 py-2 sm:py-2.5 rounded-md text-xs sm:text-sm font-medium transition-all",
                  activeTab === "new"
                    ? "bg-card text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                <Sparkles className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                <span className="hidden xs:inline">새 브랜드</span>
                <span className="xs:hidden">신규</span>
                <span className="hidden sm:inline"> 생성</span>
              </button>
            </div>

            {activeTab === "expand" ? (
              <>
                {/* Brand Selection Dropdown */}
                <div className="mb-5">
                  <label className="block text-sm font-medium text-muted-foreground mb-2.5">
                    브랜드 선택
                  </label>
                  <div className="relative">
                    <button
                      onClick={() => setShowBrandDropdown(!showBrandDropdown)}
                      className="w-full flex items-center justify-between px-4 py-3 rounded-lg bg-muted/50 border border-border text-left hover:border-primary/50 transition-all"
                    >
                      {brandsLoading ? (
                        <span className="text-muted-foreground text-sm">로딩 중...</span>
                      ) : selectedBrand ? (
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center text-xs font-bold text-primary-foreground">
                            {selectedBrand.name.charAt(0).toUpperCase()}
                          </div>
                          <div>
                            <div className="text-sm font-medium text-foreground">{selectedBrand.name}</div>
                            {selectedBrand.industry && (
                              <div className="text-xs text-muted-foreground">{selectedBrand.industry}</div>
                            )}
                          </div>
                        </div>
                      ) : (
                        <span className="text-muted-foreground text-sm">브랜드를 선택하세요</span>
                      )}
                      <ChevronDown className={cn(
                        "w-4 h-4 text-muted-foreground transition-transform",
                        showBrandDropdown && "rotate-180"
                      )} />
                    </button>

                    {showBrandDropdown && (
                      <>
                        <div className="fixed inset-0 z-10" onClick={() => setShowBrandDropdown(false)} />
                        <div className="absolute top-full left-0 right-0 mt-2 bg-card/95 backdrop-blur-xl rounded-xl border border-border shadow-2xl z-20 py-1.5 max-h-64 overflow-y-auto">
                          {brands?.map((brand) => (
                            <button
                              key={brand.id}
                              onClick={() => {
                                setLocalSelectedBrandId(brand.id);
                                setShowBrandDropdown(false);
                              }}
                              className={cn(
                                "w-full px-4 py-3 text-left flex items-center gap-3 transition-all",
                                brand.id === localSelectedBrandId
                                  ? "bg-primary/10 text-foreground"
                                  : "hover:bg-muted/50 text-muted-foreground hover:text-foreground"
                              )}
                            >
                              <div className={cn(
                                "w-8 h-8 rounded-lg flex items-center justify-center text-xs font-bold",
                                brand.id === localSelectedBrandId
                                  ? "bg-primary text-primary-foreground"
                                  : "bg-muted text-muted-foreground"
                              )}>
                                {brand.name.charAt(0).toUpperCase()}
                              </div>
                              <div className="flex-1 min-w-0">
                                <div className="font-medium text-sm truncate">{brand.name}</div>
                                {brand.industry && (
                                  <div className="text-xs text-muted-foreground truncate">{brand.industry}</div>
                                )}
                              </div>
                              {brand.id === localSelectedBrandId && (
                                <div className="w-2 h-2 rounded-full bg-primary" />
                              )}
                            </button>
                          ))}
                        </div>
                      </>
                    )}
                  </div>
                </div>

                {localSelectedBrandId ? (
                  <>
                    {renderPlatformAndTargetForm()}

                    {/* Run Button */}
                    <button
                      onClick={handleRunForExistingBrand}
                      disabled={!targetInputs.some(t => t.trim()) || runPipeline.isPending}
                      className="w-full inline-flex items-center justify-center gap-2 px-6 py-3 bg-primary text-primary-foreground text-sm font-medium rounded-lg hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {runPipeline.isPending ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <Play className="w-4 h-4" />
                      )}
                      데이터 수집 시작
                    </button>
                  </>
                ) : (
                  <div className="py-8 text-center">
                    <div className="w-12 h-12 rounded-xl bg-muted flex items-center justify-center mx-auto mb-3">
                      <Building2 className="w-6 h-6 text-muted-foreground" />
                    </div>
                    <p className="text-sm text-muted-foreground">
                      브랜드를 선택하면 데이터 수집을 시작할 수 있습니다
                    </p>
                  </div>
                )}
              </>
            ) : (
              <>
                {/* New Brand Form */}
                <div className="mb-5">
                  <label className="block text-sm font-medium text-muted-foreground mb-2.5">
                    브랜드 이름 <span className="text-red-400">*</span>
                  </label>
                  <input
                    type="text"
                    value={newBrandName}
                    onChange={(e) => setNewBrandName(e.target.value)}
                    placeholder="예: 퓨처바이오"
                    className="w-full px-4 py-3 rounded-lg bg-muted/50 border border-border text-foreground text-sm placeholder:text-muted-foreground/50 focus:border-primary/50 focus:outline-none focus:ring-1 focus:ring-primary/20 transition-all"
                  />
                </div>

                <div className="mb-5">
                  <label className="block text-sm font-medium text-muted-foreground mb-2.5">
                    산업 분야
                  </label>
                  <input
                    type="text"
                    value={newBrandIndustry}
                    onChange={(e) => setNewBrandIndustry(e.target.value)}
                    placeholder="예: 뷰티/화장품"
                    className="w-full px-4 py-3 rounded-lg bg-muted/50 border border-border text-foreground text-sm placeholder:text-muted-foreground/50 focus:border-primary/50 focus:outline-none focus:ring-1 focus:ring-primary/20 transition-all"
                  />
                </div>

                <div className="mb-5">
                  <label className="block text-sm font-medium text-muted-foreground mb-2.5">
                    브랜드 설명
                  </label>
                  <textarea
                    value={newBrandDescription}
                    onChange={(e) => setNewBrandDescription(e.target.value)}
                    placeholder="브랜드에 대한 간단한 설명을 입력하세요"
                    rows={3}
                    className="w-full px-4 py-3 rounded-lg bg-muted/50 border border-border text-foreground text-sm placeholder:text-muted-foreground/50 focus:border-primary/50 focus:outline-none focus:ring-1 focus:ring-primary/20 transition-all resize-none"
                  />
                </div>

                <div className="h-px bg-border my-6" />

                {renderPlatformAndTargetForm()}

                {/* Create & Run Button */}
                <button
                  onClick={handleCreateNewBrand}
                  disabled={!newBrandName.trim() || !targetInputs.some(t => t.trim()) || createBrand.isPending || runPipeline.isPending}
                  className="w-full inline-flex items-center justify-center gap-2 px-6 py-3 bg-gradient-to-r from-primary to-primary/80 text-primary-foreground text-sm font-medium rounded-lg hover:from-primary/90 hover:to-primary/70 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {(createBrand.isPending || runPipeline.isPending) ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Sparkles className="w-4 h-4" />
                  )}
                  브랜드 생성 및 수집 시작
                </button>
              </>
            )}
          </div>
        </div>

        {/* Jobs List */}
        <div className="xl:col-span-3">
          <div className="p-4 sm:p-6 rounded-xl bg-card border border-border">
            <div className="flex items-center justify-between mb-5">
              <h3 className="text-base font-semibold text-foreground">수집 작업 목록</h3>
              {jobsData?.jobs && jobsData.jobs.length > 0 && (
                <span className="text-xs text-muted-foreground bg-muted px-2 py-1 rounded">
                  {jobsData.jobs.length}개 작업
                </span>
              )}
            </div>

            {jobsLoading ? (
              <div className="text-center py-12">
                <Loader2 className="w-8 h-8 text-primary animate-spin mx-auto mb-3" />
                <p className="text-sm text-muted-foreground">로딩 중...</p>
              </div>
            ) : jobsData?.jobs && jobsData.jobs.length > 0 ? (
              <div className="space-y-2">
                {jobsData.jobs.map((job) => {
                  const statusConfig = getStatusConfig(job.status);
                  const StatusIcon = statusConfig.icon;
                  const JobPlatformIcon = platforms.find(p => p.id === job.platform)?.icon || Globe;
                  const isSelected = selectedJobId === job.job_id;

                  return (
                    <div
                      key={job.job_id}
                      onClick={() => setSelectedJobId(isSelected ? null : job.job_id)}
                      className={cn(
                        "flex items-center justify-between p-4 rounded-lg border transition-colors group cursor-pointer",
                        isSelected
                          ? "bg-primary/5 border-primary/30"
                          : "bg-muted/30 border-border hover:border-border/80"
                      )}
                    >
                      <div className="flex items-center gap-2 sm:gap-4 min-w-0 flex-1">
                        <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-lg bg-muted flex items-center justify-center shrink-0">
                          <JobPlatformIcon className="w-4 h-4 sm:w-5 sm:h-5 text-muted-foreground" />
                        </div>
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-2">
                            <span className="text-xs sm:text-sm font-medium text-foreground capitalize">
                              {job.platform}
                            </span>
                            <span className="text-[10px] sm:text-xs text-muted-foreground bg-muted px-1.5 py-0.5 rounded hidden xs:inline">
                              {job.target_type}
                            </span>
                          </div>
                          <div className="text-[10px] sm:text-xs text-muted-foreground mt-0.5 max-w-[120px] sm:max-w-[180px] truncate">
                            {job.targets?.slice(0, 2).join(", ")}
                            {job.targets && job.targets.length > 2 && ` +${job.targets.length - 2}`}
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center gap-2 sm:gap-4 shrink-0">
                        {/* Stats */}
                        {job.statistics && (
                          <div className="text-right hidden md:block">
                            <div className="text-sm font-medium text-foreground">
                              {job.statistics.crawled_count || 0}
                            </div>
                            <div className="text-xs text-muted-foreground">수집됨</div>
                          </div>
                        )}

                        {/* Status */}
                        <div className={cn(
                          "flex items-center gap-1 sm:gap-1.5 px-1.5 sm:px-2.5 py-1 sm:py-1.5 rounded-md border min-w-[60px] sm:min-w-[85px]",
                          statusConfig.bg
                        )}>
                          <StatusIcon className={cn("w-3 h-3 sm:w-3.5 sm:h-3.5", statusConfig.color, statusConfig.spin && "animate-spin")} />
                          <span className={cn("text-[10px] sm:text-xs font-medium", statusConfig.color)}>
                            {statusConfig.label}
                          </span>
                        </div>

                        {/* Time */}
                        <div className="text-[10px] sm:text-xs text-muted-foreground min-w-[50px] sm:min-w-[70px] text-right hidden sm:block">
                          {formatDate(job.created_at)}
                        </div>

                        {/* Arrow */}
                        <ChevronRight className={cn(
                          "w-4 h-4 text-muted-foreground transition-all hidden sm:block",
                          isSelected ? "rotate-90 opacity-100" : "opacity-0 group-hover:opacity-100"
                        )} />
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="text-center py-12">
                <div className="w-14 h-14 rounded-xl bg-muted flex items-center justify-center mx-auto mb-4">
                  <Clock className="w-7 h-7 text-muted-foreground" />
                </div>
                <p className="text-sm text-foreground mb-1">수집 작업이 없습니다</p>
                <p className="text-xs text-muted-foreground">왼쪽에서 새 작업을 시작하세요</p>
              </div>
            )}

            {/* Real-time Logs Panel */}
            {logsData && logsData.logs.length > 0 && (
              <div className="mt-4 p-4 rounded-lg bg-slate-900 border border-slate-700">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <div className={cn(
                      "w-2 h-2 rounded-full",
                      logsData.status === "running" ? "bg-blue-500 animate-pulse" :
                      logsData.status === "completed" ? "bg-emerald-500" :
                      logsData.status === "failed" ? "bg-red-500" : "bg-amber-500"
                    )} />
                    <span className="text-xs font-medium text-slate-300">
                      실시간 로그 {logsData.progress && `- ${logsData.progress}`}
                    </span>
                  </div>
                  <button
                    onClick={() => setSelectedJobId(null)}
                    className="text-slate-500 hover:text-slate-300 transition-colors"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
                <div className="space-y-1 max-h-[200px] overflow-y-auto font-mono text-xs">
                  {logsData.logs.map((log, index) => (
                    <div
                      key={index}
                      className={cn(
                        "flex gap-2 py-0.5",
                        log.level === "error" ? "text-red-400" :
                        log.level === "warning" ? "text-amber-400" : "text-slate-400"
                      )}
                    >
                      <span className="text-slate-600 shrink-0">
                        {new Date(log.timestamp).toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                      </span>
                      <span>{log.message}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
