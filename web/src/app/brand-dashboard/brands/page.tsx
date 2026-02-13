"use client";

import { useState } from "react";
import { Plus, Trash2, Edit2, Check, X, Building2, AlertCircle, Database, ChevronDown, ChevronUp, Loader2, User, Lock, Eye, EyeOff } from "lucide-react";
import { useBrands, useCreateBrand, useCurrentUser, useUpdateProfile, useChangePassword } from "@/hooks/use-api";
import { apiClient } from "@/lib/api-client";
import { cn } from "@/lib/utils";
import { useBrandContext } from "../layout";
import { useMutation, useQueryClient } from "@tanstack/react-query";

const AVAILABLE_FEATURES = [
  { id: "product_recommendation", name: "상품 추천", description: "AI 기반 상품 추천 기능" },
  { id: "analytics", name: "분석", description: "대화 및 사용자 분석" },
  { id: "content_generation", name: "콘텐츠 생성", description: "AI 콘텐츠 자동 생성" },
  { id: "social_monitoring", name: "소셜 모니터링", description: "SNS 언급 추적" },
  { id: "advisor", name: "어드바이저", description: "AI 브랜드 컨설턴트" },
  { id: "onboarding", name: "온보딩", description: "사용자 온보딩 가이드" },
];

const INDUSTRY_OPTIONS = [
  "패션/의류", "뷰티/화장품", "식품/음료", "테크/IT", "금융/핀테크",
  "헬스케어", "교육", "여행/관광", "엔터테인먼트", "리테일/유통", "기타",
];

interface FormData {
  id: string;
  name: string;
  description: string;
  industry: string;
  features: string[];
  neo4j_brand_id: string;
  neo4j_namespaces: string[];
}

const initialFormData: FormData = {
  id: "", name: "", description: "", industry: "",
  features: [], neo4j_brand_id: "", neo4j_namespaces: [],
};

export default function BrandsPage() {
  const { selectedBrandId, setSelectedBrandId } = useBrandContext();
  const queryClient = useQueryClient();
  const { data: brands, isLoading } = useBrands();
  const createBrand = useCreateBrand();

  // Account management hooks
  const { data: currentUser } = useCurrentUser();
  const updateProfile = useUpdateProfile();
  const changePassword = useChangePassword();

  const [showCreateForm, setShowCreateForm] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [namespaceInput, setNamespaceInput] = useState("");
  const [formData, setFormData] = useState<FormData>(initialFormData);
  const [errors, setErrors] = useState<Partial<Record<keyof FormData, string>>>({});

  // Account management state
  const [newUserId, setNewUserId] = useState("");
  const [newName, setNewName] = useState("");
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showCurrentPassword, setShowCurrentPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [profileMessage, setProfileMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);
  const [passwordMessage, setPasswordMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);

  const deleteBrand = useMutation({
    mutationFn: (brandId: string) => apiClient.deleteBrand(brandId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["brands"] }),
  });

  const updateBrand = useMutation({
    mutationFn: ({ brandId, data }: { brandId: string; data: Partial<FormData> }) =>
      apiClient.updateBrand(brandId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["brands"] });
      setEditingId(null);
    },
  });

  const validateBrandId = (id: string): boolean => {
    if (id.length < 2 || id.length > 50) return false;
    return /^[a-z0-9][a-z0-9-]*[a-z0-9]$|^[a-z0-9]$/.test(id);
  };

  const validateForm = (): boolean => {
    const newErrors: Partial<Record<keyof FormData, string>> = {};
    if (!formData.id) newErrors.id = "브랜드 ID는 필수입니다";
    else if (!validateBrandId(formData.id)) newErrors.id = "영문 소문자, 숫자, 하이픈만 사용 (2-50자)";
    if (!formData.name || formData.name.length < 1) newErrors.name = "브랜드 이름은 필수입니다";
    else if (formData.name.length > 100) newErrors.name = "브랜드 이름은 100자 이하";
    if (formData.description && formData.description.length > 500) newErrors.description = "설명은 500자 이하";
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleCreate = async () => {
    if (!validateForm()) return;
    try {
      await createBrand.mutateAsync({
        id: formData.id, name: formData.name,
        description: formData.description || undefined,
        industry: formData.industry || undefined,
        features: formData.features.length > 0 ? formData.features : undefined,
        neo4j_brand_id: formData.neo4j_brand_id || undefined,
        neo4j_namespaces: formData.neo4j_namespaces.length > 0 ? formData.neo4j_namespaces : undefined,
      });
      setShowCreateForm(false);
      setFormData(initialFormData);
      setShowAdvanced(false);
      setErrors({});
    } catch (error) {
      console.error("Failed to create brand:", error);
    }
  };

  const handleUpdate = async (brandId: string) => {
    try {
      await updateBrand.mutateAsync({
        brandId,
        data: {
          name: formData.name,
          description: formData.description || undefined,
          industry: formData.industry || undefined,
          features: formData.features,
          neo4j_brand_id: formData.neo4j_brand_id || undefined,
          neo4j_namespaces: formData.neo4j_namespaces,
        },
      });
    } catch (error) {
      console.error("Failed to update brand:", error);
    }
  };

  const handleDelete = async (brandId: string) => {
    if (!confirm("정말로 이 브랜드를 삭제하시겠습니까?\n\n이 작업은 되돌릴 수 없습니다.")) return;
    try {
      await deleteBrand.mutateAsync(brandId);
      if (selectedBrandId === brandId) setSelectedBrandId("");
    } catch (error) {
      console.error("Failed to delete brand:", error);
    }
  };

  const startEdit = (brand: NonNullable<typeof brands>[number]) => {
    setEditingId(brand.id);
    setFormData({
      id: brand.id, name: brand.name,
      description: brand.description || "",
      industry: brand.industry || "",
      features: brand.features || [],
      neo4j_brand_id: "", neo4j_namespaces: [],
    });
  };

  const addNamespace = () => {
    if (namespaceInput.trim() && !formData.neo4j_namespaces.includes(namespaceInput.trim())) {
      setFormData({ ...formData, neo4j_namespaces: [...formData.neo4j_namespaces, namespaceInput.trim()] });
      setNamespaceInput("");
    }
  };

  const removeNamespace = (ns: string) => {
    setFormData({ ...formData, neo4j_namespaces: formData.neo4j_namespaces.filter((n) => n !== ns) });
  };

  // Account management handlers
  const handleProfileUpdate = async () => {
    if (!newUserId && !newName) {
      setProfileMessage({ type: "error", text: "변경할 정보를 입력하세요" });
      return;
    }
    try {
      await updateProfile.mutateAsync({
        email: newUserId || undefined,
        name: newName || undefined,
      });
      setProfileMessage({ type: "success", text: "정보가 변경되었습니다" });
      setNewUserId("");
      setNewName("");
      setTimeout(() => setProfileMessage(null), 3000);
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : "변경에 실패했습니다";
      setProfileMessage({ type: "error", text: errorMessage });
    }
  };

  const handlePasswordChange = async () => {
    if (!currentPassword || !newPassword) {
      setPasswordMessage({ type: "error", text: "모든 필드를 입력하세요" });
      return;
    }
    if (newPassword !== confirmPassword) {
      setPasswordMessage({ type: "error", text: "새 비밀번호가 일치하지 않습니다" });
      return;
    }
    if (newPassword.length < 4) {
      setPasswordMessage({ type: "error", text: "비밀번호는 4자 이상이어야 합니다" });
      return;
    }
    try {
      await changePassword.mutateAsync({
        current_password: currentPassword,
        new_password: newPassword,
      });
      setPasswordMessage({ type: "success", text: "비밀번호가 변경되었습니다" });
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
      setTimeout(() => setPasswordMessage(null), 3000);
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : "비밀번호 변경에 실패했습니다";
      setPasswordMessage({ type: "error", text: errorMessage });
    }
  };

  const inputClass = "w-full px-4 py-3 rounded-lg bg-muted/50 border border-border text-foreground text-sm placeholder:text-muted-foreground focus:border-primary/50 focus:outline-none focus:ring-1 focus:ring-primary/20 transition-all";
  const inputErrorClass = "border-red-500/50 focus:border-red-500/50 focus:ring-red-500/20";
  const labelClass = "block text-sm font-medium text-muted-foreground mb-2";
  const errorTextClass = "text-xs text-red-400 mt-1";

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-6 sm:mb-8">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-foreground">브랜드 관리</h1>
          <p className="text-xs sm:text-sm text-muted-foreground mt-1">브랜드를 추가하고 Neo4j 연동 설정을 관리합니다</p>
        </div>
        <button
          onClick={() => { setShowCreateForm(true); setFormData(initialFormData); setErrors({}); }}
          className="inline-flex items-center justify-center gap-2 px-4 sm:px-5 py-2 sm:py-2.5 bg-primary text-primary-foreground text-sm font-medium rounded-lg hover:bg-primary/90 transition-colors w-full sm:w-auto"
        >
          <Plus className="w-4 h-4" />
          새 브랜드
        </button>
      </div>

      {/* Create Form */}
      {showCreateForm && (
        <div className="bg-card border border-border rounded-xl p-4 sm:p-6 mb-4 sm:mb-6">
          <div className="flex items-center justify-between mb-4 sm:mb-6">
            <h3 className="text-base sm:text-lg font-semibold text-foreground">새 브랜드 추가</h3>
            <button
              onClick={() => { setShowCreateForm(false); setFormData(initialFormData); setErrors({}); }}
              className="p-2 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 sm:gap-4 mb-4 sm:mb-6">
            <div>
              <label className={labelClass}>브랜드 ID <span className="text-red-400">*</span></label>
              <input
                type="text"
                value={formData.id}
                onChange={(e) => {
                  const value = e.target.value.toLowerCase().replace(/[^a-z0-9-]/g, "");
                  setFormData({ ...formData, id: value });
                  if (errors.id) setErrors({ ...errors, id: undefined });
                }}
                placeholder="my-brand (영문 소문자, 숫자, 하이픈)"
                className={cn(inputClass, errors.id && inputErrorClass)}
              />
              {errors.id && <p className={errorTextClass}>{errors.id}</p>}
              <p className="text-xs text-muted-foreground/70 mt-1">고유 식별자로 사용됩니다. 생성 후 변경 불가</p>
            </div>
            <div>
              <label className={labelClass}>브랜드 이름 <span className="text-red-400">*</span></label>
              <input
                type="text"
                value={formData.name}
                onChange={(e) => { setFormData({ ...formData, name: e.target.value }); if (errors.name) setErrors({ ...errors, name: undefined }); }}
                placeholder="My Brand"
                className={cn(inputClass, errors.name && inputErrorClass)}
              />
              {errors.name && <p className={errorTextClass}>{errors.name}</p>}
            </div>
            <div>
              <label className={labelClass}>산업 분류</label>
              <select value={formData.industry} onChange={(e) => setFormData({ ...formData, industry: e.target.value })} className={inputClass}>
                <option value="">선택하세요</option>
                {INDUSTRY_OPTIONS.map((ind) => (<option key={ind} value={ind}>{ind}</option>))}
              </select>
            </div>
            <div>
              <label className={labelClass}>설명</label>
              <input
                type="text"
                value={formData.description}
                onChange={(e) => { setFormData({ ...formData, description: e.target.value }); if (errors.description) setErrors({ ...errors, description: undefined }); }}
                placeholder="브랜드에 대한 간단한 설명"
                maxLength={500}
                className={cn(inputClass, errors.description && inputErrorClass)}
              />
              {errors.description && <p className={errorTextClass}>{errors.description}</p>}
              <p className="text-xs text-muted-foreground/70 mt-1">{formData.description.length}/500자</p>
            </div>
          </div>

          {/* Features */}
          <div className="mb-4 sm:mb-6">
            <label className={labelClass}>활성화할 기능</label>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2 sm:gap-3">
              {AVAILABLE_FEATURES.map((feature) => {
                const isSelected = formData.features.includes(feature.id);
                return (
                  <button
                    key={feature.id}
                    type="button"
                    onClick={() => {
                      const features = isSelected
                        ? formData.features.filter((f) => f !== feature.id)
                        : [...formData.features, feature.id];
                      setFormData({ ...formData, features });
                    }}
                    className={cn(
                      "p-4 rounded-lg text-left transition-all border",
                      isSelected ? "bg-primary/10 border-primary/50" : "bg-muted/30 border-border hover:border-border/80"
                    )}
                  >
                    <div className="flex items-start gap-3">
                      <div className={cn(
                        "w-5 h-5 rounded border-2 flex items-center justify-center flex-shrink-0 mt-0.5",
                        isSelected ? "bg-primary border-primary" : "border-muted-foreground/30"
                      )}>
                        {isSelected && <Check className="w-3 h-3 text-primary-foreground" />}
                      </div>
                      <div>
                        <div className={cn("text-sm font-medium", isSelected ? "text-primary" : "text-foreground")}>{feature.name}</div>
                        <div className="text-xs text-muted-foreground mt-0.5">{feature.description}</div>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Advanced Settings */}
          <div className="border-t border-border pt-4 mb-4 sm:mb-6">
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              <Database className="w-4 h-4" />
              <span>Neo4j 고급 설정</span>
              {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>

            {showAdvanced && (
              <div className="mt-4 p-4 rounded-lg bg-muted/30 border border-border">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <div>
                    <label className={labelClass}>Neo4j 브랜드 ID</label>
                    <input
                      type="text"
                      value={formData.neo4j_brand_id}
                      onChange={(e) => setFormData({ ...formData, neo4j_brand_id: e.target.value })}
                      placeholder="미입력시 브랜드 ID와 동일"
                      className={inputClass}
                    />
                    <p className="text-xs text-muted-foreground/70 mt-1">기존 Neo4j 데이터와 연결할 때 사용</p>
                  </div>
                  <div>
                    <label className={labelClass}>Neo4j 네임스페이스</label>
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={namespaceInput}
                        onChange={(e) => setNamespaceInput(e.target.value)}
                        onKeyPress={(e) => e.key === "Enter" && (e.preventDefault(), addNamespace())}
                        placeholder="네임스페이스 입력 후 Enter"
                        className={inputClass}
                      />
                      <button type="button" onClick={addNamespace} className="px-4 py-3 bg-muted border border-border text-muted-foreground rounded-lg hover:text-foreground hover:border-border/80 transition-colors flex-shrink-0">
                        추가
                      </button>
                    </div>
                    {formData.neo4j_namespaces.length > 0 && (
                      <div className="flex flex-wrap gap-2 mt-3">
                        {formData.neo4j_namespaces.map((ns) => (
                          <span key={ns} className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-muted border border-border text-sm text-foreground">
                            {ns}
                            <button type="button" onClick={() => removeNamespace(ns)} className="text-muted-foreground hover:text-red-400"><X className="w-3.5 h-3.5" /></button>
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Actions */}
          <div className="flex gap-3 pt-4 border-t border-border">
            <button
              onClick={handleCreate}
              disabled={createBrand.isPending}
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-primary text-primary-foreground text-sm font-medium rounded-lg hover:bg-primary/90 transition-colors disabled:opacity-50"
            >
              {createBrand.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Check className="w-4 h-4" />}
              브랜드 생성
            </button>
            <button
              onClick={() => { setShowCreateForm(false); setFormData(initialFormData); setErrors({}); }}
              className="px-5 py-2.5 bg-muted border border-border text-muted-foreground text-sm font-medium rounded-lg hover:text-foreground hover:border-border/80 transition-colors"
            >
              취소
            </button>
          </div>
        </div>
      )}

      {/* Brands List */}
      {isLoading ? (
        <div className="text-center py-12 sm:py-20 text-muted-foreground">
          <Loader2 className="w-8 h-8 sm:w-10 sm:h-10 animate-spin mx-auto mb-4 text-primary" />
          <p className="text-xs sm:text-sm">브랜드 목록을 불러오는 중...</p>
        </div>
      ) : brands && brands.length > 0 ? (
        <div className="space-y-3 sm:space-y-4">
          {brands.map((brand) => (
            <div
              key={brand.id}
              className={cn(
                "bg-card border rounded-xl p-4 sm:p-5 transition-all",
                brand.id === selectedBrandId ? "border-primary/50 bg-primary/5" : "border-border"
              )}
            >
              {editingId === brand.id ? (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h4 className="text-sm font-medium text-muted-foreground">브랜드 수정</h4>
                    <code className="px-2 py-1 rounded-md bg-muted text-xs text-muted-foreground font-mono">{brand.id}</code>
                  </div>
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    <div>
                      <label className={labelClass}>브랜드 이름</label>
                      <input type="text" value={formData.name} onChange={(e) => setFormData({ ...formData, name: e.target.value })} className={inputClass} />
                    </div>
                    <div>
                      <label className={labelClass}>산업</label>
                      <select value={formData.industry} onChange={(e) => setFormData({ ...formData, industry: e.target.value })} className={inputClass}>
                        <option value="">선택하세요</option>
                        {INDUSTRY_OPTIONS.map((ind) => (<option key={ind} value={ind}>{ind}</option>))}
                      </select>
                    </div>
                  </div>
                  <div>
                    <label className={labelClass}>설명</label>
                    <input type="text" value={formData.description} onChange={(e) => setFormData({ ...formData, description: e.target.value })} className={inputClass} />
                  </div>
                  <div>
                    <label className={labelClass}>기능</label>
                    <div className="flex flex-wrap gap-2">
                      {AVAILABLE_FEATURES.map((feature) => {
                        const isSelected = formData.features.includes(feature.id);
                        return (
                          <button
                            key={feature.id}
                            type="button"
                            onClick={() => {
                              const features = isSelected ? formData.features.filter((f) => f !== feature.id) : [...formData.features, feature.id];
                              setFormData({ ...formData, features });
                            }}
                            className={cn(
                              "px-3 py-2 rounded-lg text-sm font-medium transition-all border",
                              isSelected ? "bg-primary/10 text-primary border-primary/50" : "bg-muted/30 text-muted-foreground border-border hover:border-border/80"
                            )}
                          >
                            {feature.name}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                  <div className="flex gap-3 pt-4 border-t border-border">
                    <button
                      onClick={() => handleUpdate(brand.id)}
                      disabled={updateBrand.isPending}
                      className="inline-flex items-center gap-2 px-5 py-2.5 bg-primary text-primary-foreground text-sm font-medium rounded-lg hover:bg-primary/90 transition-colors disabled:opacity-50"
                    >
                      {updateBrand.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Check className="w-4 h-4" />}
                      저장
                    </button>
                    <button onClick={() => setEditingId(null)} className="px-5 py-2.5 bg-muted border border-border text-muted-foreground text-sm font-medium rounded-lg hover:text-foreground hover:border-border/80 transition-colors">
                      취소
                    </button>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4">
                  <div className="flex items-start gap-3 sm:gap-4 flex-1 min-w-0">
                    <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-lg bg-gradient-to-br from-primary/20 to-primary/5 flex items-center justify-center flex-shrink-0">
                      <Building2 className="w-5 h-5 sm:w-6 sm:h-6 text-primary" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 sm:gap-3 flex-wrap">
                        <h3 className="text-base sm:text-lg font-semibold text-foreground">{brand.name}</h3>
                        <code className="px-1.5 sm:px-2 py-0.5 sm:py-1 rounded-md bg-muted text-[10px] sm:text-xs text-muted-foreground font-mono">{brand.id}</code>
                        {brand.id === selectedBrandId && (
                          <span className="px-1.5 sm:px-2 py-0.5 sm:py-1 rounded-md bg-primary/10 text-[10px] sm:text-xs text-primary font-medium">선택됨</span>
                        )}
                      </div>
                      {brand.description && <p className="text-xs sm:text-sm text-muted-foreground mt-1 line-clamp-2">{brand.description}</p>}
                      <div className="flex items-center gap-2 sm:gap-3 mt-2 sm:mt-3 flex-wrap">
                        {brand.industry && <span className="text-[10px] sm:text-xs text-muted-foreground bg-muted px-2 sm:px-3 py-1 sm:py-1.5 rounded-lg">{brand.industry}</span>}
                        {brand.features && brand.features.length > 0 && (
                          <div className="flex gap-1 sm:gap-1.5 flex-wrap">
                            {brand.features.slice(0, 4).map((featureId) => {
                              const feature = AVAILABLE_FEATURES.find((f) => f.id === featureId);
                              return <span key={featureId} className="px-2 sm:px-3 py-1 sm:py-1.5 rounded-lg bg-primary/10 text-[10px] sm:text-xs text-primary">{feature?.name || featureId}</span>;
                            })}
                            {brand.features.length > 4 && <span className="px-2 sm:px-3 py-1 sm:py-1.5 rounded-lg bg-muted text-[10px] sm:text-xs text-muted-foreground">+{brand.features.length - 4}</span>}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0 self-end sm:self-auto">
                    <button
                      onClick={() => setSelectedBrandId(brand.id)}
                      className={cn(
                        "px-4 py-2 rounded-lg text-sm font-medium transition-colors",
                        brand.id === selectedBrandId
                          ? "bg-primary/10 text-primary border border-primary/30"
                          : "bg-muted text-muted-foreground border border-border hover:text-foreground hover:border-border/80"
                      )}
                    >
                      {brand.id === selectedBrandId ? "선택됨" : "선택"}
                    </button>
                    <button onClick={() => startEdit(brand)} className="p-2 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted transition-colors" title="수정">
                      <Edit2 className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => handleDelete(brand.id)}
                      disabled={deleteBrand.isPending}
                      className="p-2 rounded-lg text-muted-foreground hover:text-red-400 hover:bg-red-500/10 transition-colors"
                      title="삭제"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div className="bg-card border border-border rounded-xl p-6 sm:p-8 py-12 sm:py-16 text-center">
          <div className="w-12 h-12 sm:w-16 sm:h-16 rounded-xl bg-muted flex items-center justify-center mx-auto mb-4">
            <AlertCircle className="w-6 h-6 sm:w-8 sm:h-8 text-muted-foreground" />
          </div>
          <h3 className="text-base sm:text-lg font-semibold text-foreground mb-2">브랜드가 없습니다</h3>
          <p className="text-xs sm:text-sm text-muted-foreground mb-4 sm:mb-6">새 브랜드를 추가하여 ONTIX를 시작하세요</p>
          <button
            onClick={() => setShowCreateForm(true)}
            className="inline-flex items-center gap-2 px-4 sm:px-5 py-2 sm:py-2.5 bg-primary text-primary-foreground text-sm font-medium rounded-lg hover:bg-primary/90 transition-colors"
          >
            <Plus className="w-4 h-4" />
            첫 브랜드 추가
          </button>
        </div>
      )}

      {/* Account Management Section */}
      <div className="mt-8 sm:mt-12 pt-8 border-t border-border">
        <h2 className="text-xl sm:text-2xl font-bold text-foreground mb-6">계정 관리</h2>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
          {/* Profile Card */}
          <div className="bg-card border border-border rounded-xl p-4 sm:p-6">
            <div className="flex items-center gap-3 mb-4 sm:mb-6">
              <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                <User className="w-5 h-5 text-primary" />
              </div>
              <div>
                <h3 className="text-base sm:text-lg font-semibold text-foreground">계정 정보</h3>
                <p className="text-xs sm:text-sm text-muted-foreground">아이디와 이름을 변경합니다</p>
              </div>
            </div>

            {currentUser && (
              <div className="mb-4 p-3 rounded-lg bg-muted/50 border border-border">
                <div className="text-xs text-muted-foreground mb-1">현재 아이디</div>
                <div className="text-sm font-medium text-foreground">{currentUser.email}</div>
                <div className="text-xs text-muted-foreground mt-2 mb-1">현재 이름</div>
                <div className="text-sm font-medium text-foreground">{currentUser.name}</div>
              </div>
            )}

            <div className="space-y-4">
              <div>
                <label className={labelClass}>새 아이디</label>
                <input
                  type="text"
                  value={newUserId}
                  onChange={(e) => setNewUserId(e.target.value)}
                  placeholder="새 아이디 입력"
                  className={inputClass}
                />
              </div>
              <div>
                <label className={labelClass}>새 이름</label>
                <input
                  type="text"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  placeholder="새 이름 입력"
                  className={inputClass}
                />
              </div>

              {profileMessage && (
                <div className={cn(
                  "p-3 rounded-lg text-sm",
                  profileMessage.type === "success"
                    ? "bg-green-500/10 text-green-400 border border-green-500/20"
                    : "bg-red-500/10 text-red-400 border border-red-500/20"
                )}>
                  {profileMessage.text}
                </div>
              )}

              <button
                onClick={handleProfileUpdate}
                disabled={updateProfile.isPending}
                className="w-full inline-flex items-center justify-center gap-2 px-5 py-2.5 bg-primary text-primary-foreground text-sm font-medium rounded-lg hover:bg-primary/90 transition-colors disabled:opacity-50"
              >
                {updateProfile.isPending ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Check className="w-4 h-4" />
                )}
                정보 변경
              </button>
            </div>
          </div>

          {/* Password Card */}
          <div className="bg-card border border-border rounded-xl p-4 sm:p-6">
            <div className="flex items-center gap-3 mb-4 sm:mb-6">
              <div className="w-10 h-10 rounded-lg bg-orange-500/10 flex items-center justify-center">
                <Lock className="w-5 h-5 text-orange-500" />
              </div>
              <div>
                <h3 className="text-base sm:text-lg font-semibold text-foreground">비밀번호 변경</h3>
                <p className="text-xs sm:text-sm text-muted-foreground">계정 비밀번호를 변경합니다</p>
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <label className={labelClass}>현재 비밀번호</label>
                <div className="relative">
                  <input
                    type={showCurrentPassword ? "text" : "password"}
                    value={currentPassword}
                    onChange={(e) => setCurrentPassword(e.target.value)}
                    placeholder="현재 비밀번호 입력"
                    className={inputClass}
                  />
                  <button
                    type="button"
                    onClick={() => setShowCurrentPassword(!showCurrentPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                  >
                    {showCurrentPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>
              <div>
                <label className={labelClass}>새 비밀번호</label>
                <div className="relative">
                  <input
                    type={showNewPassword ? "text" : "password"}
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    placeholder="새 비밀번호 입력"
                    className={inputClass}
                  />
                  <button
                    type="button"
                    onClick={() => setShowNewPassword(!showNewPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                  >
                    {showNewPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>
              <div>
                <label className={labelClass}>새 비밀번호 확인</label>
                <input
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="새 비밀번호 다시 입력"
                  className={inputClass}
                />
              </div>

              {passwordMessage && (
                <div className={cn(
                  "p-3 rounded-lg text-sm",
                  passwordMessage.type === "success"
                    ? "bg-green-500/10 text-green-400 border border-green-500/20"
                    : "bg-red-500/10 text-red-400 border border-red-500/20"
                )}>
                  {passwordMessage.text}
                </div>
              )}

              <button
                onClick={handlePasswordChange}
                disabled={changePassword.isPending}
                className="w-full inline-flex items-center justify-center gap-2 px-5 py-2.5 bg-orange-500 text-white text-sm font-medium rounded-lg hover:bg-orange-600 transition-colors disabled:opacity-50"
              >
                {changePassword.isPending ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Lock className="w-4 h-4" />
                )}
                비밀번호 변경
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
