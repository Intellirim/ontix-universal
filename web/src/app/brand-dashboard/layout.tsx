"use client";

import { ReactNode, useState, useEffect, createContext, useContext } from "react";
import { usePathname } from "next/navigation";
import Link from "next/link";
import {
  ChevronDown,
  LayoutDashboard,
  Building2,
  Brain,
  Menu,
  X,
  ChevronLeft,
  Database,
  LogOut,
  Loader2,
  Shield,
  Network,
} from "lucide-react";
import { useBrands } from "@/hooks/use-api";
import { useAuth } from "@/contexts/auth-context";
import { getBrandIdFromDomain, isAdminDomain, isLocalDevelopment } from "@/lib/domain-utils";

interface BrandContextType {
  selectedBrandId: string;
  setSelectedBrandId: (id: string) => void;
  sidebarCollapsed: boolean;
  setSidebarCollapsed: (collapsed: boolean) => void;
}

export const BrandContext = createContext<BrandContextType | null>(null);

export function useBrandContext() {
  const context = useContext(BrandContext);
  if (!context) {
    throw new Error("useBrandContext must be used within BrandDashboardLayout");
  }
  return context;
}

const navItems = [
  { label: "Control Room", href: "/brand-dashboard", icon: LayoutDashboard },
  { label: "Brand Management", href: "/brand-dashboard/brands", icon: Building2 },
  { label: "Data Pipeline", href: "/brand-dashboard/pipeline", icon: Database },
  { label: "Knowledge Graph", href: "/brand-dashboard/neo4j", icon: Network },
  { label: "AI Advisor", href: "/brand-dashboard/advisor", icon: Brain },
];

export default function BrandDashboardLayout({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const { user, isLoading: authLoading, logout } = useAuth();
  const [selectedBrandId, setSelectedBrandId] = useState<string>("");
  const [showBrandSelect, setShowBrandSelect] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);

  const { data: allBrands, isLoading: brandsLoading } = useBrands();

  // Detect brand from domain
  const domainBrandId = typeof window !== "undefined" ? getBrandIdFromDomain() : null;
  const isAdmin = typeof window !== "undefined" ? isAdminDomain() : false;
  const isLocal = typeof window !== "undefined" ? isLocalDevelopment() : true;

  // Filter brands based on user's permissions
  const brands = allBrands?.filter((brand) => {
    if (!user) return false;
    if (user.role === "super_admin") return true;
    return user.brand_ids.includes(brand.id);
  });

  // In domain mode, only show the domain's brand
  const showBrandSelector = isAdmin || isLocal || !domainBrandId;

  // Check if we're on the main dashboard (Control Room)
  const isControlRoom = pathname === "/brand-dashboard";

  // Local mode: no login redirect needed

  // Set brand from domain or default to first brand
  useEffect(() => {
    if (brands && brands.length > 0) {
      if (domainBrandId) {
        // Domain-based brand selection
        const domainBrand = brands.find((b) => b.id === domainBrandId);
        if (domainBrand) {
          setSelectedBrandId(domainBrand.id);
        } else {
          // Brand not found or not authorized - redirect to admin
          if (!isLocal) {
            window.location.href = "https://admin.ontix.co.kr/brand-dashboard";
          }
        }
      } else if (!selectedBrandId) {
        // No domain brand - select first available
        setSelectedBrandId(brands[0].id);
      }
    }
  }, [brands, domainBrandId, selectedBrandId, isLocal]);

  const selectedBrand = brands?.find((b) => b.id === selectedBrandId);

  // Show loading while initializing
  if (authLoading) {
    return (
      <div className="dark min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  // For Control Room, render full-screen dark layout
  if (isControlRoom) {
    return (
      <BrandContext.Provider value={{ selectedBrandId, setSelectedBrandId, sidebarCollapsed, setSidebarCollapsed }}>
        <div className="dark min-h-screen bg-background">
          {/* Top Bar with Brand Selector and Navigation */}
          <div className="absolute top-0 left-0 right-0 z-40 p-4 sm:p-6 flex items-center justify-between">
            {/* Brand Selector - Only show dropdown when on admin/local */}
            <div className="relative">
              {showBrandSelector ? (
                <button
                  onClick={() => setShowBrandSelect(!showBrandSelect)}
                  className="group flex items-center gap-3 pl-4 pr-3 py-2 rounded-xl bg-gradient-to-r from-card to-card/80 text-card-foreground border border-border/50 hover:border-primary/30 hover:shadow-lg hover:shadow-primary/5 transition-all duration-300"
                >
                  <div className="flex flex-col items-start">
                    <span className="text-[10px] uppercase tracking-widest text-muted-foreground/70 font-medium">
                      {brandsLoading ? "Loading" : "Brand"}
                    </span>
                    <span className="text-sm font-semibold text-foreground">
                      {brandsLoading ? "..." : selectedBrand?.name || "Select"}
                    </span>
                  </div>
                  <div className={`w-6 h-6 rounded-md bg-muted/50 flex items-center justify-center transition-all duration-300 ${showBrandSelect ? "rotate-180 bg-primary/10" : "group-hover:bg-primary/10"}`}>
                    <ChevronDown className="w-3.5 h-3.5 text-muted-foreground group-hover:text-primary transition-colors" />
                  </div>
                </button>
              ) : (
                // Static brand display for domain-specific subdomains
                <div className="flex items-center gap-3 pl-4 pr-4 py-2 rounded-xl bg-gradient-to-r from-card to-card/80 text-card-foreground border border-border/50">
                  <div className="flex flex-col items-start">
                    <span className="text-[10px] uppercase tracking-widest text-muted-foreground/70 font-medium">
                      Brand
                    </span>
                    <span className="text-sm font-semibold text-foreground">
                      {brandsLoading ? "..." : selectedBrand?.name || domainBrandId}
                    </span>
                  </div>
                </div>
              )}

              {/* Brand Dropdown */}
              {showBrandSelect && showBrandSelector && (
                <>
                  <div className="fixed inset-0 z-40" onClick={() => setShowBrandSelect(false)} />
                  <div className="absolute top-full left-0 mt-2 w-64 bg-card/95 backdrop-blur-xl text-card-foreground rounded-xl border border-border/50 shadow-2xl z-50 py-1.5 max-h-80 overflow-y-auto overscroll-contain">
                    <div className="px-3 py-2 border-b border-border/50">
                      <span className="text-[10px] uppercase tracking-widest text-muted-foreground/70 font-medium">Select Brand</span>
                    </div>
                    {brands?.map((brand) => (
                      <button
                        key={brand.id}
                        onClick={() => {
                          setSelectedBrandId(brand.id);
                          setShowBrandSelect(false);
                        }}
                        className={`w-full px-3 py-2.5 text-left flex items-center gap-3 transition-all ${
                          brand.id === selectedBrandId
                            ? "bg-primary/10 text-foreground"
                            : "hover:bg-muted/50 text-muted-foreground hover:text-foreground"
                        }`}
                      >
                        <div className={`w-8 h-8 rounded-lg flex items-center justify-center text-xs font-bold ${
                          brand.id === selectedBrandId
                            ? "bg-primary text-primary-foreground"
                            : "bg-muted text-muted-foreground"
                        }`}>
                          {brand.name.charAt(0).toUpperCase()}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="font-medium text-sm truncate">{brand.name}</div>
                          {brand.industry && (
                            <div className="text-[10px] text-muted-foreground truncate">{brand.industry}</div>
                          )}
                        </div>
                        {brand.id === selectedBrandId && (
                          <div className="w-1.5 h-1.5 rounded-full bg-primary" />
                        )}
                      </button>
                    ))}
                  </div>
                </>
              )}
            </div>

            {/* Navigation Toggle */}
            <div>
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className={`group flex items-center gap-2.5 pl-4 pr-3 py-2 rounded-xl border transition-all duration-300 ${
                mobileMenuOpen
                  ? "bg-primary/10 border-primary/30 shadow-lg shadow-primary/10"
                  : "bg-gradient-to-r from-card to-card/80 border-border/50 hover:border-primary/30 hover:shadow-lg hover:shadow-primary/5"
              }`}
            >
              <span className="text-sm font-medium text-foreground">Navigation</span>
              <div className={`w-6 h-6 rounded-md flex items-center justify-center transition-all duration-300 ${
                mobileMenuOpen ? "bg-primary/20" : "bg-muted/50 group-hover:bg-primary/10"
              }`}>
                {mobileMenuOpen ? (
                  <X className="w-3.5 h-3.5 text-primary" />
                ) : (
                  <Menu className="w-3.5 h-3.5 text-muted-foreground group-hover:text-primary transition-colors" />
                )}
              </div>
            </button>
            </div>
          </div>

          {/* Slide-out Navigation - Premium Design */}
          {mobileMenuOpen && (
            <>
              <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40" onClick={() => setMobileMenuOpen(false)} />
              <aside className="fixed right-0 top-0 bottom-0 w-72 bg-card/95 backdrop-blur-xl text-card-foreground border-l border-border/50 z-50 overflow-y-auto shadow-2xl">
                <div className="p-5">
                  {/* Header */}
                  <div className="flex items-center justify-between mb-6 pb-4 border-b border-border/50">
                    <div>
                      <span className="text-[10px] uppercase tracking-widest text-muted-foreground/70 font-medium">ONTIX</span>
                      <h2 className="text-sm font-semibold text-foreground">Navigation</h2>
                    </div>
                    <button
                      onClick={() => setMobileMenuOpen(false)}
                      className="w-8 h-8 rounded-lg flex items-center justify-center hover:bg-muted/50 transition-colors"
                    >
                      <X className="w-4 h-4 text-muted-foreground" />
                    </button>
                  </div>

                  {/* Nav Items */}
                  <nav className="space-y-1">
                    {navItems.map((item) => {
                      const isActive = pathname === item.href ||
                        (item.href !== "/brand-dashboard" && pathname.startsWith(item.href));
                      const Icon = item.icon;

                      return (
                        <Link
                          key={item.href}
                          href={item.href}
                          onClick={() => setMobileMenuOpen(false)}
                          className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all ${
                            isActive
                              ? "bg-primary/10 text-foreground"
                              : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
                          }`}
                        >
                          <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                            isActive ? "bg-primary text-primary-foreground" : "bg-muted/50"
                          }`}>
                            <Icon className="w-4 h-4" />
                          </div>
                          <span className="text-sm font-medium">{item.label}</span>
                        </Link>
                      );
                    })}
                  </nav>
                </div>
              </aside>
            </>
          )}

          {/* Main Content - Full Screen */}
          <main className="min-h-screen">
            {children}
          </main>
        </div>
      </BrandContext.Provider>
    );
  }

  // Regular layout for other pages - Dark theme
  return (
    <BrandContext.Provider value={{ selectedBrandId, setSelectedBrandId, sidebarCollapsed, setSidebarCollapsed }}>
      <div className="dark min-h-screen bg-background">
        {/* Header */}
        <header className="fixed top-0 left-0 right-0 z-50 h-16 bg-card border-b border-border">
          <div className="flex items-center justify-between h-full px-4 lg:px-6">
            {/* Left */}
            <div className="flex items-center gap-4">
              {/* Mobile menu button */}
              <button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="lg:hidden p-2 rounded-lg hover:bg-muted text-foreground"
              >
                {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </button>

              {/* Logo */}
              <Link href="/brand-dashboard" className="flex items-center">
                <span className="font-bold text-lg text-foreground tracking-tight">ONTIX</span>
              </Link>

              {/* Back to Control Room */}
              <Link
                href="/brand-dashboard"
                className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-muted hover:bg-accent transition-colors text-sm text-muted-foreground hover:text-foreground"
              >
                <ChevronLeft className="w-4 h-4" />
                Control Room
              </Link>

              {/* Brand Selector - Only show dropdown when on admin/local */}
              <div className="ml-2">
                {showBrandSelector ? (
                  <button
                    onClick={() => setShowBrandSelect(!showBrandSelect)}
                    className="group flex items-center gap-2 pl-3 pr-2 py-1.5 rounded-lg border border-border/50 hover:border-primary/30 hover:bg-muted/50 transition-all duration-200"
                  >
                    <div className={`w-6 h-6 rounded-md flex items-center justify-center text-[10px] font-bold ${
                      selectedBrand ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground"
                    }`}>
                      {selectedBrand?.name?.charAt(0).toUpperCase() || "?"}
                    </div>
                    <span className="text-sm font-medium text-foreground">
                      {brandsLoading ? "..." : selectedBrand?.name || "선택"}
                    </span>
                    <div className={`w-5 h-5 rounded flex items-center justify-center transition-all duration-200 ${showBrandSelect ? "rotate-180" : ""}`}>
                      <ChevronDown className="w-3.5 h-3.5 text-muted-foreground" />
                    </div>
                  </button>
                ) : (
                  // Static brand display for domain-specific subdomains
                  <div className="flex items-center gap-2 pl-3 pr-3 py-1.5 rounded-lg border border-border/50 bg-muted/30">
                    <div className={`w-6 h-6 rounded-md flex items-center justify-center text-[10px] font-bold ${
                      selectedBrand ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground"
                    }`}>
                      {selectedBrand?.name?.charAt(0).toUpperCase() || "?"}
                    </div>
                    <span className="text-sm font-medium text-foreground">
                      {brandsLoading ? "..." : selectedBrand?.name || domainBrandId}
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* Brand Dropdown - Fixed Position */}
            {showBrandSelect && showBrandSelector && !isControlRoom && (
              <>
                <div className="fixed inset-0 z-40" onClick={() => setShowBrandSelect(false)} />
                <div className="fixed top-16 left-[280px] sm:left-[320px] w-64 bg-card/95 backdrop-blur-xl rounded-xl border border-border/50 shadow-2xl z-50 py-1.5 max-h-72 overflow-y-auto">
                  <div className="px-3 py-2 border-b border-border/50">
                    <span className="text-[10px] uppercase tracking-widest text-muted-foreground/70 font-medium">브랜드 선택</span>
                  </div>
                  {brands?.map((brand) => (
                    <button
                      key={brand.id}
                      onClick={() => {
                        setSelectedBrandId(brand.id);
                        setShowBrandSelect(false);
                      }}
                      className={`w-full px-3 py-2.5 text-left flex items-center gap-3 transition-all ${
                        brand.id === selectedBrandId
                          ? "bg-primary/10 text-foreground"
                          : "hover:bg-muted/50 text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      <div className={`w-7 h-7 rounded-md flex items-center justify-center text-xs font-bold ${
                        brand.id === selectedBrandId
                          ? "bg-primary text-primary-foreground"
                          : "bg-muted text-muted-foreground"
                      }`}>
                        {brand.name.charAt(0).toUpperCase()}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-sm truncate">{brand.name}</div>
                        {brand.industry && (
                          <div className="text-[10px] text-muted-foreground truncate">{brand.industry}</div>
                        )}
                      </div>
                      {brand.id === selectedBrandId && (
                        <div className="w-1.5 h-1.5 rounded-full bg-primary" />
                      )}
                    </button>
                  ))}
                </div>
              </>
            )}

            {/* Right */}
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-emerald-500" />
                <span className="text-xs font-medium text-muted-foreground hidden sm:block">LIVE</span>
              </div>

              {/* User Menu */}
              <div className="relative">
                <button
                  onClick={() => setShowUserMenu(!showUserMenu)}
                  className="flex items-center gap-2 pl-2 pr-3 py-1.5 rounded-lg hover:bg-muted transition-colors"
                >
                  <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                    <span className="text-xs font-bold text-primary">
                      {user?.name?.charAt(0).toUpperCase() || "?"}
                    </span>
                  </div>
                  <div className="hidden sm:block text-left">
                    <div className="text-sm font-medium text-foreground">{user?.name}</div>
                    <div className="text-[10px] text-muted-foreground flex items-center gap-1">
                      {user?.role === "super_admin" && <Shield className="w-3 h-3" />}
                      {user?.role === "super_admin" ? "Super Admin" : "Brand Owner"}
                    </div>
                  </div>
                  <ChevronDown className={`w-4 h-4 text-muted-foreground transition-transform ${showUserMenu ? "rotate-180" : ""}`} />
                </button>

                {showUserMenu && (
                  <>
                    <div className="fixed inset-0 z-10" onClick={() => setShowUserMenu(false)} />
                    <div className="absolute top-full right-0 mt-2 w-56 bg-card/95 backdrop-blur-xl rounded-xl border border-border/50 shadow-2xl z-20 py-2">
                      <div className="px-4 py-2 border-b border-border/50">
                        <div className="text-sm font-medium text-foreground">{user?.email}</div>
                        <div className="text-xs text-muted-foreground mt-0.5">
                          {user?.role === "super_admin" ? "전체 브랜드 관리" : `${user?.brand_ids?.length || 0}개 브랜드`}
                        </div>
                      </div>
                      <button
                        onClick={() => {
                          setShowUserMenu(false);
                          logout();
                        }}
                        className="w-full px-4 py-2.5 text-left flex items-center gap-3 text-red-400 hover:bg-red-500/10 transition-colors"
                      >
                        <LogOut className="w-4 h-4" />
                        <span className="text-sm">로그아웃</span>
                      </button>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        </header>

        {/* Sidebar - Desktop */}
        <aside className="hidden lg:block fixed left-0 top-16 bottom-0 w-56 bg-card border-r border-border overflow-y-auto">
          <nav className="p-3">
            {navItems.map((item) => {
              const isActive = pathname === item.href ||
                (item.href !== "/brand-dashboard" && pathname.startsWith(item.href));
              const Icon = item.icon;

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`flex items-center gap-3 px-3 py-2.5 rounded-lg mb-1 transition-colors ${
                    isActive
                      ? "bg-primary/10 text-primary"
                      : "text-muted-foreground hover:bg-muted hover:text-foreground"
                  }`}
                >
                  <Icon className="w-5 h-5 flex-shrink-0" />
                  <span className="text-sm font-medium">{item.label}</span>
                </Link>
              );
            })}
          </nav>
        </aside>

        {/* Mobile Sidebar */}
        {mobileMenuOpen && (
          <>
            <div className="fixed inset-0 bg-black/50 z-40 lg:hidden" onClick={() => setMobileMenuOpen(false)} />
            <aside className="fixed left-0 top-16 bottom-0 w-64 bg-card z-50 lg:hidden overflow-y-auto">
              <nav className="p-3">
                {navItems.map((item) => {
                  const isActive = pathname === item.href ||
                    (item.href !== "/brand-dashboard" && pathname.startsWith(item.href));
                  const Icon = item.icon;

                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      onClick={() => setMobileMenuOpen(false)}
                      className={`flex items-center gap-3 px-3 py-2.5 rounded-lg mb-1 transition-colors ${
                        isActive
                          ? "bg-primary/10 text-primary"
                          : "text-muted-foreground hover:bg-muted hover:text-foreground"
                      }`}
                    >
                      <Icon className="w-5 h-5 flex-shrink-0" />
                      <span className="text-sm font-medium">{item.label}</span>
                    </Link>
                  );
                })}
              </nav>
            </aside>
          </>
        )}

        {/* Main Content */}
        <main className="pt-16 lg:pl-56 min-h-screen">
          <div className="p-4 lg:p-6">
            {children}
          </div>
        </main>
      </div>
    </BrandContext.Provider>
  );
}
