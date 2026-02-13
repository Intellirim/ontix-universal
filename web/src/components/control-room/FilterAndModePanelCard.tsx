"use client";

import Link from "next/link";
import { Settings, Brain, BarChart3, Package, FileText, Users, Globe, ExternalLink } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const AI_MODES = [
  { id: "advisor", label: "Advisor", icon: Brain, href: "/brand-dashboard/advisor", description: "AI 브랜드 어드바이저" },
  { id: "analytics", label: "Analytics", icon: BarChart3, href: "/brand-dashboard/analytics", description: "대화 분석" },
  { id: "product", label: "Product", icon: Package, href: "/brand-dashboard/product-recommendation", description: "상품 추천" },
  { id: "content", label: "Content", icon: FileText, href: "/brand-dashboard/content-generation", description: "콘텐츠 생성" },
  { id: "onboarding", label: "Onboarding", icon: Users, href: "/brand-dashboard/onboarding", description: "온보딩" },
  { id: "monitoring", label: "Monitoring", icon: Globe, href: "/brand-dashboard/social-monitoring", description: "소셜 모니터링" },
];

export default function FilterAndModePanelCard() {
  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-sm font-medium">
          <Settings className="h-4 w-4 text-muted-foreground" />
          Control Panel
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col">
        <div className="text-xs text-muted-foreground mb-2">AI Modules</div>
        <div className="space-y-1 flex-1">
          {AI_MODES.map((m) => {
            const Icon = m.icon;
            return (
              <Link
                key={m.id}
                href={m.href}
                className="w-full flex items-center gap-3 px-3 py-2 rounded text-sm transition-colors hover:bg-accent text-muted-foreground hover:text-accent-foreground group"
              >
                <Icon className="h-4 w-4" />
                <div className="flex-1">
                  <span>{m.label}</span>
                  <span className="text-xs text-muted-foreground ml-2 hidden group-hover:inline">{m.description}</span>
                </div>
                <ExternalLink className="h-3 w-3 opacity-0 group-hover:opacity-100 transition-opacity" />
              </Link>
            );
          })}
        </div>
        <div className="pt-3 mt-3 border-t text-xs text-muted-foreground">
          <span>Click to open module</span>
        </div>
      </CardContent>
    </Card>
  );
}
