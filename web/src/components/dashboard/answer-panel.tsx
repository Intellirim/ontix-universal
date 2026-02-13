"use client";

import type { OntixAnswer, Grade } from "@/types/dashboard";
import { GRADE_INFO, MODES, CHANNELS } from "@/types/dashboard";

interface AnswerPanelProps {
  answer: OntixAnswer;
}

const GradeBadge = ({
  grade,
  size = "default",
}: {
  grade: Grade;
  size?: "default" | "large";
}) => {
  const sizeClass = size === "large" ? "grade-badge-lg" : "grade-badge";
  return (
    <div className={`${sizeClass} grade-${grade.toLowerCase()}`}>
      {grade}
    </div>
  );
};

export function AnswerPanel({ answer }: AnswerPanelProps) {
  const modeLabel =
    MODES.find((m) => m.id === answer.mode)?.label.toUpperCase() || answer.mode;
  const channelLabel =
    CHANNELS.find((c) => c.id === answer.channel)?.label.toUpperCase() ||
    answer.channel;
  const gradeInfo = GRADE_INFO[answer.grade];

  // Parse text to render markdown-like formatting
  const renderText = (text: string) => {
    const paragraphs = text.split("\n\n");
    return paragraphs.map((para, i) => {
      // Bold text
      const withBold = para.replace(
        /\*\*(.*?)\*\*/g,
        '<strong class="text-primary font-medium">$1</strong>'
      );

      // Numbered lists
      if (para.match(/^\d+\./m)) {
        const items = para.split(/\n/).filter((line) => line.trim());
        return (
          <ol key={i} className="list-decimal list-inside space-y-2 mb-4">
            {items.map((item, j) => (
              <li
                key={j}
                className="text-secondary text-sm leading-relaxed"
                dangerouslySetInnerHTML={{
                  __html: item.replace(/^\d+\.\s*/, "").replace(
                    /\*\*(.*?)\*\*/g,
                    '<strong class="text-primary font-medium">$1</strong>'
                  ),
                }}
              />
            ))}
          </ol>
        );
      }

      // Bullet lists
      if (para.includes("• ") || para.includes("- ")) {
        const items = para.split(/\n/).filter((line) => line.trim());
        return (
          <ul key={i} className="space-y-2 mb-4">
            {items.map((item, j) => (
              <li key={j} className="flex items-start gap-2 text-secondary text-sm">
                <span className="text-sapphire mt-1.5">•</span>
                <span
                  dangerouslySetInnerHTML={{
                    __html: item.replace(/^[•\-]\s*/, "").replace(
                      /\*\*(.*?)\*\*/g,
                      '<strong class="text-primary font-medium">$1</strong>'
                    ),
                  }}
                />
              </li>
            ))}
          </ul>
        );
      }

      return (
        <p
          key={i}
          className="text-secondary text-sm leading-relaxed mb-4"
          dangerouslySetInnerHTML={{ __html: withBold }}
        />
      );
    });
  };

  return (
    <div className="ontix-card h-full flex flex-col">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="label text-sapphire">{modeLabel}</span>
            <span className="text-muted">/</span>
            <span className="label">{channelLabel}</span>
          </div>
          <p className="text-xs text-muted">
            AI 분석 응답 • 4단계 필터 검증 완료
          </p>
        </div>

        {/* Grade Badge Area */}
        <div className="flex flex-col items-end gap-1">
          <GradeBadge grade={answer.grade} size="large" />
          <span className="label-sm text-center">
            {gradeInfo.status.toUpperCase()}
          </span>
        </div>
      </div>

      {/* Divider */}
      <div className="h-px bg-[#0f172a] mb-4" />

      {/* Content */}
      <div className="answer-content flex-1">{renderText(answer.text)}</div>

      {/* Footer - Grade explanation */}
      <div className="mt-4 pt-4 border-t border-[#0f172a]">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="label-sm">CONFIDENCE</span>
            <span className={`text-xs ${
              answer.grade === "A" || answer.grade === "B"
                ? "text-sapphire"
                : "text-secondary"
            }`}>
              {gradeInfo.label}
            </span>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1.5">
              <div
                className={`w-1.5 h-1.5 rounded-full ${
                  answer.filterStatus === "passed"
                    ? "bg-sapphire"
                    : answer.filterStatus === "warning"
                    ? "bg-[#94A3B8]"
                    : "bg-[#64748B]"
                }`}
              />
              <span className="label-sm">
                {answer.filterStatus === "passed"
                  ? "FILTER PASSED"
                  : answer.filterStatus === "warning"
                  ? "REVIEW SUGGESTED"
                  : "LOW CONFIDENCE"}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
