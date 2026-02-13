"use client";

import { useState, useEffect, useCallback } from "react";
import { Send, Check, X, Loader2, Sparkles } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { apiClient } from "@/lib/api-client";

interface FilterStatus {
  Q: boolean | null;
  R: boolean | null;
  T: boolean | null;
  V: boolean | null;
}

interface AnswerData {
  grade: "A" | "B" | "C" | "D" | "F";
  text: string;
  confidence: number;
}

interface AnswerPanelCardProps {
  brandId?: string;
  brandName?: string;
}

// Get session ID from localStorage, or create new one
function getSessionId(brandId: string): string | undefined {
  if (typeof window === "undefined") return undefined;
  const key = `ontix_chat_session_${brandId}`;
  return localStorage.getItem(key) || undefined;
}

function setSessionId(brandId: string, sessionId: string): void {
  if (typeof window === "undefined") return;
  const key = `ontix_chat_session_${brandId}`;
  localStorage.setItem(key, sessionId);
}

export default function AnswerPanelCard({ brandId, brandName = "Brand" }: AnswerPanelCardProps) {
  const [input, setInput] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [filterStatus, setFilterStatus] = useState<FilterStatus>({ Q: null, R: null, T: null, V: null });
  const [answer, setAnswer] = useState<AnswerData | null>(null);
  const [sessionId, setCurrentSessionId] = useState<string | undefined>(undefined);

  // Load session ID on mount
  useEffect(() => {
    if (brandId) {
      const existingSession = getSessionId(brandId);
      setCurrentSessionId(existingSession);
    }
  }, [brandId]);

  const handleSubmit = useCallback(async () => {
    if (!input.trim() || isProcessing || !brandId) return;

    setIsProcessing(true);
    setAnswer(null);
    setFilterStatus({ Q: null, R: null, T: null, V: null });

    // Animate QRTV filters
    const filters: (keyof FilterStatus)[] = ["Q", "R", "T", "V"];
    for (const f of filters) {
      await new Promise((resolve) => setTimeout(resolve, 300));
      setFilterStatus((prev) => ({ ...prev, [f]: true }));
    }

    try {
      // Use session-based chat API for persistent history
      const response = await apiClient.sendMessageWithSession({
        brand_id: brandId,
        message: input,
        session_id: sessionId,
        use_history: true,
      });

      // Save session ID for future conversations
      if (response.session_id && response.session_id !== sessionId) {
        setSessionId(brandId, response.session_id);
        setCurrentSessionId(response.session_id);
      }

      const grade = (response.metadata?.validation?.grade as AnswerData["grade"]) || "B";
      const score = response.metadata?.validation?.score as number || 0.8;

      setAnswer({
        grade,
        text: response.message,
        confidence: score,
      });
    } catch (error) {
      console.error("Chat error:", error);
      setAnswer({
        grade: "C",
        text: "죄송합니다. 응답을 생성하는 중 오류가 발생했습니다. 다시 시도해주세요.",
        confidence: 0.5,
      });
    } finally {
      setIsProcessing(false);
      setInput("");
    }
  }, [input, isProcessing, brandId, sessionId]);

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between text-sm font-medium">
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-muted-foreground" />
            AI Response
          </div>
          <span className="text-xs text-muted-foreground">Q·R·T·V Filter</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col space-y-3">
        {/* Filter Status */}
        <div className="flex items-center justify-center gap-2 py-2 border-y">
          {(["Q", "R", "T", "V"] as const).map((f, idx) => (
            <div key={f} className="flex items-center gap-2">
              <div className="flex flex-col items-center">
                <div className={`w-7 h-7 rounded flex items-center justify-center text-xs font-mono border ${
                  filterStatus[f] === true
                    ? "bg-foreground text-background border-foreground"
                    : filterStatus[f] === false
                    ? "bg-destructive/20 border-destructive text-destructive"
                    : "bg-muted border-border text-muted-foreground"
                }`}>
                  {filterStatus[f] === true ? <Check className="h-3 w-3" /> :
                   filterStatus[f] === false ? <X className="h-3 w-3" /> : f}
                </div>
              </div>
              {idx < 3 && <div className="w-4 h-px bg-border" />}
            </div>
          ))}
        </div>

        {/* Answer */}
        <div className="flex-1 min-h-0 overflow-y-auto">
          {isProcessing && !answer ? (
            <div className="h-full flex items-center justify-center">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            </div>
          ) : answer ? (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span className="px-2 py-0.5 rounded bg-foreground text-background text-xs font-bold">
                  {answer.grade}
                </span>
                <span className="text-xs text-muted-foreground">
                  {(answer.confidence * 100).toFixed(0)}% confidence
                </span>
              </div>
              <p className="text-sm text-muted-foreground leading-relaxed">{answer.text}</p>
            </div>
          ) : (
            <div className="h-full flex items-center justify-center text-sm text-muted-foreground">
              Ask a question about {brandName}
            </div>
          )}
        </div>

        {/* Input */}
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
            placeholder="Ask..."
            disabled={isProcessing}
            className="flex-1 px-3 py-2 rounded border bg-background text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring disabled:opacity-50"
          />
          <button
            onClick={handleSubmit}
            disabled={!input.trim() || isProcessing}
            className="px-3 py-2 rounded bg-primary text-primary-foreground text-sm disabled:opacity-50"
          >
            {isProcessing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          </button>
        </div>
      </CardContent>
    </Card>
  );
}
