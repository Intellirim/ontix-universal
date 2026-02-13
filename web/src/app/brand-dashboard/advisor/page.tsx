"use client";

import { useState, useRef, useEffect } from "react";
import { Brain, Send, RefreshCw, Lightbulb, MessageSquare, Sparkles, User, Bot } from "lucide-react";
import { useBrandContext } from "../layout";
import { useSendAdvisorMessageWithSession, useBrand } from "@/hooks/use-api";
import { cn } from "@/lib/utils";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  metadata?: {
    quality?: { score: number; level?: string };
    trust?: { score: number; hallucination_risk?: number };
    validation?: { grade: string; score: number };
  };
}

export default function AdvisorPage() {
  const { selectedBrandId } = useBrandContext();
  const { data: brand } = useBrand(selectedBrandId);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string>("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const sendAdvisorMessage = useSendAdvisorMessageWithSession();

  const suggestedQuestions = [
    "이 브랜드의 강점은 무엇인가요?",
    "마케팅 전략을 추천해주세요",
    "고객 세그먼트 분석을 해주세요",
    "경쟁사 대비 차별점은?",
    "매출 향상 방법을 알려주세요",
  ];

  // Generate sessionId only on client side to avoid hydration mismatch
  useEffect(() => {
    if (!sessionId) {
      setSessionId(`advisor-${Date.now()}`);
    }
  }, [sessionId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async (text?: string) => {
    const messageText = text || input.trim();
    if (!messageText || sendAdvisorMessage.isPending) return;

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: "user",
      content: messageText,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");

    try {
      // 전용 AI Advisor 엔드포인트 사용 (/api/v1/advisor)
      const response = await sendAdvisorMessage.mutateAsync({
        brand_id: selectedBrandId,
        message: messageText,
        session_id: sessionId,
      });

      // 세션 ID 업데이트 (서버에서 생성된 경우)
      if (response.session_id && response.session_id !== sessionId) {
        setSessionId(response.session_id);
      }

      const assistantMessage: Message = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: response.message,
        metadata: response.metadata as Message["metadata"],
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch {
      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        role: "assistant",
        content: "죄송합니다. 오류가 발생했습니다. 다시 시도해주세요.",
      };
      setMessages((prev) => [...prev, errorMessage]);
    }
  };

  const handleNewSession = () => {
    setMessages([]);
    setSessionId(`advisor-${Date.now()}`);
  };

  return (
    <div className="h-[calc(100vh-7rem)] sm:h-[calc(100vh-8rem)] flex flex-col">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4 sm:mb-6">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-foreground">AI 어드바이저</h1>
          <p className="text-muted-foreground text-xs sm:text-sm mt-1">
            {brand?.name || "브랜드"} 전문 AI 상담사
          </p>
        </div>
        <button
          onClick={handleNewSession}
          className="flex items-center justify-center gap-2 px-4 py-2 rounded-xl bg-primary text-primary-foreground hover:bg-primary/90 transition-all w-full sm:w-auto"
        >
          <RefreshCw className="w-4 h-4" />
          <span>새 대화</span>
        </button>
      </div>

      {/* Chat Container */}
      <div className="flex-1 flex gap-6 min-h-0">
        {/* Main Chat */}
        <div className="flex-1 flex flex-col rounded-2xl bg-card border shadow-sm overflow-hidden">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-3 sm:p-6 space-y-3 sm:space-y-4">
            {messages.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-center px-4">
                <div className="w-14 h-14 sm:w-20 sm:h-20 rounded-2xl bg-primary/10 flex items-center justify-center mb-4 sm:mb-6">
                  <Brain className="w-7 h-7 sm:w-10 sm:h-10 text-primary" />
                </div>
                <h3 className="text-lg sm:text-xl font-semibold text-foreground mb-2">
                  AI 어드바이저에게 물어보세요
                </h3>
                <p className="text-muted-foreground text-xs sm:text-sm max-w-md mb-6 sm:mb-8">
                  브랜드 전략, 마케팅 조언, 고객 분석 등 다양한 질문에 답변드립니다.
                </p>

                {/* Suggested Questions */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 sm:gap-3 max-w-lg w-full">
                  {suggestedQuestions.slice(0, 4).map((q, idx) => (
                    <button
                      key={idx}
                      onClick={() => handleSend(q)}
                      className="p-2.5 sm:p-3 rounded-xl bg-muted border text-xs sm:text-sm text-muted-foreground hover:text-foreground hover:border-primary/30 transition-all text-left"
                    >
                      <Lightbulb className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-amber-500 mb-1.5 sm:mb-2" />
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((msg) => (
                <div
                  key={msg.id}
                  className={cn(
                    "flex gap-2 sm:gap-3",
                    msg.role === "user" ? "justify-end" : "justify-start"
                  )}
                >
                  {msg.role === "assistant" && (
                    <div className="w-7 h-7 sm:w-8 sm:h-8 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                      <Bot className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-primary" />
                    </div>
                  )}
                  <div
                    className={cn(
                      "max-w-[85%] sm:max-w-[70%] rounded-2xl p-3 sm:p-4",
                      msg.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted border"
                    )}
                  >
                    <p className={cn(
                      "text-sm whitespace-pre-wrap",
                      msg.role === "user" ? "text-primary-foreground" : "text-foreground"
                    )}>
                      {msg.content}
                    </p>

                    {/* Quality Indicators */}
                    {msg.role === "assistant" && msg.metadata?.validation && (
                      <div className="flex items-center gap-3 mt-3 pt-3 border-t border-border">
                        <span className={cn(
                          "text-xs px-2 py-1 rounded-lg",
                          msg.metadata.validation.grade === "A" ? "bg-emerald-500/20 text-emerald-400" :
                          msg.metadata.validation.grade === "B" ? "bg-blue-500/20 text-blue-400" :
                          msg.metadata.validation.grade === "C" ? "bg-amber-500/20 text-amber-400" :
                          "bg-red-500/20 text-red-400"
                        )}>
                          Grade {msg.metadata.validation.grade}
                        </span>
                        {msg.metadata.quality && (
                          <span className="text-xs text-muted-foreground">
                            품질 {(msg.metadata.quality.score * 100).toFixed(0)}%
                          </span>
                        )}
                        {msg.metadata.trust && (msg.metadata.trust.hallucination_risk ?? 0) > 0.3 && (
                          <span className="text-xs text-amber-400">
                            ⚠️ 환각 위험
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                  {msg.role === "user" && (
                    <div className="w-7 h-7 sm:w-8 sm:h-8 rounded-lg bg-muted flex items-center justify-center flex-shrink-0">
                      <User className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-muted-foreground" />
                    </div>
                  )}
                </div>
              ))
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="p-3 sm:p-4 border-t">
            <div className="flex items-center gap-2 sm:gap-3">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend()}
                placeholder="질문을 입력하세요..."
                className="flex-1 px-3 sm:px-4 py-2.5 sm:py-3 rounded-xl bg-background border text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
                disabled={sendAdvisorMessage.isPending}
              />
              <button
                onClick={() => handleSend()}
                disabled={!input.trim() || sendAdvisorMessage.isPending}
                className={cn(
                  "w-10 h-10 sm:w-12 sm:h-12 rounded-xl flex items-center justify-center transition-all shrink-0",
                  input.trim() && !sendAdvisorMessage.isPending
                    ? "bg-primary text-primary-foreground hover:bg-primary/90"
                    : "bg-muted text-muted-foreground cursor-not-allowed"
                )}
              >
                {sendAdvisorMessage.isPending ? (
                  <RefreshCw className="w-4 h-4 sm:w-5 sm:h-5 animate-spin" />
                ) : (
                  <Send className="w-4 h-4 sm:w-5 sm:h-5" />
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Sidebar - Quick Actions */}
        <div className="w-72 space-y-4 hidden xl:block">
          <div className="p-4 rounded-2xl bg-card border shadow-sm">
            <h3 className="text-sm font-semibold text-foreground mb-4 flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-primary" />
              추천 질문
            </h3>
            <div className="space-y-2">
              {suggestedQuestions.map((q, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSend(q)}
                  className="w-full p-3 rounded-xl bg-muted border text-xs text-muted-foreground hover:text-foreground hover:border-primary/30 transition-all text-left"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>

          <div className="p-4 rounded-2xl bg-card border shadow-sm">
            <h3 className="text-sm font-semibold text-foreground mb-4 flex items-center gap-2">
              <MessageSquare className="w-4 h-4 text-emerald-500" />
              세션 정보
            </h3>
            <div className="space-y-3">
              <div className="flex justify-between text-xs">
                <span className="text-muted-foreground">세션 ID</span>
                <span className="text-foreground font-mono">{sessionId ? sessionId.slice(-8) : "--------"}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-muted-foreground">메시지 수</span>
                <span className="text-foreground">{messages.length}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-muted-foreground">필터 버전</span>
                <span className="text-primary">v2.0</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
