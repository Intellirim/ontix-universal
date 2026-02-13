import Link from "next/link";

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center relative bg-[#0a0a0f]">
      {/* Gradient Orbs */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-[500px] h-[500px] bg-blue-500/10 rounded-full blur-[120px]" />
        <div className="absolute bottom-1/4 right-1/4 w-[400px] h-[400px] bg-purple-500/10 rounded-full blur-[100px]" />
      </div>

      {/* Hero */}
      <div className="text-center max-w-3xl mx-auto px-8 relative z-10">
        {/* Logo Badge */}
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 mb-8">
          <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-xs text-white/60 font-medium tracking-wide">
            AI-Powered Platform
          </span>
        </div>

        <h1 className="text-6xl md:text-7xl font-bold tracking-tighter mb-6 bg-gradient-to-b from-white via-white to-white/40 bg-clip-text text-transparent">
          ONTIX
        </h1>
        <p className="text-lg text-white/50 mb-16 text-center max-w-md mx-auto">
          Precision Brand Intelligence Solutions
        </p>

        {/* Navigation Buttons */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
          {/* Admin Button */}
          <Link
            href="/brand-dashboard"
            className="group inline-flex items-center justify-center min-w-[200px] px-8 py-4 rounded-xl bg-white/5 backdrop-blur-xl border border-white/10 text-white/80 font-medium text-sm shadow-lg hover:bg-white/10 hover:border-white/20 hover:text-white transition-all duration-300"
          >
            <svg
              className="w-4 h-4 mr-2 opacity-60 group-hover:opacity-100 transition-opacity"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2"
              />
            </svg>
            Admin Console
          </Link>

          {/* Chat Button */}
          <Link
            href="/chat"
            className="group inline-flex items-center justify-center min-w-[200px] px-8 py-4 rounded-xl bg-gradient-to-r from-blue-600 to-blue-500 text-white font-medium text-sm shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 hover:scale-[1.02] transition-all duration-300"
          >
            <svg
              className="w-4 h-4 mr-2"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
              />
            </svg>
            Brand AI Agent
          </Link>
        </div>
      </div>

      {/* Footer */}
      <footer className="absolute bottom-8 text-center">
        <p className="text-[10px] text-white/30 tracking-wider uppercase">
          Ontix Intelligence
        </p>
      </footer>
    </div>
  );
}
