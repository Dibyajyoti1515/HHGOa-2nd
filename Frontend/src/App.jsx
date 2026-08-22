// frontend/src/App.jsx

import { useEffect, useState } from "react";
import VoiceAssistant from "./Voiceassistant";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

function useHealthCheck(intervalMs = 10000) {
  const [health, setHealth] = useState("checking");

  useEffect(() => {
    let cancelled = false;

    const check = async () => {
      try {
        const response = await fetch(`${API_BASE}/v1/health`);
        if (!cancelled) setHealth(response.ok ? "ok" : "down");
      } catch {
        if (!cancelled) setHealth("down");
      }
    };

    check();
    const interval = setInterval(check, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [intervalMs]);

  return health;
}

const HEALTH_CONFIG = {
  checking: { color: "#f5f0e1", label: "Connecting", pulse: false },
  ok: { color: "#34d399", label: "Ready", pulse: true },
  down: { color: "#ff2d78", label: "Backend offline", pulse: false },
};

function HealthBadge({ health }) {
  const config = HEALTH_CONFIG[health];
  return (
    <div className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 backdrop-blur-sm">
      <span className="relative flex h-2 w-2">
        {config.pulse && (
          <span
            className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-75"
            style={{ backgroundColor: config.color }}
          />
        )}
        <span
          className="relative inline-flex h-2 w-2 rounded-full"
          style={{ backgroundColor: config.color }}
        />
      </span>
      <span className="font-mono-hh text-[10px] uppercase tracking-[0.2em] text-[#f5f0e1]/70">
        {config.label}
      </span>
    </div>
  );
}

export default function App() {
  const health = useHealthCheck();

  return (
    <div className="relative flex min-h-screen w-full flex-col items-center bg-[#0b2e28] text-[#f5f0e1]">
      {/* ambient background wash */}
      <div
        className="pointer-events-none absolute inset-0 opacity-60"
        style={{
          background:
            "radial-gradient(60% 50% at 50% 0%, rgba(232,197,71,0.08), transparent 60%), radial-gradient(50% 40% at 100% 100%, rgba(255,45,120,0.06), transparent 60%)",
        }}
      />

      {/* header */}
      <div className="relative z-10 flex w-full max-w-2xl items-center justify-between px-4 pt-10">
        <div>
          <h1 className="text-2xl font-extrabold tracking-tight text-[#f5f0e1] sm:text-3xl">
            HH GOA
          </h1>
          <p className="font-mono-hh text-[10px] uppercase tracking-[0.25em] text-[#f5f0e1]/50">
            Goa, India · Ask Anything
          </p>
        </div>
        <HealthBadge health={health} />
      </div>

      {/* main */}
      <main className="relative z-10 flex w-full flex-1 items-center justify-center px-4 py-8">
        <VoiceAssistant />
      </main>

      {/* footer */}
      <footer className="relative z-10 flex items-center justify-center gap-2 px-4 pb-6 font-mono-hh text-[10px] uppercase tracking-[0.15em] text-[#f5f0e1]/30">
        <span>Sarvam STT</span>
        <span className="text-[#f5f0e1]/15">•</span>
        <span>Hybrid Retrieval</span>
        <span className="text-[#f5f0e1]/15">•</span>
        <span>Bulbul v3</span>
      </footer>
    </div>
  );
}