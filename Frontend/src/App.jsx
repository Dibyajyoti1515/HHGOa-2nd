// frontend/src/App.jsx

import { useEffect, useState } from "react";
import VoiceAssistant from "./VoiceAssistant";

const API_BASE =
  import.meta.env.VITE_API_BASE ||
  "http://localhost:8000";

function useHealthCheck(intervalMs = 10000) {
  const [health, setHealth] =
    useState("checking");

  useEffect(() => {
    let cancelled = false;

    const check = async () => {
      try {
        const response = await fetch(
          `${API_BASE}/v1/health`
        );

        if (!cancelled) {
          setHealth(
            response.ok ? "ok" : "down"
          );
        }
      } catch {
        if (!cancelled) {
          setHealth("down");
        }
      }
    };

    check();

    const interval = setInterval(
      check,
      intervalMs
    );

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [intervalMs]);

  return health;
}

function HealthBadge({ health }) {
  const config = {
    checking: {
      color: "#9ca3af",
      label: "Connecting",
    },

    ok: {
      color: "#16a34a",
      label: "Backend connected",
    },

    down: {
      color: "#dc2626",
      label: "Backend offline",
    },
  }[health];

  return (
    <div style={styles.healthBadge}>
      <span
        style={{
          ...styles.healthDot,
          background: config.color,
        }}
      />

      <span>
        {config.label}
      </span>
    </div>
  );
}

export default function App() {
  const health = useHealthCheck();

  return (
    <div style={styles.app}>
      {/* =================================================
          HEADER
      ================================================= */}

      <header style={styles.header}>
        <div style={styles.brand}>
          <div style={styles.brandIcon}>
            ✦
          </div>

          <div>
            <div style={styles.brandTitle}>
              Voice Assistant
            </div>

            <div style={styles.brandSubtitle}>
              Knowledge powered
            </div>
          </div>
        </div>

        <HealthBadge health={health} />
      </header>

      {/* =================================================
          MAIN
      ================================================= */}

      <main style={styles.main}>
        <div style={styles.assistantWrapper}>
          <VoiceAssistant />
        </div>
      </main>

      {/* =================================================
          FOOTER
      ================================================= */}

      <footer style={styles.footer}>
        <span>Sarvam STT</span>

        <span style={styles.footerDot}>
          •
        </span>

        <span>Hybrid Retrieval</span>

        <span style={styles.footerDot}>
          •
        </span>

        <span>Bulbul v3</span>
      </footer>
    </div>
  );
}

const styles = {
  app: {
    width: "100%",
    minWidth: "100%",
    minHeight: "100vh",

    display: "flex",
    flexDirection: "column",

    background:
      "radial-gradient(circle at 50% 0%, #f5f7ff 0%, #ffffff 32%, #ffffff 100%)",
  },

  // -------------------------------------------------------
  // Header
  // -------------------------------------------------------

  header: {
    width: "100%",
    height: 72,

    flexShrink: 0,

    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",

    padding:
      "0 clamp(20px, 4vw, 56px)",

    background:
      "rgba(255,255,255,0.88)",

    borderBottom:
      "1px solid #eef0f4",

    backdropFilter:
      "blur(18px)",
  },

  brand: {
    display: "flex",
    alignItems: "center",
    gap: 11,
  },

  brandIcon: {
    width: 34,
    height: 34,

    display: "flex",
    alignItems: "center",
    justifyContent: "center",

    borderRadius: 10,

    background: "#111827",
    color: "#ffffff",

    fontSize: 15,

    boxShadow:
      "0 5px 16px rgba(17,24,39,.16)",
  },

  brandTitle: {
    fontSize: 14,
    fontWeight: 700,
    lineHeight: 1.2,
    color: "#111827",
  },

  brandSubtitle: {
    marginTop: 3,
    fontSize: 10,
    color: "#9ca3af",
  },

  healthBadge: {
    display: "flex",
    alignItems: "center",
    gap: 8,

    padding:
      "7px 11px",

    borderRadius: 999,

    background:
      "#ffffff",

    border:
      "1px solid #edf0f4",

    color: "#6b7280",

    fontSize: 11,

    boxShadow:
      "0 3px 12px rgba(17,24,39,.04)",
  },

  healthDot: {
    width: 7,
    height: 7,
    flexShrink: 0,
    borderRadius: "50%",
  },

  // -------------------------------------------------------
  // Main
  // -------------------------------------------------------

  main: {
    flex: 1,

    width: "100%",
    minWidth: 0,

    display: "flex",
    alignItems: "center",
    justifyContent: "center",

    padding:
      "30px 20px",

    overflowY: "auto",
  },

  assistantWrapper: {
    width: "100%",
    maxWidth: 820,

    display: "flex",
    justifyContent: "center",
  },

  // -------------------------------------------------------
  // Footer
  // -------------------------------------------------------

  footer: {
    width: "100%",

    flexShrink: 0,

    display: "flex",
    alignItems: "center",
    justifyContent: "center",

    gap: 9,

    padding:
      "12px 20px 18px",

    color: "#b0b5bd",

    fontSize: 10,
  },

  footerDot: {
    color: "#d1d5db",
  },
};