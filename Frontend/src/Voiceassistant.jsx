import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const WS_BASE = import.meta.env.VITE_WS_BASE || "ws://localhost:8000";
const WS_URL = `${WS_BASE}/v1/voice`;

// Browser microphone -> STT
const INPUT_SAMPLE_RATE = 16000;

// Sarvam Bulbul v3 -> browser
const OUTPUT_SAMPLE_RATE = 24000;

export default function VoiceAssistant() {
  const [status, setStatus] = useState("idle");
  const [transcript, setTranscript] = useState("");
  const [answerText, setAnswerText] = useState("");
  const [timings, setTimings] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  const [currentStage, setCurrentStage] = useState(null);
  const [isPaused, setIsPaused] = useState(false);

  // ---------------------------------------------------------
  // WebSocket
  // ---------------------------------------------------------

  const wsRef = useRef(null);

  // ---------------------------------------------------------
  // Recording
  // ---------------------------------------------------------

  const recordingCtxRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const processorRef = useRef(null);
  const sourceRef = useRef(null);

  // ---------------------------------------------------------
  // Playback
  // ---------------------------------------------------------

  const playbackCtxRef = useRef(null);
  const playbackSourcesRef = useRef(new Set());
  const playbackTimeRef = useRef(0);

  // ---------------------------------------------------------
  // Animation
  // ---------------------------------------------------------

  const typingTimerRef = useRef(null);

  // ---------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------

  const clearTypingTimer = useCallback(() => {
    if (typingTimerRef.current) {
      clearInterval(typingTimerRef.current);
      typingTimerRef.current = null;
    }
  }, []);

  const cleanupRecording = useCallback(() => {
    if (processorRef.current) {
      try {
        processorRef.current.disconnect();
      } catch { }

      processorRef.current = null;
    }

    if (sourceRef.current) {
      try {
        sourceRef.current.disconnect();
      } catch { }

      sourceRef.current = null;
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch { }
      });

      mediaStreamRef.current = null;
    }
  }, []);

  const stopPlayback = useCallback(() => {
    playbackSourcesRef.current.forEach((source) => {
      try {
        source.stop();
      } catch { }
    });

    playbackSourcesRef.current.clear();

    if (playbackCtxRef.current) {
      playbackTimeRef.current =
        playbackCtxRef.current.currentTime;
    } else {
      playbackTimeRef.current = 0;
    }
  }, []);

  const cleanupEverything = useCallback(() => {
    clearTypingTimer();

    cleanupRecording();
    stopPlayback();

    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch { }

      wsRef.current = null;
    }

    setCurrentStage(null);
  }, [
    clearTypingTimer,
    cleanupRecording,
    stopPlayback,
  ]);

  const resetTurn = useCallback(() => {
    cleanupEverything();

    setTranscript("");
    setAnswerText("");
    setTimings(null);
    setErrorMsg(null);
    setCurrentStage(null);
    setIsPaused(false);
    setStatus("idle");
  }, [cleanupEverything]);

  // ---------------------------------------------------------
  // Recording AudioContext
  // ---------------------------------------------------------

  const ensureRecordingCtx = useCallback(() => {
    if (!recordingCtxRef.current) {
      recordingCtxRef.current =
        new (
          window.AudioContext ||
          window.webkitAudioContext
        )({
          sampleRate: INPUT_SAMPLE_RATE,
        });
    }

    return recordingCtxRef.current;
  }, []);

  // ---------------------------------------------------------
  // Playback AudioContext
  // ---------------------------------------------------------

  const ensurePlaybackCtx = useCallback(() => {
    if (!playbackCtxRef.current) {
      playbackCtxRef.current =
        new (
          window.AudioContext ||
          window.webkitAudioContext
        )({
          sampleRate: OUTPUT_SAMPLE_RATE,
        });
    }

    return playbackCtxRef.current;
  }, []);

  // ---------------------------------------------------------
  // Fast typewriter
  // ---------------------------------------------------------

  const animateAnswer = useCallback(
    (text) => {
      clearTypingTimer();

      if (!text) {
        setAnswerText("");
        return;
      }

      setAnswerText("");

      /*
       * Fast typewriter.
       *
       * Target duration ~200 ms.
       * This avoids the annoying 1-2 second artificial
       * "loading" feeling.
       */

      const targetDuration = 200;

      const interval =
        Math.max(
          1,
          Math.floor(
            targetDuration / text.length
          )
        );

      let index = 0;

      typingTimerRef.current =
        setInterval(() => {
          index += 1;

          setAnswerText(
            text.slice(0, index)
          );

          if (index >= text.length) {
            clearTypingTimer();
          }
        }, interval);
    },
    [clearTypingTimer]
  );

  // ---------------------------------------------------------
  // Audio playback
  // ---------------------------------------------------------

  const playChunk = useCallback(
    async (arrayBuffer) => {
      const ctx = ensurePlaybackCtx();

      if (ctx.state === "suspended") {
        try {
          await ctx.resume();
        } catch { }
      }

      /*
       * Sarvam Bulbul v3 is configured for
       * linear16 / 24kHz.
       */

      const int16 = new Int16Array(
        arrayBuffer
      );

      if (!int16.length) {
        return;
      }

      const float32 =
        new Float32Array(int16.length);

      for (
        let i = 0;
        i < int16.length;
        i++
      ) {
        float32[i] =
          int16[i] / 32768;
      }

      const audioBuffer =
        ctx.createBuffer(
          1,
          float32.length,
          OUTPUT_SAMPLE_RATE
        );

      audioBuffer.copyToChannel(
        float32,
        0
      );

      const source =
        ctx.createBufferSource();

      source.buffer = audioBuffer;

      source.connect(
        ctx.destination
      );

      const now =
        ctx.currentTime;

      const startAt = Math.max(
        now,
        playbackTimeRef.current
      );

      source.start(startAt);

      playbackTimeRef.current =
        startAt +
        audioBuffer.duration;

      playbackSourcesRef.current.add(
        source
      );

      source.onended = () => {
        playbackSourcesRef.current.delete(
          source
        );
      };

      setStatus("speaking");
    },
    [ensurePlaybackCtx]
  );

  // ---------------------------------------------------------
  // Start recording
  // ---------------------------------------------------------

  const startRecording = useCallback(
    async () => {
      try {
        setErrorMsg(null);
        setTranscript("");
        setAnswerText("");
        setTimings(null);
        setCurrentStage("stt");
        setIsPaused(false);

        playbackTimeRef.current = 0;

        const ws =
          new WebSocket(WS_URL);

        ws.binaryType =
          "arraybuffer";

        wsRef.current = ws;

        ws.onopen = async () => {
          try {
            setStatus("recording");

            const stream =
              await navigator.mediaDevices.getUserMedia(
                {
                  audio: {
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                  },
                }
              );

            mediaStreamRef.current =
              stream;

            const ctx =
              ensureRecordingCtx();

            if (
              ctx.state ===
              "suspended"
            ) {
              await ctx.resume();
            }

            const source =
              ctx.createMediaStreamSource(
                stream
              );

            sourceRef.current =
              source;

            const processor =
              ctx.createScriptProcessor(
                4096,
                1,
                1
              );

            processorRef.current =
              processor;

            processor.onaudioprocess =
              (event) => {
                if (
                  ws.readyState !==
                  WebSocket.OPEN
                ) {
                  return;
                }

                const input =
                  event.inputBuffer.getChannelData(
                    0
                  );

                const int16 =
                  new Int16Array(
                    input.length
                  );

                for (
                  let i = 0;
                  i < input.length;
                  i++
                ) {
                  const sample =
                    Math.max(
                      -1,
                      Math.min(
                        1,
                        input[i]
                      )
                    );

                  int16[i] =
                    sample < 0
                      ? sample * 0x8000
                      : sample * 0x7fff;
                }

                ws.send(
                  int16.buffer
                );
              };

            source.connect(
              processor
            );

            /*
             * Connect to destination so Chrome
             * continues processing the ScriptProcessor.
             *
             * Volume is muted through a zero-gain
             * node rather than sending microphone
             * audio back to the speakers.
             */

            const silentGain =
              ctx.createGain();

            silentGain.gain.value = 0;

            processor.connect(
              silentGain
            );

            silentGain.connect(
              ctx.destination
            );
          } catch (error) {
            console.error(
              error
            );

            setErrorMsg(
              "Microphone permission failed."
            );

            resetTurn();
          }
        };

        // ---------------------------------------------------
        // WebSocket messages
        // ---------------------------------------------------

        ws.onmessage = async (
          event
        ) => {
          // -----------------------------------------------
          // JSON
          // -----------------------------------------------

          if (
            typeof event.data ===
            "string"
          ) {
            let msg;

            try {
              msg = JSON.parse(
                event.data
              );
            } catch {
              return;
            }

            // ---------------------------------------------
            // Transcript
            // ---------------------------------------------

            if (
              msg.type ===
              "transcript"
            ) {
              setTranscript(
                msg.text || ""
              );

              setCurrentStage(
                "retrieval"
              );

              setStatus(
                "processing"
              );

              cleanupRecording();

              return;
            }

            // ---------------------------------------------
            // Final response
            // ---------------------------------------------

            if (
              msg.type ===
              "final"
            ) {
              setTimings(
                msg.timings || null
              );

              /*
               * Answer appears immediately and
               * types in ~200ms.
               */

              animateAnswer(
                msg.answer_text || ""
              );

              setCurrentStage(
                "complete"
              );

              /*
               * If audio has already completed,
               * allow user to read it.
               */

              setStatus(
                "speaking"
              );

              return;
            }

            // ---------------------------------------------
            // Error
            // ---------------------------------------------

            if (
              msg.type ===
              "error"
            ) {
              setErrorMsg(
                msg.detail ||
                msg.error ||
                "Something went wrong."
              );

              setCurrentStage(
                null
              );

              setStatus(
                "error"
              );

              cleanupRecording();

              return;
            }
          }

          // -----------------------------------------------
          // Binary audio
          // -----------------------------------------------

          else {
            await playChunk(
              event.data
            );
          }
        };

        // ---------------------------------------------------
        // Close
        // ---------------------------------------------------

        ws.onclose = () => {
          cleanupRecording();

          setStatus(
            (current) => {
              if (
                current ===
                "recording" ||
                current ===
                "processing"
              ) {
                return "idle";
              }

              return current;
            }
          );
        };

        // ---------------------------------------------------
        // Error
        // ---------------------------------------------------

        ws.onerror = () => {
          setErrorMsg(
            `Connection error. Is the backend running on ${WS_BASE}?`
          );

          cleanupRecording();

          setStatus(
            "error"
          );
        };
      } catch (error) {
        console.error(
          error
        );

        setErrorMsg(
          "Unable to start voice assistant."
        );

        resetTurn();
      }
    },
    [
      animateAnswer,
      cleanupRecording,
      ensureRecordingCtx,
      playChunk,
      resetTurn,
    ]
  );

  // ---------------------------------------------------------
  // Stop recording
  // ---------------------------------------------------------

  const stopRecording =
    useCallback(() => {
      if (
        wsRef.current &&
        wsRef.current.readyState ===
        WebSocket.OPEN
      ) {
        wsRef.current.send(
          JSON.stringify({
            event: "stop",
          })
        );
      }

      cleanupRecording();

      setCurrentStage(
        "processing"
      );

      setStatus(
        "processing"
      );
    }, [cleanupRecording]);

  // ---------------------------------------------------------
  // Pause / Resume audio
  // ---------------------------------------------------------

  const togglePause =
    useCallback(() => {
      if (
        playbackSourcesRef.current
          .size === 0
      ) {
        return;
      }

      if (!isPaused) {
        stopPlayback();

        setIsPaused(true);
        setStatus("paused");
      } else {
        /*
         * Already-sent chunks cannot be reconstructed
         * here. The user has chosen to stop listening,
         * so we keep the answer visible and don't restart
         * old audio.
         */

        setIsPaused(false);
        setStatus("speaking");
      }
    }, [
      isPaused,
      stopPlayback,
    ]);

  // ---------------------------------------------------------
  // Button
  // ---------------------------------------------------------

  const handleButtonClick =
    () => {
      if (
        status === "idle" ||
        status === "error"
      ) {
        startRecording();
      } else if (
        status === "recording"
      ) {
        stopRecording();
      } else if (
        status === "speaking" ||
        status === "paused"
      ) {
        togglePause();
      }
    };

  // ---------------------------------------------------------
  // Cleanup
  // ---------------------------------------------------------

  useEffect(() => {
    return () => {
      resetTurn();

      if (
        recordingCtxRef.current
      ) {
        try {
          recordingCtxRef.current.close();
        } catch { }
      }

      if (
        playbackCtxRef.current
      ) {
        try {
          playbackCtxRef.current.close();
        } catch { }
      }
    };
  }, [resetTurn]);

  // ---------------------------------------------------------
  // Timing display
  // ---------------------------------------------------------

  const timingItems =
    useMemo(() => {
      if (!timings) {
        return [];
      }

      const labels = {
        stt_ms: "Speech recognition",
        guardrail_check_ms:
          "Guardrails",
        guardrails_ms:
          "Guardrails",

        retrieval_wall_ms:
          "Retrieval",
        retrieval_total_ms:
          "Retrieval engine",
        retrieval_embedding_ms:
          "Embedding",
        retrieval_qdrant_ms:
          "Qdrant",
        retrieval_fusion_ms:
          "Fusion",

        text_trim_ms:
          "Text processing",
        groq_call_ms:
          "Groq generation",

        tts_ms:
          "Voice generation",
        tts_generation_ms:
          "Voice generation",

        tts_first_chunk_ms:
          "First audio",

        total_turn_ms:
          "Total",

        stt_to_tts_gap_ms:
          "STT → TTS",
      };

      return Object.entries(
        timings
      )
        .filter(
          ([key, value]) =>
            typeof value ===
            "number" &&
            Number.isFinite(value)
        )
        .map(
          ([key, value]) => ({
            key,
            label:
              labels[key] ||
              formatTimingKey(key),
            value,
          })
        );
    }, [timings]);

  // ---------------------------------------------------------
  // UI text
  // ---------------------------------------------------------

  const statusText = {
    idle: "Tap to speak",
    recording: "Listening",
    processing: "Finding your answer",
    speaking: "Answer ready",
    paused: "Audio paused",
    error: "Try again",
  }[status];

  return (
    <div style={styles.container}>
      {/* =====================================================
          HERO
      ====================================================== */}

      <div style={styles.hero}>
        <div
          style={{
            ...styles.orb,
            ...(status ===
              "recording"
              ? styles.orbRecording
              : {}),
            ...(status ===
              "processing"
              ? styles.orbProcessing
              : {}),
            ...(status ===
              "speaking"
              ? styles.orbSpeaking
              : {}),
          }}
        >
          <button
            onClick={
              handleButtonClick
            }
            style={
              styles.mainButton
            }
            aria-label={
              statusText
            }
          >
            <MicIcon
              status={status}
            />
          </button>
        </div>

        <div style={styles.statusTitle}>
          {statusText}
        </div>

        <div style={styles.statusSubtitle}>
          {status ===
            "recording"
            ? "Speak naturally, then tap stop"
            : status ===
              "processing"
              ? "Searching the knowledge base…"
              : status ===
                "speaking"
                ? "You can pause the voice and read"
                : status ===
                  "paused"
                  ? "Audio paused · answer remains visible"
                  : "Ask anything from the knowledge base"}
        </div>
      </div>

      {/* =====================================================
          LIVE PIPELINE
      ====================================================== */}

      {status ===
        "processing" && (
          <div style={styles.pipeline}>
            <Stage
              label="Speech"
              active={
                currentStage ===
                "stt"
              }
              done={
                Boolean(
                  transcript
                )
              }
            />

            <StageLine />

            <Stage
              label="Search"
              active={
                currentStage ===
                "retrieval"
              }
              done={false}
            />

            <StageLine />

            <Stage
              label="Answer"
              active={false}
              done={false}
            />

            <StageLine />

            <Stage
              label="Voice"
              active={false}
              done={false}
            />
          </div>
        )}

      {/* =====================================================
          ERROR
      ====================================================== */}

      {errorMsg && (
        <div style={styles.errorCard}>
          <div style={styles.errorIcon}>
            !
          </div>

          <div>
            <div
              style={
                styles.errorTitle
              }
            >
              Something went wrong
            </div>

            <div
              style={
                styles.errorText
              }
            >
              {errorMsg}
            </div>
          </div>
        </div>
      )}

      {/* =====================================================
          RESULT
      ====================================================== */}

      {(transcript ||
        answerText) && (
          <div
            style={
              styles.resultCard
            }
          >
            {/* User question */}

            {transcript && (
              <div
                style={
                  styles.questionSection
                }
              >
                <div
                  style={
                    styles.sectionLabel
                  }
                >
                  YOU ASKED
                </div>

                <div
                  style={
                    styles.questionText
                  }
                >
                  {transcript}
                </div>
              </div>
            )}

            {/* Divider */}

            {answerText && (
              <>
                <div
                  style={
                    styles.divider
                  }
                />

                <div
                  style={
                    styles.answerSection
                  }
                >
                  <div
                    style={
                      styles.answerHeader
                    }
                  >
                    <div
                      style={
                        styles.sectionLabel
                      }
                    >
                      ANSWER
                    </div>

                    {(status ===
                      "speaking" ||
                      status ===
                      "paused") && (
                        <button
                          onClick={
                            togglePause
                          }
                          style={
                            styles.pauseButton
                          }
                        >
                          <PauseIcon
                            paused={
                              isPaused
                            }
                          />

                          {isPaused
                            ? "Resume"
                            : "Pause"}
                        </button>
                      )}
                  </div>

                  <div
                    style={
                      styles.answerText
                    }
                  >
                    {answerText}
                  </div>
                </div>
              </>
            )}

            {/* =================================================
              TIMINGS
          ================================================== */}

            {timings && (
              <div
                style={
                  styles.timingPanel
                }
              >
                <div
                  style={
                    styles.timingHeader
                  }
                >
                  <span>
                    Performance
                  </span>

                  {timings.total_turn_ms !=
                    null && (
                      <strong>
                        {formatMs(
                          timings.total_turn_ms
                        )}
                      </strong>
                    )}
                </div>

                <div
                  style={
                    styles.timingGrid
                  }
                >
                  {timingItems.map(
                    (item) => (
                      <div
                        key={
                          item.key
                        }
                        style={
                          styles.timingItem
                        }
                      >
                        <span>
                          {item.label}
                        </span>

                        <strong>
                          {formatMs(
                            item.value
                          )}
                        </strong>
                      </div>
                    )
                  )}
                </div>
              </div>
            )}
          </div>
        )}

      {/* =====================================================
          RESET
      ====================================================== */}

      {(answerText ||
        transcript ||
        status ===
        "paused") && (
          <button
            onClick={resetTurn}
            style={
              styles.resetButton
            }
          >
            New question
          </button>
        )}
    </div>
  );
}

// =============================================================
// Stage
// =============================================================

function Stage({
  label,
  active,
  done,
}) {
  return (
    <div
      style={{
        ...styles.stage,
        ...(active
          ? styles.stageActive
          : {}),
      }}
    >
      <div
        style={{
          ...styles.stageDot,
          ...(done
            ? styles.stageDone
            : {}),
          ...(active
            ? styles.stageActiveDot
            : {}),
        }}
      >
        {done ? "✓" : ""}
      </div>

      <span>
        {label}
      </span>
    </div>
  );
}

function StageLine() {
  return (
    <div
      style={
        styles.stageLine
      }
    />
  );
}

// =============================================================
// Icons
// =============================================================

function MicIcon({
  status,
}) {
  if (
    status ===
    "recording"
  ) {
    return (
      <svg
        width="28"
        height="28"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
      >
        <rect
          x="6"
          y="6"
          width="12"
          height="12"
          rx="2"
        />
      </svg>
    );
  }

  return (
    <svg
      width="28"
      height="28"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
    >
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
      <line
        x1="12"
        y1="19"
        x2="12"
        y2="23"
      />
    </svg>
  );
}

function PauseIcon({
  paused,
}) {
  if (paused) {
    return (
      <svg
        width="14"
        height="14"
        viewBox="0 0 24 24"
        fill="currentColor"
      >
        <path d="M8 5v14l11-7z" />
      </svg>
    );
  }

  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="currentColor"
    >
      <rect
        x="6"
        y="5"
        width="4"
        height="14"
        rx="1"
      />
      <rect
        x="14"
        y="5"
        width="4"
        height="14"
        rx="1"
      />
    </svg>
  );
}

// =============================================================
// Formatting
// =============================================================

function formatMs(value) {
  if (
    value === null ||
    value === undefined ||
    !Number.isFinite(value)
  ) {
    return "—";
  }

  if (value < 1000) {
    return `${Math.round(
      value
    )} ms`;
  }

  return `${(
    value / 1000
  ).toFixed(2)} s`;
}

function formatTimingKey(
  key
) {
  return key
    .replace(/_ms$/, "")
    .replace(/_/g, " ")
    .replace(
      /\b\w/g,
      (char) =>
        char.toUpperCase()
    );
}

// =============================================================
// Styles
// =============================================================

const styles = {
  container: {
    width: "100%",
    maxWidth: 820,
    margin: "0 auto",
    padding:
      "clamp(20px, 4vw, 48px) clamp(12px, 3vw, 32px) 60px",
    boxSizing: "border-box",
    fontFamily:
      "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    color: "#111827",
  },

  // -----------------------------------------------------------
  // Hero
  // -----------------------------------------------------------

  hero: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    textAlign: "center",
    marginBottom: 32,
  },

  orb: {
    width: 108,
    height: 108,
    borderRadius: "50%",
    padding: 8,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background:
      "linear-gradient(135deg, #eef2ff, #f5f3ff)",
    boxShadow:
      "0 12px 40px rgba(79,70,229,.12)",
    transition:
      "transform .2s ease, box-shadow .2s ease",
  },

  orbRecording: {
    transform: "scale(1.05)",
    boxShadow:
      "0 12px 45px rgba(220,38,38,.28)",
  },

  orbProcessing: {
    boxShadow:
      "0 12px 45px rgba(79,70,229,.22)",
  },

  orbSpeaking: {
    boxShadow:
      "0 12px 45px rgba(37,99,235,.22)",
  },

  mainButton: {
    width: 92,
    height: 92,
    borderRadius: "50%",
    border: "none",
    background:
      "#111827",
    color: "#fff",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    cursor: "pointer",
    boxShadow:
      "0 8px 25px rgba(17,24,39,.22)",
    transition:
      "transform .18s ease, background .18s ease",
  },

  statusTitle: {
    marginTop: 18,
    fontSize: 18,
    fontWeight: 650,
    letterSpacing:
      "-0.02em",
  },

  statusSubtitle: {
    marginTop: 6,
    color: "#9ca3af",
    fontSize: 13,
  },

  // -----------------------------------------------------------
  // Pipeline
  // -----------------------------------------------------------

  pipeline: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    margin:
      "0 auto 24px",
    padding:
      "12px 16px",
    border:
      "1px solid #eef0f4",
    borderRadius: 999,
    background:
      "rgba(255,255,255,.8)",
    boxShadow:
      "0 6px 24px rgba(17,24,39,.05)",
  },

  stage: {
    display: "flex",
    alignItems: "center",
    gap: 6,
    fontSize: 11,
    color: "#9ca3af",
    whiteSpace: "nowrap",
  },

  stageActive: {
    color: "#4f46e5",
    fontWeight: 600,
  },

  stageDot: {
    width: 16,
    height: 16,
    borderRadius: "50%",
    border:
      "1.5px solid #d1d5db",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: 9,
  },

  stageDone: {
    background: "#111827",
    color: "#fff",
    borderColor:
      "#111827",
  },

  stageActiveDot: {
    borderColor:
      "#4f46e5",
    boxShadow:
      "0 0 0 3px rgba(79,70,229,.1)",
  },

  stageLine: {
    width: 18,
    height: 1,
    background:
      "#e5e7eb",
  },

  // -----------------------------------------------------------
  // Result
  // -----------------------------------------------------------

  resultCard: {
    background: "#fff",
    border:
      "1px solid #edf0f4",
    borderRadius: 20,
    overflow: "hidden",
    boxShadow:
      "0 12px 40px rgba(17,24,39,.055)",
  },

  questionSection: {
    padding:
      "20px 22px 18px",
  },

  sectionLabel: {
    fontSize: 10,
    fontWeight: 700,
    letterSpacing:
      ".12em",
    color: "#9ca3af",
    marginBottom: 7,
  },

  questionText: {
    fontSize: 15,
    lineHeight: 1.55,
    color: "#374151",
  },

  divider: {
    height: 1,
    background:
      "#f0f1f3",
  },

  answerSection: {
    padding:
      "20px 22px 22px",
  },

  answerHeader: {
    display: "flex",
    alignItems: "center",
    justifyContent:
      "space-between",
    gap: 12,
  },

  answerText: {
    marginTop: 10,
    fontSize: 16,
    lineHeight: 1.75,
    color: "#1f2937",
    whiteSpace:
      "pre-wrap",
  },

  // -----------------------------------------------------------
  // Pause
  // -----------------------------------------------------------

  pauseButton: {
    border:
      "1px solid #e5e7eb",
    background: "#fff",
    color: "#374151",
    borderRadius: 999,
    padding:
      "7px 11px",
    display: "flex",
    alignItems: "center",
    gap: 6,
    fontSize: 11,
    fontWeight: 600,
    cursor: "pointer",
    transition:
      "background .15s ease",
  },

  // -----------------------------------------------------------
  // Timings
  // -----------------------------------------------------------

  timingPanel: {
    borderTop:
      "1px solid #f0f1f3",
    background:
      "#fafafa",
    padding:
      "15px 22px 18px",
  },

  timingHeader: {
    display: "flex",
    justifyContent:
      "space-between",
    alignItems: "center",
    color: "#6b7280",
    fontSize: 11,
    marginBottom: 11,
  },

  timingGrid: {
    display: "grid",
    gridTemplateColumns:
      "repeat(auto-fit, minmax(150px, 1fr))",
    gap: 8,
  },

  timingItem: {
    background: "#fff",
    border:
      "1px solid #eef0f3",
    borderRadius: 9,
    padding:
      "9px 10px",
    display: "flex",
    flexDirection:
      "column",
    gap: 3,
  },

  // -----------------------------------------------------------
  // Error
  // -----------------------------------------------------------

  errorCard: {
    display: "flex",
    gap: 12,
    alignItems: "flex-start",
    padding: 14,
    marginBottom: 20,
    borderRadius: 14,
    background: "#fff7f7",
    border:
      "1px solid #fee2e2",
  },

  errorIcon: {
    width: 24,
    height: 24,
    borderRadius: "50%",
    background: "#dc2626",
    color: "#fff",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontWeight: 700,
    fontSize: 12,
  },

  errorTitle: {
    fontSize: 13,
    fontWeight: 650,
    color: "#991b1b",
  },

  errorText: {
    marginTop: 3,
    fontSize: 12,
    color: "#b91c1c",
  },

  // -----------------------------------------------------------
  // Reset
  // -----------------------------------------------------------

  resetButton: {
    marginTop: 16,
    border: "none",
    background: "transparent",
    color: "#6b7280",
    fontSize: 12,
    cursor: "pointer",
    padding: 8,
  },
};