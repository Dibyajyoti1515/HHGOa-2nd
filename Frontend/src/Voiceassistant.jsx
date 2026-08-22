// frontend/src/Voiceassistant.jsx
//
// HH GOA — Ask Anything
// Central card: Text / Voice toggle, pipeline status strip, conversation history.
//
// Voice mode talks to the REAL backend contract:
//   WS   /v1/voice   — mic sends raw PCM16 @ 16kHz frames, {event:"stop"} ends the turn
//                       server replies with {type:"transcript"}, {type:"translation"},
//                       {type:"answer"}, {type:"final"}, {type:"error"}
//                       plus binary PCM16 @ 24kHz audio chunks (Sarvam Bulbul v3), streamed live.
// Text mode talks to:
//   POST /v1/text     { text, language_code } -> {
//                        type: "text_response" | "error",
//                        text, retrieval_query, answer, language_code,
//                        generation_method, llm_used, top1_score,
//                        results_count, timings
//                      }
//                      (this endpoint returns text only — no audio_url)

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
const WS_BASE = import.meta.env.VITE_WS_BASE || "ws://localhost:8000";
const WS_URL = `${WS_BASE}/v1/voice`;

const INPUT_SAMPLE_RATE = 16000; // browser mic -> STT
const OUTPUT_SAMPLE_RATE = 24000; // Sarvam Bulbul v3 -> browser

const STAGES = ["Listening", "Transcribing", "Retrieving", "Generating", "Speaking"];
const STAGE_INDEX = STAGES.reduce((acc, s, i) => ({ ...acc, [s]: i }), {});

// Order + labels for the per-turn latency breakdown. Anything present in the
// backend `timings` payload that matches a key here gets a friendly label;
// everything else is skipped so we never render noisy/duplicate fields.
const TIMING_LABELS = [
  // shared / voice (ws_voice.py, POST /v1/query)
  ["stt_ms", "Transcription"],
  ["guardrail_ms", "Guardrails"],
  ["guardrails", "Guardrails"],
  ["translation_to_english_ms", "Translate → EN"],
  ["retrieval_wall_ms", "Retrieval"],
  ["embedding_ms", "· Embedding"],
  ["qdrant_ms", "· Vector search"],
  ["fusion_ms", "· Fusion"],
  ["retrieval_engine_ms", "· Retrieval engine"],
  ["retrieval_total_ms", "· Retrieval engine"],
  ["groq_ms", "Generation (Groq)"],
  ["groq_call_ms", "Generation (Groq)"],
  ["text_trim_ms", "Answer trim"],
  ["translation_to_user_language_ms", "Translate → you"],
  ["answer_translation_ms", "Translate → you"],
  ["tts_ms", "Speech synthesis"],
  ["tts_generation_ms", "Speech synthesis"],
  ["tts_first_chunk_ms", "· First audio chunk"],
  ["total_turn_ms", "Total"],
  ["total_ms", "Total"],
];

let turnId = 0;
const nextId = () => `turn-${Date.now()}-${turnId++}`;

// ---------------------------------------------------------------------------
// PCM helpers (also used to stream an uploaded file through the same socket)
// ---------------------------------------------------------------------------

function downsampleBuffer(buffer, inputSampleRate, targetSampleRate) {
  if (targetSampleRate === inputSampleRate) return buffer;
  const ratio = inputSampleRate / targetSampleRate;
  const newLength = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;
  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
    let accum = 0;
    let count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
      accum += buffer[i];
      count++;
    }
    result[offsetResult] = count > 0 ? accum / count : 0;
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }
  return result;
}

function floatTo16BitPCM(float32Array) {
  const out = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

function formatMs(value) {
  if (value === null || value === undefined || !Number.isFinite(value)) return null;
  return value < 1000 ? `${Math.round(value)}ms` : `${(value / 1000).toFixed(2)}s`;
}

function buildTimingItems(timings) {
  if (!timings || typeof timings !== "object") return [];
  return TIMING_LABELS
    .map(([key, label]) => {
      const raw = timings[key];
      const formatted = formatMs(typeof raw === "number" ? raw : null);
      const isTotal = key === "total_turn_ms" || key === "total_ms";
      return formatted ? { key, label, value: formatted, isTotal } : null;
    })
    .filter(Boolean);
}

// ---------------------------------------------------------------------------
// Small presentational pieces
// ---------------------------------------------------------------------------

function ModeToggle({ mode, onChange, disabled }) {
  return (
    <div className="relative grid grid-cols-2 rounded-full border border-white/10 bg-white/[0.04] p-1 shadow-[inset_0_1px_0_rgba(255,255,255,0.06)] backdrop-blur-xl">
      <span
        className={`absolute inset-y-1 w-[calc(50%-4px)] rounded-full bg-gradient-to-r from-[#e8c547] to-[#f2d876] shadow-[0_2px_14px_rgba(232,197,71,0.4)] transition-transform duration-300 ease-out ${mode === "voice" ? "translate-x-[calc(100%+8px)]" : "translate-x-0"
          }`}
      />
      {["text", "voice"].map((m) => (
        <button
          key={m}
          type="button"
          disabled={disabled}
          onClick={() => onChange(m)}
          className={`relative z-10 rounded-full px-4 py-2 font-mono-hh text-xs uppercase tracking-[0.15em] transition-colors duration-300 disabled:cursor-not-allowed disabled:opacity-50 ${mode === m ? "text-[#0b2e28]" : "text-[#f5f0e1]/60 hover:text-[#f5f0e1]"
            }`}
        >
          {m === "text" ? "Text" : "Voice"}
        </button>
      ))}
    </div>
  );
}

function PipelineStrip({ stage, busy }) {
  const activeIndex = STAGES.indexOf(stage);
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.04] px-3 py-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.06)] backdrop-blur-xl sm:px-5">
      <div className="hh-scrollbar-hidden flex items-center justify-between gap-1.5 overflow-x-auto sm:gap-3">
        {STAGES.map((s, i) => {
          const isActive = i === activeIndex;
          const isDone = activeIndex > i;
          return (
            <div key={s} className="flex flex-1 items-center last:flex-none">
              <div className="flex flex-col items-center gap-2 px-0.5 sm:px-1">
                <span className="relative flex h-3 w-3 items-center justify-center">
                  {isActive && busy && (
                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-[#e8c547]/60" />
                  )}
                  <span
                    className={`relative h-1.5 w-1.5 rounded-full transition-all duration-300 ${isActive
                        ? "scale-150 bg-[#e8c547] shadow-[0_0_10px_3px_rgba(232,197,71,0.55)]"
                        : isDone
                          ? "bg-[#f5f0e1]/50"
                          : "bg-white/15"
                      }`}
                  />
                </span>
                <span
                  className={`whitespace-nowrap font-mono-hh text-[9px] uppercase tracking-[0.18em] transition-colors duration-300 sm:text-[10px] ${isActive ? "font-bold text-[#e8c547]" : isDone ? "text-[#f5f0e1]/50" : "text-white/25"
                    }`}
                >
                  {s}
                </span>
              </div>
              {i < STAGES.length - 1 && (
                <div
                  className={`mx-1.5 mb-5 h-px flex-1 transition-colors duration-300 sm:mx-3 ${isDone ? "bg-[#f5f0e1]/30" : "bg-white/10"
                    }`}
                />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function AudioBar({ src, autoPlay, isStreaming, isPlaying }) {
  const audioRef = useRef(null);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (autoPlay && audioRef.current) {
      audioRef.current.play().catch(() => { });
    }
  }, [autoPlay, src]);

  const toggle = () => {
    if (!audioRef.current) return;
    if (playing) audioRef.current.pause();
    else audioRef.current.play().catch(() => { });
  };

  const active = src ? playing : isPlaying;

  return (
    <div className="mt-2 flex items-center gap-2 rounded-xl border border-white/10 bg-white/[0.04] px-3 py-2 backdrop-blur-md">
      {src && (
        <audio
          ref={audioRef}
          src={src}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={() => setPlaying(false)}
          className="hidden"
        />
      )}
      {src ? (
        <button
          type="button"
          onClick={toggle}
          className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-[#e8c547] text-[#0b2e28] transition-transform hover:scale-105 active:scale-95"
          aria-label={playing ? "Pause" : "Play"}
        >
          {playing ? (
            <svg viewBox="0 0 16 16" className="h-3.5 w-3.5 fill-current">
              <rect x="3" y="2" width="3.5" height="12" />
              <rect x="9.5" y="2" width="3.5" height="12" />
            </svg>
          ) : (
            <svg viewBox="0 0 16 16" className="ml-0.5 h-3.5 w-3.5 fill-current">
              <path d="M4 2.5v11l10-5.5z" />
            </svg>
          )}
        </button>
      ) : (
        <span
          className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full ${isStreaming ? "bg-[#e8c547]" : "bg-white/10"
            }`}
        >
          <svg viewBox="0 0 16 16" className="h-3.5 w-3.5 fill-current text-[#0b2e28]">
            <path d="M8 1.5a3 3 0 0 0-3 3v3.5a3 3 0 0 0 6 0V4.5a3 3 0 0 0-3-3Z" />
            <path d="M12.5 7.5a1 1 0 1 0-2 0 2.5 2.5 0 0 1-5 0 1 1 0 1 0-2 0 4.5 4.5 0 0 0 4 4.47V13.5H6a1 1 0 1 0 0 2h4a1 1 0 1 0 0-2H8.5v-1.53a4.5 4.5 0 0 0 4-4.47Z" />
          </svg>
        </span>
      )}
      <div className="flex h-5 flex-1 items-end gap-[3px]">
        {Array.from({ length: 24 }).map((_, i) => (
          <span
            key={i}
            className="w-[3px] flex-1 rounded-full"
            style={{
              height: `${20 + Math.abs(Math.sin(i * 0.9)) * 80}%`,
              animation: active ? `hh-eq 0.9s ease-in-out ${i * 0.04}s infinite alternate` : "none",
              backgroundColor: "#f5f0e1",
              opacity: active ? 0.8 : 0.3,
            }}
          />
        ))}
      </div>
      {!src && (
        <span className="shrink-0 font-mono-hh text-[9px} uppercase tracking-[0.1em] text-[#f5f0e1]/40">
          {isStreaming ? "live" : "done"}
        </span>
      )}
    </div>
  );
}

function TimingPanel({ items }) {
  const [open, setOpen] = useState(false);
  if (!items.length) return null;
  const total = items.find((i) => i.isTotal);
  const rest = items.filter((i) => !i.isTotal);

  return (
    <div className="mt-2 w-full rounded-xl border border-white/10 bg-white/[0.04] backdrop-blur-md">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left"
      >
        <span className="flex items-center gap-1.5 font-mono-hh text-[9px] uppercase tracking-[0.15em] text-[#f5f0e1]/50">
          <svg
            viewBox="0 0 16 16"
            className={`h-3 w-3 fill-current transition-transform duration-200 ${open ? "rotate-90" : ""}`}
          >
            <path d="M6 3.5 10.5 8 6 12.5z" />
          </svg>
          Timing
        </span>
        {total && (
          <span className="font-mono-hh text-[10px] font-bold uppercase tracking-[0.1em] text-[#e8c547]">
            {total.value}
          </span>
        )}
      </button>
      {open && (
        <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 border-t border-white/10 px-3 py-2.5">
          {rest.map((item) => (
            <div key={item.key} className="flex items-center justify-between gap-2">
              <span className="truncate font-mono-hh text-[9px] uppercase tracking-[0.1em] text-[#f5f0e1]/40">
                {item.label}
              </span>
              <span className="shrink-0 font-mono-hh text-[10px] text-[#f5f0e1]/80">{item.value}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function MessageBubble({ turn, autoPlay, streamingId, playingId }) {
  const isUser = turn.role === "user";
  const timingItems = useMemo(() => buildTimingItems(turn.timings), [turn.timings]);
  const isStreamingAudio = turn.id === streamingId;
  const isPlayingAudio = turn.id === playingId;

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`flex max-w-[85%] flex-col ${isUser ? "items-end" : "items-start"}`}>
        <span className="mb-1 font-mono-hh text-[9px] uppercase tracking-[0.2em] text-[#f5f0e1]/35">
          {isUser ? "You" : "AI"}
        </span>
        <div
          className={`rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${isUser
              ? "rounded-tr-sm border border-[#e8c547]/25 bg-[#e8c547]/15 text-[#f5f0e1] backdrop-blur-md"
              : "rounded-tl-sm border border-white/10 bg-white/[0.05] text-[#f5f0e1]/90 backdrop-blur-md"
            }`}
        >
          {turn.text || (turn.pending ? <PendingDots /> : "")}
        </div>
        {(turn.audioUrl || turn.hasVoiceAudio) && (
          <AudioBar
            src={turn.audioUrl}
            autoPlay={autoPlay}
            isStreaming={isStreamingAudio}
            isPlaying={isPlayingAudio}
          />
        )}
        <TimingPanel items={timingItems} />
      </div>
    </div>
  );
}

function PendingDots() {
  return (
    <span className="inline-flex items-center gap-1">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="h-1.5 w-1.5 rounded-full bg-[#f5f0e1]/50"
          style={{ animation: `hh-bounce 1s ease-in-out ${i * 0.15}s infinite` }}
        />
      ))}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function VoiceAssistant() {
  const [mode, setMode] = useState("text");
  const [conversation, setConversation] = useState([]);
  const [error, setError] = useState("");
  const [autoPlay, setAutoPlay] = useState(true);

  // text mode
  const [textQuery, setTextQuery] = useState("");
  const [textLoading, setTextLoading] = useState(false);
  const [textStage, setTextStage] = useState(null);

  // voice mode
  const [voiceStatus, setVoiceStatus] = useState("idle"); // idle | recording | processing | speaking | paused | error
  const [voiceStage, setVoiceStage] = useState(null);
  const [isPaused, setIsPaused] = useState(false);
  const [streamingAudioId, setStreamingAudioId] = useState(null);
  const [playingAudioId, setPlayingAudioId] = useState(null);

  const wsRef = useRef(null);
  const recordingCtxRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const processorRef = useRef(null);
  const sourceRef = useRef(null);

  const playbackCtxRef = useRef(null);
  const playbackSourcesRef = useRef(new Set());
  const playbackTimeRef = useRef(0);
  const audioStartedRef = useRef(false);
  const turnFinishedRef = useRef(false); // true once server has sent "final" (all audio already streamed)

  const currentAiTurnIdRef = useRef(null);
  const maxStageRef = useRef(-1); // guards the pipeline strip from ever going "backwards"
  const fileInputRef = useRef(null);

  const conversationEndRef = useRef(null);
  useEffect(() => {
    conversationEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [conversation]);

  const isRecording = voiceStatus === "recording";
  const isVoiceBusy = voiceStatus === "recording" || voiceStatus === "processing";

  // Forward-only stage setter: a turn's pipeline should only ever move
  // Listening -> Transcribing -> Retrieving -> Generating -> Speaking,
  // never jump backwards because a later message arrived out of order.
  const advanceVoiceStage = useCallback((stage) => {
    const idx = STAGE_INDEX[stage];
    if (idx === undefined) return;
    if (idx >= maxStageRef.current) {
      maxStageRef.current = idx;
      setVoiceStage(stage);
    }
  }, []);

  const resetVoiceStage = useCallback((stage) => {
    maxStageRef.current = STAGE_INDEX[stage] ?? -1;
    setVoiceStage(stage);
  }, []);

  // --- cleanup helpers -------------------------------------------------------

  const cleanupRecording = useCallback(() => {
    try {
      processorRef.current?.disconnect();
    } catch { }
    try {
      sourceRef.current?.disconnect();
    } catch { }
    mediaStreamRef.current?.getTracks().forEach((t) => {
      try {
        t.stop();
      } catch { }
    });
    processorRef.current = null;
    sourceRef.current = null;
    mediaStreamRef.current = null;
  }, []);

  const stopPlayback = useCallback(() => {
    playbackSourcesRef.current.forEach((source) => {
      try {
        source.stop();
      } catch { }
    });
    playbackSourcesRef.current.clear();
    playbackTimeRef.current = playbackCtxRef.current ? playbackCtxRef.current.currentTime : 0;
    setPlayingAudioId(null);
  }, []);

  const ensureRecordingCtx = useCallback(() => {
    if (!recordingCtxRef.current) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      recordingCtxRef.current = new AudioCtx({ sampleRate: INPUT_SAMPLE_RATE });
    }
    return recordingCtxRef.current;
  }, []);

  const ensurePlaybackCtx = useCallback(() => {
    if (!playbackCtxRef.current) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      playbackCtxRef.current = new AudioCtx({ sampleRate: OUTPUT_SAMPLE_RATE });
    }
    return playbackCtxRef.current;
  }, []);

  // --- streamed PCM playback (voice mode) -------------------------------------

  const playChunk = useCallback(
    async (arrayBuffer) => {
      const ctx = ensurePlaybackCtx();
      if (ctx.state === "suspended") {
        try {
          await ctx.resume();
        } catch { }
      }

      const int16 = new Int16Array(arrayBuffer);
      if (!int16.length) return;

      const float32 = new Float32Array(int16.length);
      for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;

      const audioBuffer = ctx.createBuffer(1, float32.length, OUTPUT_SAMPLE_RATE);
      audioBuffer.copyToChannel(float32, 0);

      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(ctx.destination);

      const startAt = Math.max(ctx.currentTime, playbackTimeRef.current);
      source.start(startAt);
      playbackTimeRef.current = startAt + audioBuffer.duration;

      playbackSourcesRef.current.add(source);
      source.onended = () => {
        playbackSourcesRef.current.delete(source);
        if (playbackSourcesRef.current.size === 0) {
          setPlayingAudioId((current) => (current === currentAiTurnIdRef.current ? null : current));
          // Only the server's "final" message tells us no more chunks are
          // coming — once that's true and playback has actually drained,
          // the turn is genuinely over: drop the "Speaking" state and hand
          // control back to the mic so it's ready for the next question.
          if (turnFinishedRef.current) {
            setStreamingAudioId((current) => (current === currentAiTurnIdRef.current ? null : current));
            setVoiceStatus((current) => (current === "speaking" ? "idle" : current));
            setIsPaused(false);
          }
        }
      };

      // First chunk of a turn: mark this turn as the one actively streaming/playing.
      if (!audioStartedRef.current) {
        audioStartedRef.current = true;
        advanceVoiceStage("Speaking");
        setStreamingAudioId(currentAiTurnIdRef.current);
      }
      setPlayingAudioId(currentAiTurnIdRef.current);
      setVoiceStatus("speaking");
    },
    [ensurePlaybackCtx, advanceVoiceStage],
  );

  // --- shared WS message handling --------------------------------------------

  const attachSocketHandlers = useCallback(
    (ws) => {
      ws.onmessage = async (event) => {
        if (typeof event.data === "string") {
          let msg;
          try {
            msg = JSON.parse(event.data);
          } catch {
            return;
          }

          if (msg.type === "transcript") {
            const turn = { id: nextId(), role: "user", text: msg.text || "" };
            setConversation((prev) => [...prev, turn]);
            advanceVoiceStage("Retrieving");
            setVoiceStatus("processing");
            cleanupRecording();
            return;
          }

          if (msg.type === "translation") {
            advanceVoiceStage("Retrieving");
            return;
          }

          // Server sends the answer text BEFORE audio starts streaming —
          // show it right away as a "running" placeholder turn, then attach
          // audio + timings once they arrive.
          if (msg.type === "answer") {
            advanceVoiceStage("Generating");
            turnFinishedRef.current = false;
            const id = nextId();
            currentAiTurnIdRef.current = id;
            setConversation((prev) => [
              ...prev,
              { id, role: "ai", text: msg.text || "", hasVoiceAudio: true, pending: false },
            ]);
            return;
          }

          if (msg.type === "final") {
            turnFinishedRef.current = true;
            // Audio may have already fully drained by the time "final"
            // lands (fast/short answers) — in that case there's nothing
            // left to wait on, so close the turn out right here too.
            if (playbackSourcesRef.current.size === 0) {
              setStreamingAudioId((current) => (current === currentAiTurnIdRef.current ? null : current));
              setVoiceStatus((current) => (current === "speaking" ? "idle" : current));
              setIsPaused(false);
            }
            const id = currentAiTurnIdRef.current;
            setConversation((prev) =>
              prev.map((t) =>
                t.id === id
                  ? { ...t, text: msg.answer_text ?? t.text, timings: msg.timings, hasVoiceAudio: true }
                  : t,
              ),
            );
            return;
          }

          if (msg.type === "error") {
            setError(msg.detail || msg.error || "Something went wrong.");
            setVoiceStage(null);
            setVoiceStatus("error");
            cleanupRecording();
            return;
          }
        } else {
          await playChunk(event.data);
        }
      };

      ws.onclose = () => {
        cleanupRecording();
        setStreamingAudioId((current) => (current === currentAiTurnIdRef.current ? null : current));
        setVoiceStatus((current) => (current === "recording" || current === "processing" ? "idle" : current));
      };

      ws.onerror = () => {
        setError(`Connection error. Is the backend running on ${WS_BASE}?`);
        cleanupRecording();
        setVoiceStatus("error");
      };
    },
    [cleanupRecording, playChunk, advanceVoiceStage],
  );

  // --- start / stop recording --------------------------------------------------

  const startRecording = useCallback(async () => {
    try {
      setError("");
      resetVoiceStage("Listening");
      setIsPaused(false);
      audioStartedRef.current = false;
      turnFinishedRef.current = false;
      playbackTimeRef.current = 0;
      currentAiTurnIdRef.current = null;
      setStreamingAudioId(null);
      setPlayingAudioId(null);

      const ws = new WebSocket(WS_URL);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;
      attachSocketHandlers(ws);

      ws.onopen = async () => {
        try {
          setVoiceStatus("recording");

          const stream = await navigator.mediaDevices.getUserMedia({
            audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
          });
          mediaStreamRef.current = stream;

          const ctx = ensureRecordingCtx();
          if (ctx.state === "suspended") await ctx.resume();

          const source = ctx.createMediaStreamSource(stream);
          sourceRef.current = source;

          const processor = ctx.createScriptProcessor(4096, 1, 1);
          processorRef.current = processor;

          processor.onaudioprocess = (event) => {
            if (ws.readyState !== WebSocket.OPEN) return;
            const input = event.inputBuffer.getChannelData(0);
            const int16 = new Int16Array(input.length);
            for (let i = 0; i < input.length; i++) {
              const sample = Math.max(-1, Math.min(1, input[i]));
              int16[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
            }
            ws.send(int16.buffer);
          };

          source.connect(processor);

          // Route through a silent gain so Chrome keeps the ScriptProcessor
          // running without echoing mic audio back to the speakers.
          const silentGain = ctx.createGain();
          silentGain.gain.value = 0;
          processor.connect(silentGain);
          silentGain.connect(ctx.destination);
        } catch {
          setError("Microphone permission failed.");
          cleanupRecording();
          setVoiceStatus("error");
        }
      };
    } catch {
      setError("Unable to start voice assistant.");
      setVoiceStatus("error");
    }
  }, [attachSocketHandlers, cleanupRecording, ensureRecordingCtx, resetVoiceStage]);

  const stopRecording = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ event: "stop" }));
    }
    cleanupRecording();
    advanceVoiceStage("Transcribing");
    setVoiceStatus("processing");
  }, [cleanupRecording, advanceVoiceStage]);

  const handleMicTap = () => {
    if (isRecording) stopRecording();
    else if (!isVoiceBusy) startRecording();
  };

  const togglePause = useCallback(() => {
    if (playbackSourcesRef.current.size === 0 && !isPaused) return;
    if (!isPaused) {
      stopPlayback();
      setIsPaused(true);
    } else {
      setIsPaused(false);
    }
  }, [isPaused, stopPlayback]);

  // --- upload-a-file fallback (voice mode) -------------------------------------

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;

    setError("");
    try {
      const arrayBuffer = await file.arrayBuffer();
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const decodeCtx = new AudioCtx();
      const decoded = await decodeCtx.decodeAudioData(arrayBuffer.slice(0));
      const mono = decoded.getChannelData(0);
      const downsampled = downsampleBuffer(mono, decoded.sampleRate, INPUT_SAMPLE_RATE);
      const pcm16 = floatTo16BitPCM(downsampled);
      decodeCtx.close();

      audioStartedRef.current = false;
      turnFinishedRef.current = false;
      playbackTimeRef.current = 0;
      currentAiTurnIdRef.current = null;
      setStreamingAudioId(null);
      setPlayingAudioId(null);
      resetVoiceStage("Transcribing");
      setVoiceStatus("processing");

      const ws = new WebSocket(WS_URL);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;
      attachSocketHandlers(ws);

      ws.onopen = () => {
        const chunkSize = 4096;
        for (let i = 0; i < pcm16.length; i += chunkSize) {
          ws.send(pcm16.slice(i, i + chunkSize).buffer);
        }
        ws.send(JSON.stringify({ event: "stop" }));
      };
    } catch {
      setError("Could not read the uploaded audio file.");
      setVoiceStatus("error");
    }
  };

  // --- text mode ---------------------------------------------------------------

  const handleTextSubmit = async (e) => {
    e.preventDefault();
    const query = textQuery.trim();
    if (!query || textLoading) return;

    setError("");
    setTextLoading(true);
    setTextStage("Retrieving");
    setConversation((prev) => [...prev, { id: nextId(), role: "user", text: query }]);
    setTextQuery("");

    try {
      const res = await fetch(`${API_BASE}/v1/text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: query, language_code: "en-IN" }),
      });
      if (!res.ok) throw new Error(`Request failed (${res.status})`);
      setTextStage("Generating");
      const data = await res.json();

      if (data.type === "error") {
        setError(data.detail || data.error || "Something went wrong.");
        setConversation((prev) => [
          ...prev,
          { id: nextId(), role: "ai", text: `⚠ ${data.detail || data.error || "Request failed."}`, timings: data.timings },
        ]);
        return;
      }

      setConversation((prev) => [
        ...prev,
        {
          id: nextId(),
          role: "ai",
          text: data.answer ?? "(no answer returned)",
          timings: data.timings,
        },
      ]);
    } catch (err) {
      setError(err.message === "Failed to fetch" ? "Failed to fetch — check the backend connection." : err.message);
    } finally {
      setTextLoading(false);
      setTimeout(() => setTextStage(null), 1200);
    }
  };

  // --- cleanup on unmount --------------------------------------------------------

  useEffect(() => {
    return () => {
      cleanupRecording();
      stopPlayback();
      try {
        wsRef.current?.close();
      } catch { }
      try {
        recordingCtxRef.current?.close();
      } catch { }
      try {
        playbackCtxRef.current?.close();
      } catch { }
    };
  }, [cleanupRecording, stopPlayback]);

  const stage = mode === "text" ? textStage : voiceStage;
  const stageBusy = mode === "text" ? textLoading : isVoiceBusy || voiceStatus === "speaking";

  const micHint = useMemo(() => {
    if (voiceStatus === "recording") return "Tap to stop";
    if (voiceStatus === "processing") return "Working…";
    if (voiceStatus === "speaking") return isPaused ? "Paused — tap mic for a new question" : "Answering…";
    if (voiceStatus === "error") return "Tap to try again";
    return "Tap to speak";
  }, [voiceStatus, isPaused]);

  return (
    <div className="relative flex w-full max-w-4xl flex-col gap-5 overflow-hidden rounded-3xl border border-white/10 bg-white/[0.05] p-5 shadow-[0_25px_70px_rgba(0,0,0,0.45),inset_0_1px_0_rgba(255,255,255,0.08)] backdrop-blur-2xl sm:p-8 lg:p-10">
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-br from-[#e8c547]/[0.06] via-transparent to-[#ff2d78]/[0.04]" />
      <ModeToggle mode={mode} onChange={setMode} disabled={isVoiceBusy} />

      {/* input area */}
      <div className="relative min-h-[92px]">
        <div
          className={`transition-all duration-300 ${mode === "text" ? "translate-x-0 opacity-100" : "pointer-events-none absolute inset-0 -translate-x-2 opacity-0"
            }`}
        >
          <span className="mb-2 block font-mono-hh text-[10px] uppercase tracking-[0.2em] text-[#f5f0e1]/40">
            Text Query
          </span>
          <form onSubmit={handleTextSubmit} className="flex items-center gap-2">
            <input
              type="text"
              value={textQuery}
              onChange={(e) => setTextQuery(e.target.value)}
              placeholder="Ask about anything in Goa…"
              className="flex-1 rounded-xl border border-white/10 bg-white/[0.04] px-4 py-3 text-sm text-[#f5f0e1] placeholder:text-[#f5f0e1]/30 outline-none backdrop-blur-md transition-colors focus:border-[#e8c547]/50 focus:bg-white/[0.06]"
            />
            <button
              type="submit"
              disabled={textLoading || !textQuery.trim()}
              className="shrink-0 rounded-xl bg-[#e8c547] px-5 py-3 font-mono-hh text-xs font-bold uppercase tracking-[0.15em] text-[#0b2e28] transition-transform hover:scale-[1.03] active:scale-95 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:scale-100"
            >
              {textLoading ? "Asking…" : "Ask AI"}
            </button>
          </form>
        </div>

        <div
          className={`transition-all duration-300 ${mode === "voice" ? "translate-x-0 opacity-100" : "pointer-events-none absolute inset-0 translate-x-2 opacity-0"
            }`}
        >
          <span className="mb-3 block text-center font-mono-hh text-[10px] uppercase tracking-[0.2em] text-[#f5f0e1]/40">
            Voice Query
          </span>
          <div className="flex flex-col items-center gap-3">
            <button
              type="button"
              onClick={voiceStatus === "speaking" ? togglePause : handleMicTap}
              disabled={voiceStatus === "processing"}
              className="relative flex h-20 w-20 items-center justify-center rounded-full transition-transform active:scale-95 disabled:cursor-wait"
              aria-label={micHint}
            >
              {isRecording &&
                [0, 0.5, 1].map((delay) => (
                  <span
                    key={delay}
                    className="absolute inset-0 rounded-full border-2"
                    style={{ borderColor: "#ff2d78", animation: `hh-ripple 1.8s ease-out ${delay}s infinite` }}
                  />
                ))}
              {voiceStatus === "processing" && (
                <span
                  className="absolute inset-[-3px] rounded-full border-2 border-transparent border-t-[#e8c547]"
                  style={{ animation: "hh-spin 0.9s linear infinite" }}
                />
              )}
              <span
                className={`flex h-16 w-16 items-center justify-center rounded-full transition-colors duration-300 ${isRecording ? "bg-[#ff2d78]" : voiceStatus === "processing" ? "bg-[#e8c547]/50" : "bg-[#e8c547]"
                  }`}
              >
                {isRecording ? (
                  <svg viewBox="0 0 24 24" className="h-6 w-6 fill-[#0b2e28]">
                    <rect x="6" y="6" width="12" height="12" rx="2" />
                  </svg>
                ) : voiceStatus === "speaking" ? (
                  isPaused ? (
                    <svg viewBox="0 0 24 24" className="ml-0.5 h-7 w-7 fill-[#0b2e28]">
                      <path d="M8 5v14l11-7z" />
                    </svg>
                  ) : (
                    <svg viewBox="0 0 24 24" className="h-7 w-7 fill-[#0b2e28]">
                      <rect x="6" y="5" width="4" height="14" rx="1" />
                      <rect x="14" y="5" width="4" height="14" rx="1" />
                    </svg>
                  )
                ) : (
                  <svg viewBox="0 0 24 24" className="h-7 w-7 fill-[#0b2e28]">
                    <path d="M12 14a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v5a3 3 0 0 0 3 3Z" />
                    <path d="M19 11a1 1 0 1 0-2 0 5 5 0 0 1-10 0 1 1 0 1 0-2 0 7 7 0 0 0 6 6.93V20H9a1 1 0 1 0 0 2h6a1 1 0 1 0 0-2h-2v-2.07A7 7 0 0 0 19 11Z" />
                  </svg>
                )}
              </span>
            </button>
            <span className="font-mono-hh text-[10px] uppercase tracking-[0.15em] text-[#f5f0e1]/40">
              {micHint}
            </span>

            <label className="mt-1 cursor-pointer font-mono-hh text-[10px] uppercase tracking-[0.15em] text-[#f5f0e1]/40 underline decoration-white/20 underline-offset-4 transition-colors hover:text-[#e8c547]">
              Upload audio file
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                onChange={handleFileUpload}
                disabled={isVoiceBusy}
                className="hidden"
              />
            </label>
          </div>
        </div>
      </div>

      {/* pipeline status strip */}
      <div>
        <span className="mb-2 block font-mono-hh text-[10px] uppercase tracking-[0.2em] text-[#f5f0e1]/40">
          Pipeline Status
        </span>
        <PipelineStrip stage={stage} busy={stageBusy} />
      </div>

      {/* error */}
      {error && (
        <p className="rounded-lg border border-[#ff2d78]/30 bg-[#ff2d78]/10 px-3 py-2 text-xs text-[#ff2d78]">
          {error}
        </p>
      )}

      {/* conversation */}
      <div>
        <div className="mb-2 flex items-center justify-between">
          <span className="font-mono-hh text-[10px] uppercase tracking-[0.2em] text-[#f5f0e1]/40">
            Conversation
          </span>
          <label className="flex items-center gap-1.5 font-mono-hh text-[9px] uppercase tracking-[0.15em] text-[#f5f0e1]/40">
            <input
              type="checkbox"
              checked={autoPlay}
              onChange={(e) => setAutoPlay(e.target.checked)}
              className="h-3 w-3 accent-[#e8c547]"
            />
            Autoplay
          </label>
        </div>
        <div className="hh-scrollbar flex max-h-72 flex-col gap-3 overflow-y-auto rounded-xl border border-white/10 bg-white/[0.02] p-3 backdrop-blur-md">
          {conversation.length === 0 ? (
            <p className="py-6 text-center font-mono-hh text-[11px] uppercase tracking-[0.15em] text-[#f5f0e1]/25">
              No questions yet — start typing or tap the mic
            </p>
          ) : (
            conversation.map((turn) => (
              <MessageBubble
                key={turn.id}
                turn={turn}
                autoPlay={autoPlay}
                streamingId={streamingAudioId}
                playingId={playingAudioId}
              />
            ))
          )}
          <div ref={conversationEndRef} />
        </div>
      </div>
    </div>
  );
}