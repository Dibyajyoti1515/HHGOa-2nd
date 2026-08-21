Absolutely. Based on the project structure and voice/retrieval pipeline we've been working on, here's a solid GitHub-ready `README.md`.

````markdown
# 🎙️ Voice Knowledge Assistant

A multilingual, knowledge-powered voice assistant that combines **Sarvam AI Speech-to-Text**, **hybrid vector retrieval with Qdrant**, **Groq LLM fallback**, and **Sarvam AI Text-to-Speech**.

The system is designed around a retrieval-first architecture:

```text
                    ┌─────────────────────┐
                    │      Browser        │
                    │   Microphone Input  │
                    └──────────┬──────────┘
                               │
                               │ WebSocket
                               ▼
                    ┌─────────────────────┐
                    │    Sarvam STT       │
                    │   Saaras v3 Realtime│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   User Transcript  │
                    │  + Language Code   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Guardrails      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Translation to      │
                    │ English (if needed) │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Hybrid Retrieval   │
                    │       Qdrant        │
                    └──────────┬──────────┘
                               │
                     ┌─────────┴─────────┐
                     │                   │
                High Confidence     Low Confidence
                     │                   │
                     ▼                   ▼
                Retrieved Text       Groq LLM
                     │                   │
                     └─────────┬─────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Translate Answer    │
                    │ to User Language    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Sarvam TTS      │
                    │      Bulbul v3      │
                    └──────────┬──────────┘
                               │
                               │ Audio
                               ▼
                    ┌─────────────────────┐
                    │      Browser        │
                    │   Audio Playback    │
                    └─────────────────────┘
````

---

# ✨ Features

* 🎤 Real-time voice input
* 🧠 Hybrid vector retrieval
* 🔎 Qdrant vector database
* 🌐 Multilingual input support
* 🇮🇳 Hindi support
* 🟠 Odia support
* 🟡 Tamil support
* 🇬🇧 English support
* 🔤 Automatic language detection through STT
* 🌍 Translation of non-English queries to English for retrieval
* 🤖 Groq LLM fallback for low-confidence retrieval
* 🔊 Sarvam Bulbul v3 text-to-speech
* 🛡️ Input guardrails
* ⚡ WebSocket-based voice pipeline
* 📊 Detailed stage-by-stage latency measurement
* 📈 Retrieval evaluation against 100 questions
* ⏱️ STT, translation, retrieval, Groq and TTS timing
* ⏸️ Pause/stop audio playback from the frontend
* 💬 Typing-style answer display
* ❤️ Backend health monitoring

---

# 🏗️ Architecture

## Voice Pipeline

```text
Browser Microphone
        │
        ▼
WebSocket /v1/voice
        │
        ▼
Sarvam Saaras v3 Realtime STT
        │
        ├── transcript
        └── language_code
                 │
                 ▼
             Guardrails
                 │
                 ▼
       Translation → English
                 │
                 ▼
       Hybrid Weighted Retrieval
                 │
                 ▼
             Qdrant
                 │
                 ▼
        Confidence Gate
          threshold = 0.85
             │
       ┌─────┴─────┐
       │           │
      HIGH        LOW
       │           │
       ▼           ▼
   Retrieved     Groq
     Text         LLM
       │           │
       └─────┬─────┘
             │
             ▼
    Answer Translation
      English → User
        Language
             │
             ▼
       Sarvam Bulbul v3
             │
             ▼
        Audio Stream
             │
             ▼
          Browser
```

---

# 🧩 Project Structure

```text
HHGoa/
│
├── project/
│   ├── api/
│   │   ├── app.py
│   │   └── ws_voice.py
│   │
│   ├── config/
│   │   └── settings.py
│   │
│   ├── guardrails/
│   │   └── input_guardrails.py
│   │
│   ├── llm_fallback/
│   │   └── groq_client.py
│   │
│   ├── stt/
│   │   └── sarvam_stt_client.py
│   │
│   ├── translation/
│   │   └── sarvam_translate_client.py
│   │
│   ├── tts/
│   │   ├── sarvam_tts_client.py
│   │   └── text_trim.py
│   │
│   └── timing/
│       └── stage_timer.py
│
├── retrieval/
│   └── factory.py
│
├── scripts/
│   └── retrieval_100_questions.py
│
├── Frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   └── VoiceAssistant.jsx
│   ├── package.json
│   └── package-lock.json
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

# 🛠️ Tech Stack

## Backend

* Python
* FastAPI
* WebSockets
* Uvicorn

## Speech

### Speech-to-Text

**Sarvam Saaras v3 Realtime**

Used for:

* Speech recognition
* Language detection
* Real-time transcription

### Text-to-Speech

**Sarvam Bulbul v3**

Used to generate audio responses in the detected user language.

---

# 🔎 Retrieval

The retrieval system uses:

```text
Query
  │
  ├── Embedding
  │
  ├── Qdrant Search
  │
  └── Hybrid Fusion
          │
          ▼
     Ranked Results
```

The retrieval API exposes:

```python
retrieve(
    query_text,
    mode="hybrid_weighted"
)
```

The result contains:

```python
{
    "results": [...],
    "embedding_ms": ...,
    "qdrant_ms": ...,
    "fusion_ms": ...,
    "total_ms": ...
}
```

---

# 🎯 Confidence Gate

The system uses a confidence threshold of:

```text
0.85
```

### High confidence

If:

```text
score >= 0.85
```

the system uses the retrieved context directly and avoids an unnecessary LLM call.

```text
Query
  ↓
Qdrant
  ↓
Score >= 0.85
  ↓
Retrieved Answer
```

### Low confidence

If:

```text
score < 0.85
```

the system sends the query and retrieved context to Groq.

```text
Query
  ↓
Qdrant
  ↓
Score < 0.85
  ↓
Groq
  ↓
Generated Answer
```

This reduces unnecessary LLM calls when the knowledge base already contains a strong answer.

---

# 🌐 Multilingual Processing

The original user language is preserved.

For example:

```text
User language:
hi-IN

User transcript:
अमेरिका का राष्ट्रपति कौन है?
```

The retrieval query is converted to English:

```text
Who is the president of America?
```

The English query is used for Qdrant retrieval.

The final answer is then converted back to the user's original language:

```text
Hindi → Hindi answer
Odia → Odia answer
Tamil → Tamil answer
English → English answer
```

The language code detected by STT is preserved throughout the pipeline.

---

# 🗣️ Supported Languages

The architecture supports Sarvam's supported Indian language codes.

Examples:

| Language  | Code    |
| --------- | ------- |
| English   | `en-IN` |
| Hindi     | `hi-IN` |
| Odia      | `od-IN` |
| Tamil     | `ta-IN` |
| Telugu    | `te-IN` |
| Bengali   | `bn-IN` |
| Marathi   | `mr-IN` |
| Gujarati  | `gu-IN` |
| Kannada   | `kn-IN` |
| Malayalam | `ml-IN` |
| Punjabi   | `pa-IN` |

The detected language is passed to the TTS stage so that the answer can be spoken in the user's language.

---

# 🛡️ Guardrails

Guardrails are executed **before retrieval**.

```text
STT
 ↓
Guardrails
 ↓
Translation
 ↓
Retrieval
```

If a request violates the configured guardrails, the request is stopped before the retrieval pipeline continues.

---

# 📡 API

## Health

```http
GET /v1/health
```

Example:

```bash
curl http://localhost:8000/v1/health
```

---

# 🎙️ Voice WebSocket

```text
ws://localhost:8000/v1/voice
```

The browser sends raw PCM audio frames.

Control messages can be sent through the same WebSocket.

Example:

```json
{
  "event": "stop"
}
```

The server sends messages such as:

### Transcript

```json
{
  "type": "transcript",
  "text": "What causes earthquakes?",
  "language": "en-IN"
}
```

### Final response

```json
{
  "type": "final",
  "transcript": "What causes earthquakes?",
  "answer_text": "Earthquakes are caused by...",
  "timings": {}
}
```

---

# 📊 Performance Monitoring

The voice pipeline records timing information for individual stages.

Example:

```text
STT
Translation
Guardrails
Embedding
Qdrant
Fusion
Retrieval
Groq
TTS
Total
```

Example server log:

```text
voice.timing_summary
stt_ms=7951.76
guardrail_ms=0.62
retrieval_wall_ms=876.15
embedding_ms=111.65
qdrant_ms=57.42
fusion_ms=0.14
retrieval_total_ms=169.22
generation=groq
llm_used=True
tts_ms=11250.58
total_turn_ms=25749.09
```

The timing information is also returned to the frontend.

---

# 💻 Frontend

The frontend is built with:

* React
* Vite
* WebSocket
* Web Audio API

It provides:

* Voice recording
* Recording state
* Processing state
* Answer display
* Typing animation
* Audio playback
* Pause functionality
* Performance metrics
* Backend health status

Run the frontend:

```bash
cd Frontend
npm install
npm run dev
```

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd HHGoa
```

---

## 2. Create virtual environment

Windows PowerShell:

```powershell
python -m venv mlenv
```

Activate:

```powershell
.\mlenv\Scripts\Activate.ps1
```

If using Git Bash:

```bash
source /d/mlenv/Scripts/activate
```

---

## 3. Install Python dependencies

```powershell
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create:

```text
.env
```

from:

```text
.env.example
```

Example configuration:

```env
SARVAM_API_KEY=your_sarvam_api_key
GROQ_API_KEY=your_groq_api_key

QDRANT_URL=http://localhost:6333
```

Do **not** commit `.env`.

The repository should contain:

```text
.env.example
```

but never:

```text
.env
```

---

# 🗄️ Qdrant

The project uses Qdrant as the vector database.

Expected local endpoint:

```text
http://localhost:6333
```

Verify Qdrant:

```powershell
curl http://localhost:6333
```

The retrieval system expects the configured collection to exist before running the voice assistant.

---

# ▶️ Running the Backend

From the project root:

```powershell
uvicorn project.api.app:app --reload
```

The API should be available at:

```text
http://localhost:8000
```

Health check:

```text
http://localhost:8000/v1/health
```

---

# ▶️ Running the Frontend

Open another terminal:

```powershell
cd Frontend
npm install
npm run dev
```

Then open the URL shown by Vite.

---

# 🧪 Retrieval Evaluation

The project contains a retrieval evaluation script:

```text
scripts/retrieval_100_questions.py
```

It evaluates 100 questions against the hybrid retrieval system.

Run it from the **project root**:

```powershell
python .\scripts\retrieval_100_questions.py
```

Do not run it from inside `scripts/` unless the Python path is configured appropriately.

The evaluation reports:

```text
Total questions
Score > 0.90
Score <= 0.90
Pass rate
Average score
Best score
Worst score
```

---

# 📈 Retrieval Evaluation

One evaluation produced:

```text
Total questions : 100
Score > 0.90    : 36
Score <= 0.90   : 64
Pass rate       : 36.00%

Average score   : 0.8463
Best score      : 1.0000
Worst score     : 0.7500
```

The evaluation also identifies questions suitable for demonstration based on retrieval score.

---

# 🎥 Recommended Demo Questions

Some high-scoring questions from the retrieval evaluation include:

```text
Why do tectonic plates move?

What causes a rainbow?

What causes tides in the ocean?

What causes muscle soreness after exercise?

What causes soil erosion?

What causes air pollution?

What is the difference between speed and velocity?

What is potential energy?

How does cloud computing work?

How did the Roman Empire fall?
```

These questions achieved retrieval scores at or very close to `1.0` in the evaluation.

For demonstrations, the retrieved context should also be manually checked rather than selecting questions solely by numerical score.

---

# 🧠 Design Principles

The project follows several important design principles.

### Retrieval first

The LLM is not automatically called for every query.

```text
Strong retrieval → use knowledge base
Weak retrieval   → use LLM fallback
```

### Preserve user language

The original transcript and detected language are preserved independently from the English retrieval query.

```text
User Speech
    ↓
Original Transcript
    ↓
Language Code
    ↓
English Retrieval Query
```

### No unnecessary LLM calls

High-confidence retrieval avoids Groq generation.

### Measurable pipeline

Each major stage reports latency so that performance bottlenecks can be identified.

---

# 📁 Data and Large Files

Large datasets and generated files are intentionally excluded from Git.

Examples:

```text
MSMARCO-XI/
*.parquet
*.db
qdrant_storage*/
hf_cache/
fastembed_cache/
logs/
tts_out/
Frontend/node_modules/
```

These files should be obtained/generated locally rather than committed to the repository.

---

# 🚫 What Is Not Committed

The following are intentionally excluded:

```text
.env
MSMARCO-XI/
qdrant_storage/
*.parquet
*.db
hf_cache/
fastembed_cache/
logs/
tts_out/
node_modules/
mlenv/
```

This keeps the Git repository lightweight and prevents API keys, local databases, vector stores, datasets, caches, and generated files from being uploaded.

---

# 🔧 Troubleshooting

## Backend cannot start

Check the virtual environment:

```powershell
where python
```

Expected:

```text
...\mlenv\Scripts\python.exe
```

Check dependencies:

```powershell
pip install -r requirements.txt
```

---

## Qdrant connection failed

Check:

```powershell
curl http://localhost:6333
```

If Qdrant is not running, start the configured Qdrant instance.

---

## Frontend cannot connect

Check:

```text
VITE_WS_BASE
VITE_API_BASE
```

The WebSocket endpoint should point to:

```text
ws://localhost:8000
```

and the HTTP API should point to:

```text
http://localhost:8000
```

---

## `ModuleNotFoundError: retrieval`

Run scripts from the repository root:

```powershell
cd E:\HHGoa
python .\scripts\retrieval_100_questions.py
```

---

# 🔒 Security

Never commit:

```text
SARVAM_API_KEY
GROQ_API_KEY
.env
```

If an API key is accidentally committed to GitHub, revoke/rotate it immediately.

---

# 🚀 Future Improvements

Potential improvements include:

* Streaming answer generation
* Lower STT latency
* Faster embedding initialization
* Persistent SentenceTransformer loading
* Faster TTS first-byte latency
* Better retrieval evaluation
* Multilingual retrieval evaluation
* Better confidence calibration
* Streaming text response to the frontend
* Improved audio buffering
* Production WebSocket deployment
* Authentication and rate limiting
* Observability dashboard

---

# 📜 License

Add your project license here.

Example:

```text
MIT License
```

---

# 👨‍💻 Project

**HHGoa**

A multilingual, retrieval-first voice knowledge assistant combining speech recognition, vector search, LLM fallback, translation, and speech synthesis.

````

This is ready to save as:

```text
E:\HHGoa\README.md
````

One thing I intentionally left as a placeholder is the **GitHub repository URL** and **license**, since those weren't established yet.
