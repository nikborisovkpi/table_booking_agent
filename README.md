# London Restaurant Booking Agent (Telegram MTProto + OpenRouter + Voice)

Autonomous Telegram agent that initiates a polite **British English** conversation with a London restaurant manager (`@vlad_dev_7`), negotiates and books a table for **2 guests at 17:00 today** (`Europe/London`), escalates any unknown questions to the human owner without hallucinating, supports **bidirectional voice notes** (STT + British TTS), and writes a **single-line decision log** per turn.

---

## Features

- **Initiates Conversation First (MTProto Userbot):** Built on `Telethon` because standard Telegram Bot API bots cannot DM a user first.
- **Polite British Etiquette & Anti-Hallucination:** Powered by `anthropic/claude-haiku-5.5` via OpenRouter with a strict tool-calling loop (`ask_owner` + `submit_turn`). Anything outside known baseline facts (`Nikita`, `2 guests`, `17:00 today in London`) is escalated to the owner's **Saved Messages (`me`)**.
- **Mandatory 4-Field Confirmation Guard:** Pydantic + turn-level self-correction ensures `booking_confirmed=True` is only accepted once the restaurant's exact name is known and the final message explicitly repeats **time, date, party size, and restaurant name**.
- **Message Debouncing:** Coalesces multi-message bursts from the manager (`DEBOUNCE_SECONDS=6.0`, extended automatically while the manager is typing or recording audio).
- **Voice Messages (Bonus Task):**
  - **STT:** Converts incoming `.ogg` voice notes to `16kHz mono .wav` via `ffmpeg` and transcribes via OpenRouter multimodal audio (`OPENROUTER_STT_MODEL`).
  - **TTS:** Synthesizes British speech with `edge-tts` (`en-GB-RyanNeural`) and encodes `.ogg (libopus, voip)` via `ffmpeg` for native Telegram voice waveforms, mirroring the manager's modality (text → text, voice → voice).
- **Single-Account Self-Test Mode:** Supports testing both roles (Manager in a private 1-person Telegram group + Owner in `Saved Messages`) from a single Telegram account with lock-synchronized self-echo filtering.

---

## Project Structure

```text
├── src/
│   ├── config.py            # .env settings & Europe/London date helpers
│   ├── models.py            # Pydantic schemas & confirmation validators
│   ├── prompts.py           # British etiquette & anti-hallucination system prompt
│   ├── llm_agent.py         # OpenRouter tool-calling loop & Req 4 self-correction
│   ├── human_escalation.py  # Saved Messages ('me') owner consultation bridge
│   ├── voice.py             # OpenRouter STT + Edge-TTS / ffmpeg Opus pipeline
│   ├── decision_logger.py   # Strict 1-line-per-turn decision logger
│   ├── telegram_client.py   # Telethon runner, debouncer & modality mirroring
│   └── main.py              # CLI entry point & runtime safety guards
├── tests/                   # 23 unit & integration tests (pytest + pytest-asyncio)
├── logs/                    # Generated decision.log (1 line per turn)
├── SUBMISSION_NOTES.md      # Deliverables #4 and #5 notes (Ukrainian)
├── .env.example             # Configuration template (safe for GitHub)
└── requirements.txt
```

---

## Setup

1. Ensure **Python 3.11+** and **`ffmpeg`** (with `libopus`) are installed and available on `PATH`.
2. Create virtual environment and install dependencies:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your credentials (`TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, `OPENROUTER_API_KEY`).

---

## Step 1: Self-Testing (Single Telegram Account)

1. Create a private group in Telegram with only yourself (e.g. `Test London Restaurant`) and copy its invite link (`https://t.me/+...`) or exact title.
2. In `.env`, set:
   ```env
   SELF_TEST_MODE=true
   TARGET_CHAT=https://t.me/+your_invite_link_or_group_title
   OWNER_ESCALATION_CHAT=me
   ```
3. Run the agent interactively in PowerShell / Terminal:
   ```powershell
   .\.venv\Scripts\python.exe -m src.main
   ```
   *(On the first run, Telethon will ask for your phone number and Telegram login code once to create the local git-ignored `booking_agent.session`.)*
4. Test scenarios in your private group (acting as the restaurant manager):
   - Reply in text and in voice notes; send 2–3 short messages in a row to observe debouncing.
   - Ask an unknown question (e.g. *"5pm is full, would 5:30pm work? Any allergies?"*) → check your **Saved Messages** (`me`), reply there as the owner, and watch the agent answer in the restaurant group.
   - Confirm the booking including a restaurant name → verify the agent repeats **time, date, party size, and restaurant name** and exits cleanly.

---

## Step 2: Live Run with `@vlad_dev_7`

1. Clear or move the test `logs/decision.log` so the final log contains only the real conversation:
   ```powershell
   Remove-Item logs\decision.log -ErrorAction SilentlyContinue
   ```
2. Update `.env` for production:
   ```env
   SELF_TEST_MODE=false
   TARGET_CHAT=@vlad_dev_7
   ```
3. Keep your **Saved Messages** open in Telegram (in case the manager asks something outside your baseline facts) and start the agent:
   ```powershell
   .\.venv\Scripts\python.exe -m src.main
   ```

---

## Running Tests

```powershell
.\.venv\Scripts\pytest.exe -v
```
