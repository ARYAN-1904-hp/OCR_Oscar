# Intake Agent OCR Engine

End-to-end document intake OCR: upload → PaddleOCR / layout → text & JSON output → handler validation.

| Piece           | Path                                   | Role                                   |
| --------------- | -------------------------------------- | -------------------------------------- |
| Backend API     | `app/main.py`                        | FastAPI`POST /ocr`                   |
| OCR pipeline    | `ocr/`                               | PaddleOCR, preprocess, hybrid layout   |
| Handler / queue | `handler_checker/`                   | Validation, metadata, storage, retries |
| Frontend        | `frontend/`                          | Upload UI + results                    |
| Docker          | `Dockerfile`, `docker-compose.yml` | Handler checker container              |

**Remote:** `https://github.com/rohit-manna-healthplans-ai/Intake_Agent_OCR_Engine.git`

---

# Quick start (local)

All commands assume PowerShell from the project root:

```powershell
cd C:\Users\mysel\OneDrive\Desktop\Intake_Agent_OCR_Engine
```

## 1. Backend (API)

```powershell
# Create / activate venv (once)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies (once)
pip install -r requirements.txt

# Start API on http://127.0.0.1:8000
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# Or with auto-reload while developing
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Health check:

```powershell
curl.exe http://127.0.0.1:8000/health
```

OCR upload (example):

```powershell
curl.exe -F "file=@.\path\to\document.pdf" "http://127.0.0.1:8000/ocr?max_pages=5"
```

API docs (when server is running):

- Swagger UI: http://127.0.0.1:8000/docs
- OpenAPI: http://127.0.0.1:8000/openapi.json

Useful query params on `POST /ocr`:

| Param             | Default   | Notes                                                                               |
| ----------------- | --------- | ----------------------------------------------------------------------------------- |
| `preset`        | `auto`  | `auto`, `clean_doc`, `photo`, `low_light`, `scan_enhance`, `printed_hq` |
| `dpi`           | `150`   | PDF render DPI                                                                      |
| `max_pages`     | `5`     | Cap pages processed                                                                 |
| `return_debug`  | `false` | Extra debug fields                                                                  |
| `return_layout` | `true`  | Include layout summary                                                              |

Optional env vars (speed / features):

```powershell
$env:OCR_MULTI_PASS = "0"          # 1 = multi-pass OCR (slower)
$env:OCR_ANGLE_CLS = "0"           # 1 = enable angle classification
$env:OCR_USE_PPSTRUCTURE = "0"     # 1 = enable PPStructure merge
$env:OCR_MAX_DIM = "1600"          # max image side before OCR
```

## 2. Frontend

In a **second** terminal:

```powershell
cd C:\Users\mysel\OneDrive\Desktop\Intake_Agent_OCR_Engine\frontend
python -m http.server 5500
```

Open: http://127.0.0.1:5500/

The UI calls `http://127.0.0.1:8000/ocr` by default. To point elsewhere, set before load:

```html
<script>window.OCR_API_BASE = "http://127.0.0.1:8000";</script>
```

Hard refresh if UI looks stale: `Ctrl + F5`.

## 3. Handler checker (orchestration)

```powershell
.\.venv\Scripts\Activate.ps1

# Run handler main loop
python handler_checker/main.py

# Or via project runner
python run.py workflow
python run.py integration_test
python run.py all
```

List runner options:

```powershell
python run.py
```

---

# Docker

Requires Docker Desktop (Windows) and a `.env` file if your compose `env_file` expects one.

```powershell
cd C:\Users\mysel\OneDrive\Desktop\Intake_Agent_OCR_Engine

# Build image
docker compose build

# Start handler-checker service
docker compose up -d

# Logs
docker compose logs -f handler-checker

# Stop
docker compose down
```

Build / run image directly:

```powershell
docker build -t intake-ocr-engine .
docker run --rm --name handler_checker --env-file .env `
  -v ${PWD}/handler_checker/storage:/app/handler_checker/storage `
  -v ${PWD}/handler_checker/logs:/app/handler_checker/logs `
  -v ${PWD}/handler_checker/metadata:/app/handler_checker/metadata `
  -v ${PWD}/handler_checker/queue:/app/handler_checker/queue `
  intake-ocr-engine
```

> Note: Current `Dockerfile` / `docker-compose.yml` run `handler_checker/main.py`. The FastAPI UI stack is typically run locally with uvicorn + the frontend static server (see Quick start).

---

# GitHub / Git

Remote:

```powershell
git remote -v
# origin  https://github.com/rohit-manna-healthplans-ai/Intake_Agent_OCR_Engine.git
```

## Clone / pull

```powershell
git clone https://github.com/rohit-manna-healthplans-ai/Intake_Agent_OCR_Engine.git
cd Intake_Agent_OCR_Engine

git checkout develop
git pull origin develop
```

## Daily workflow

```powershell
git status
git checkout develop
git pull origin develop

git checkout -b feature/<feature-name>
# ... make changes ...

git add .
git commit -m "$(Get-Date -Format 'yyyy-MM-dd') | feat: short description"
git push -u origin HEAD
```

Open a PR with GitHub CLI:

```powershell
gh pr create --base develop --title "feat: short description" --body "## Summary`n- ...`n`n## Test plan`n- [ ] Local API`n- [ ] Frontend upload`n- [ ] Docker (if touched)"
```

## Branch strategy

| Branch                                 | Purpose                             |
| -------------------------------------- | ----------------------------------- |
| `main`                               | Production-ready only               |
| `develop`                            | Active development (default)        |
| `testing`                            | QA / validation before release      |
| `archive_ocr_ov1` / archive branches | Historical OCR versions (read-only) |

### Merge develop → testing

```powershell
git checkout testing
git pull origin testing
git merge develop
git push origin testing
```

### Merge testing → main

```powershell
git checkout main
git pull origin main
git merge testing
git push origin main
```

### Merge develop → main (approved releases only)

```powershell
git checkout main
git pull origin main
git merge develop
git push origin main
```

### Keep develop updated with main

```powershell
git checkout develop
git pull origin develop
git merge main
git push origin develop
```

### Feature branches

```powershell
git checkout develop
git pull origin develop
git checkout -b feature/<feature-name>
git push -u origin feature/<feature-name>

# cleanup
git branch -d feature/<feature-name>
git push origin --delete feature/<feature-name>
```

---

# Commit message standard

```text
YYYY-MM-DD | <type>: <short description>
```

Example:

```text
2026-07-21 | feat: add processing details accordion to frontend
```

| Type     | Description                         |
| -------- | ----------------------------------- |
| feat     | New feature                         |
| fix      | Bug fix                             |
| docs     | Documentation                       |
| refactor | Restructure without behavior change |
| perf     | Performance                         |
| test     | Tests                               |
| style    | Formatting only                     |
| chore    | Maintenance                         |
| build    | Build config                        |
| ci       | CI/CD                               |
| revert   | Revert                              |

---

# Best practices

- Do not commit directly to `main`.
- Develop on `develop` or `feature/*` branches.
- Validate on `testing` before production.
- Never commit `.env`, secrets, or large model caches.
- Keep commit messages dated and typed as above.
- Archive branches are historical only — no new development there.

---

# Project layout (short)

```text
Intake_Agent_OCR_Engine/
├── app/                 # FastAPI OCR API
├── ocr/                 # OCR + layout pipeline
├── handler_checker/     # Queue, validation, storage
├── frontend/            # Static upload UI
├── docs/                # Module docs
├── tests/               # Unit / integration tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── run.py
└── README.md
```
