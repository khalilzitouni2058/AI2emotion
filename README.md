# ISS - API and Frontend Launch Guide

This repository contains a Python FastAPI backend and a React (Vite) frontend.

## Prerequisites

- Python 3.10+
- Node.js 18+
- npm

## 1. Start the API

From project root:

```bash
python scripts/api_server.py
```

API default address:

- http://localhost:8000

Health endpoint:

- http://localhost:8000/api/v1/health

## 2. Start the Frontend

Open a new terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend dev address (default Vite):

- http://localhost:5173

## 3. Typical Development Flow

1. Start backend with `python scripts/api_server.py`.
2. Start frontend with `npm run dev` inside `frontend`.
3. Open frontend in browser and test upload/analyze flow.
4. Check API logs in backend terminal for request/debug output.

## 4. Project Structure (Short)

- `scripts/`: runnable entry points
- `app/`: backend code (core/domain/infrastructure/services/presentation)
- `frontend/`: React app
- `data/uploads/`: uploaded audio files
- `data/analysis/`: generated outputs/results
- `static/`: fallback static assets

## 5. Notes

- Previous root entrypoints were reorganized into `scripts/`.
- If a command fails due to missing modules, activate your Python environment and reinstall dependencies:

```bash
pip install -r requirements.txt
```

- If frontend dependencies are missing:

```bash
cd frontend
npm install
```
