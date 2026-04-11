# PowerShell Command Quick Reference

## Quick Start

### 1. Activate virtual environment (required each new terminal)
```powershell
& .\.venv\Scripts\Activate.ps1
```

### 2. Run with defaults (TSLA vs SPY)
```powershell
python main.py
```

### 3. Run with explicit ticker
```powershell
python main.py --ticker AAPL --benchmark SPY --lookback 252
```

---

## Main CLI Commands

### Basic risk run
```powershell
python main.py --ticker TSLA
```

### Disable LLM summary
```powershell
python main.py --ticker TSLA --disable-llm
```

### Run all LLM providers independently
```powershell
python main.py --ticker TSLA --llm-all-providers
```

### Show debug diagnostics
```powershell
python main.py --ticker TSLA --debug
```

### Force sample market data (no live Yahoo fetch)
```powershell
python main.py --ticker TSLA --no-real-market-data
```

### Render 30-day ASCII chart
```powershell
python main.py --ticker TSLA --chart
```

### Export output JSON
```powershell
python main.py --ticker TSLA --output-json main_output_tsla.json
```

### Use external config JSON
```powershell
python main.py --config config/custom_run.json
```

---

## LLM and Environment

### Local secret file (not committed)
Create `.env.local` in project root and add:
```env
GEMINI_API_KEY=your_key_here
STOCK_LIST_FILE=C:\path\to\stock_list.txt
# Optional
# GEMINI_MODEL=gemini-1.5-flash
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_MODEL=llama3:latest
# OLLAMA_TIMEOUT_SECONDS=600
# BATCH_SAVE_COMBINED_LOG=1
```

### Verify Ollama service is running
```powershell
Invoke-RestMethod http://localhost:11434/api/tags
```

---

## Tests

### Run all unit tests
```powershell
python -m unittest discover -s tests
```

### Run a single test file
```powershell
python -m unittest tests.test_llm
```

---

## Useful Shortcuts

### Recommended daily workflow
```powershell
& .\.venv\Scripts\Activate.ps1
python main.py --ticker TSLA --chart
```

### Investigate provider failures
```powershell
& .\.venv\Scripts\Activate.ps1
python main.py --ticker TSLA --llm-all-providers --debug --chart
```

### Batch run from ticker list TXT
```powershell
& .\run_batch_analysis.bat
```

Notes:
- `run_batch_analysis.bat` reads `STOCK_LIST_FILE` from `.env.local`.
- Supports blank lines and comment lines (`#`, `;`, `//`) in the ticker TXT.
- Saves one file per ticker under `outputs/`.
- Saves a per-run combined log like `outputs/batch_run_20260411_153206.log` (disable with `BATCH_SAVE_COMBINED_LOG=0`).

---

## Troubleshooting

### If activation is blocked by execution policy
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& .\.venv\Scripts\Activate.ps1
```

### If `.venv` does not exist
```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Check Python and pip in current terminal
```powershell
python --version
pip --version
```

---

Last updated: 2026-04-11
