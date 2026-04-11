@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" >nul

set "ENV_FILE=%SCRIPT_DIR%.env.local"
set "OUTPUT_DIR=%SCRIPT_DIR%outputs"
set "TIMESTAMP=%DATE:~0,4%%DATE:~5,2%%DATE:~8,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%"
set "TIMESTAMP=%TIMESTAMP: =0%"
set "COMBINED_LOG=%OUTPUT_DIR%\batch_run_%TIMESTAMP%.log"

set "STOCK_LIST_FILE="
set "BATCH_SAVE_COMBINED_LOG=1"

if not exist "%ENV_FILE%" (
  echo [ERROR] Missing .env.local file: "%ENV_FILE%"
  popd >nul
  exit /b 1
)

for /f "usebackq tokens=* delims=" %%R in ("%ENV_FILE%") do (
  set "line=%%R"
  for /f "tokens=* delims= " %%A in ("!line!") do set "line=%%A"
  if defined line (
    if not "!line:~0,1!"=="#" (
      if not "!line:~0,1!"==";" (
        for /f "tokens=1* delims==" %%K in ("!line!") do (
          set "env_key=%%K"
          set "env_val=%%L"
          for /f "tokens=* delims= " %%P in ("!env_key!") do set "env_key=%%P"
          for /f "tokens=* delims= " %%Q in ("!env_val!") do set "env_val=%%Q"
          set "env_key=!env_key: =!"
          if /i "!env_key!"=="STOCK_LIST_FILE" set "STOCK_LIST_FILE=!env_val!"
          if /i "!env_key!"=="BATCH_SAVE_COMBINED_LOG" set "BATCH_SAVE_COMBINED_LOG=!env_val!"
        )
      )
    )
  )
)

if not defined STOCK_LIST_FILE (
  echo [ERROR] STOCK_LIST_FILE is missing in .env.local
  echo [HINT] Add: STOCK_LIST_FILE=C:\path\to\stocks.txt
  popd >nul
  exit /b 1
)

if "%STOCK_LIST_FILE:~0,1%"=="\" set "STOCK_LIST_FILE=%STOCK_LIST_FILE:~1%"
if "%STOCK_LIST_FILE:~-1%"=="\" set "STOCK_LIST_FILE=%STOCK_LIST_FILE:~0,-1%"

if "%BATCH_SAVE_COMBINED_LOG:~0,1%"=="\" set "BATCH_SAVE_COMBINED_LOG=%BATCH_SAVE_COMBINED_LOG:~1%"
if "%BATCH_SAVE_COMBINED_LOG:~-1%"=="\" set "BATCH_SAVE_COMBINED_LOG=%BATCH_SAVE_COMBINED_LOG:~0,-1%"

if not exist "%STOCK_LIST_FILE%" (
  echo [ERROR] STOCK_LIST_FILE does not exist: "%STOCK_LIST_FILE%"
  popd >nul
  exit /b 1
)

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

if /i "%BATCH_SAVE_COMBINED_LOG%"=="1" (
  > "%COMBINED_LOG%" echo [INFO] Batch run started at %DATE% %TIME%
  >> "%COMBINED_LOG%" echo [INFO] Stock list file: %STOCK_LIST_FILE%
)

echo [INFO] Using stock list: %STOCK_LIST_FILE%
echo [INFO] Output directory: %OUTPUT_DIR%

set /a total=0
set /a success=0
set /a failed=0
set /a skipped=0

for /f "usebackq tokens=* delims=" %%T in ("%STOCK_LIST_FILE%") do (
  set "line=%%T"
  set "ticker="

  for /f "tokens=* delims= " %%A in ("!line!") do set "line=%%A"
  if not defined line (
    set /a skipped+=1
  ) else if "!line:~0,1!"=="#" (
    set /a skipped+=1
  ) else if "!line:~0,1!"==";" (
    set /a skipped+=1
  ) else if "!line:~0,2!"=="//" (
    set /a skipped+=1
  ) else (
    set "ticker=!line!"
    if "!ticker:~0,1!"=="\" set "ticker=!ticker:~1!"
    if "!ticker:~-1!"=="\" set "ticker=!ticker:~0,-1!"

    echo(!ticker!| findstr /r /i /c:"^[A-Z0-9][A-Z0-9.\-]*$" >nul
    if errorlevel 1 (
      echo [WARN] Skipping invalid ticker line: !line!
      if /i "%BATCH_SAVE_COMBINED_LOG%"=="1" >> "%COMBINED_LOG%" echo [WARN] Invalid ticker skipped: !line!
      set /a skipped+=1
    ) else (
      set /a total+=1
      set "ticker_log=%OUTPUT_DIR%\!ticker!_%TIMESTAMP%.log"

      echo.
      echo [INFO] Running !total!: !ticker!
      echo [INFO] Writing per-ticker log: !ticker_log!
      if /i "%BATCH_SAVE_COMBINED_LOG%"=="1" (
        >> "%COMBINED_LOG%" echo.
        >> "%COMBINED_LOG%" echo ============================================================
        >> "%COMBINED_LOG%" echo [TICKER START] !ticker! ^| %DATE% %TIME%
        >> "%COMBINED_LOG%" echo ============================================================
      )

      call .\.venv\Scripts\python.exe main.py --ticker !ticker! --llm-all-providers --debug --chart > "!ticker_log!" 2>&1
      set "rc=!errorlevel!"
      if not "!rc!"=="0" (
        set /a failed+=1
        echo [ERROR] !ticker! failed with exit code !rc!. Continuing...
        if /i "%BATCH_SAVE_COMBINED_LOG%"=="1" (
          >> "%COMBINED_LOG%" type "!ticker_log!"
          >> "%COMBINED_LOG%" echo.
          >> "%COMBINED_LOG%" echo [TICKER END] !ticker! ^| status=failed ^| exit_code=!rc!
          >> "%COMBINED_LOG%" echo ============================================================
          >> "%COMBINED_LOG%" echo.
        )
      ) else (
        set /a success+=1
        echo [OK] !ticker! completed
        if /i "%BATCH_SAVE_COMBINED_LOG%"=="1" (
          >> "%COMBINED_LOG%" type "!ticker_log!"
          >> "%COMBINED_LOG%" echo.
          >> "%COMBINED_LOG%" echo [TICKER END] !ticker! ^| status=success ^| exit_code=0
          >> "%COMBINED_LOG%" echo ============================================================
          >> "%COMBINED_LOG%" echo.
        )
      )
    )
  )
)

echo.
echo [DONE] Batch run finished.
echo [DONE] Processed tickers: %total%
echo [DONE] Success: %success%  Failed: %failed%  Skipped: %skipped%

if /i "%BATCH_SAVE_COMBINED_LOG%"=="1" (
  >> "%COMBINED_LOG%" echo [DONE] Finished at %DATE% %TIME%
  >> "%COMBINED_LOG%" echo [DONE] Processed=%total% Success=%success% Failed=%failed% Skipped=%skipped%
  echo [DONE] Combined log: %COMBINED_LOG%
)

popd >nul
exit /b 0

pause
