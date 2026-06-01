@echo off
REM ============================================================
REM watchdog.bat — Auto-Healing Wrapper para Workers Vision-MVP
REM
REM Uso: watchdog.bat <script.py>
REM   Ejemplo: watchdog.bat worker_bandejas.py
REM
REM Reinicia automáticamente el worker si termina por cualquier
REM razón (excepción, crash, error de cámara).
REM ============================================================

if "%~1"=="" (
    echo [WATCHDOG] ERROR: Especifica el script a ejecutar.
    echo [WATCHDOG] Uso: watchdog.bat worker_bandejas.py
    exit /b 1
)

echo ============================================================
echo   VISION-MVP WATCHDOG
echo   Script: %1
echo   Inicio: %date% %time%
echo ============================================================

:loop
echo.
echo [WATCHDOG] [%time%] Iniciando %1...
echo ------------------------------------------------------------
python %1
set EXIT_CODE=%ERRORLEVEL%
echo ------------------------------------------------------------
echo [WATCHDOG] [%time%] %1 termino con codigo %EXIT_CODE%.

if %EXIT_CODE%==0 (
    echo [WATCHDOG] Salida limpia (codigo 0). No se reiniciara.
    exit /b 0
)

echo [WATCHDOG] Reiniciando en 3 segundos...
timeout /t 3 /nobreak >nul
goto loop
