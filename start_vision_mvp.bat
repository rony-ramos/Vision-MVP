@echo off
TITLE Vision-MVP Launcher
echo ============================================================
echo   INICIANDO VISION-MVP (Sistema Completo)
echo ============================================================

REM Intenta activar el entorno virtual si existe
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Entorno virtual detectado.
    set "ACTIVATE_CMD=call .venv\Scripts\activate.bat && "
) else (
    echo [INFO] Entorno virtual no detectado. Usando Python global.
    set "ACTIVATE_CMD="
)

echo.
echo [1/3] Levantando Worker de Bandejas...
start "Worker Bandejas" cmd /k "%ACTIVATE_CMD% watchdog.bat worker_bandejas.py"

echo [2/3] Levantando Worker de Postura...
start "Worker Postura" cmd /k "%ACTIVATE_CMD% watchdog.bat worker_postura.py"

echo [3/3] Levantando Dashboard Streamlit...
start "Dashboard Vision-MVP" cmd /k "%ACTIVATE_CMD% streamlit run dashboard.py"

echo.
echo ============================================================
echo  [EXITO] Las 3 instancias estan iniciando en nuevas ventanas.
echo  Puedes cerrar esta ventana principal.
echo ============================================================
timeout /t 5 >nul
