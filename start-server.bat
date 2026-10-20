@echo off
setlocal DisableDelayedExpansion

set "SD_SERVER_PATH=%SD_SERVER_PATH%"
set "MODEL_DIR=./models"
set "COMMANDLINE_ARGS="

if exist "%~dp0start-server.local.bat" (
    call "%~dp0start-server.local.bat"
    if errorlevel 1 exit /b 1
)

if defined SD_SERVER_PATH goto explicit

for %%F in (
    "%~dp0sd-server.exe"
    "%~dp0bin\Release\sd-server.exe"
    "%~dp0bin\sd-server.exe"
    "%~dp0build\bin\Release\sd-server.exe"
    "%~dp0build\bin\sd-server.exe"
) do (
    if exist "%%~F" (
        set "SD_SERVER_PATH=%%~F"
        goto run
    )
)

>&2 echo sd-server.exe was not found. Build it first or set SD_SERVER_PATH to its executable path.
exit /b 1

:explicit
if exist "%SD_SERVER_PATH%" goto run
>&2 echo SD_SERVER_PATH does not exist: "%SD_SERVER_PATH%"
exit /b 1

:run
if "%~1"=="--version" goto version
echo Running: "%SD_SERVER_PATH%" --model-dir "%MODEL_DIR%" %COMMANDLINE_ARGS% %*
"%SD_SERVER_PATH%" --model-dir "%MODEL_DIR%" %COMMANDLINE_ARGS% %*
exit /b %errorlevel%

:version
echo Running: "%SD_SERVER_PATH%" %*
"%SD_SERVER_PATH%" %*
exit /b %errorlevel%
