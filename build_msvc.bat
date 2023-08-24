@echo OFF

REM Determine if this batch file was launched by double-clicking it from Explorer.
REM If it was, we will pause before exiting so the user can read the output.
set explorer_launched=0
for %%x in (%cmdcmdline%) do if /i "%%~x"=="/c" set explorer_launched=1

REM Make sure the Visual Studio compiler is present
where cl.exe > nul 2> nul
if %errorlevel% neq 0 (
    echo Unable to find `cl.exe`, the Visual Studio compiler.
    echo Please make sure Visual Studio is installed and in your PATH environment variable.
    echo if Visual Studio is installed, you may have to launch "x64 Native Tools Command Prompt for VS".
    if "%explorer_launched%" == "1" pause
    exit /b 1
)

REM  Make sure that the x64 compiler is selected by looking for "for x86" in cl.exe's statement of its version.
for /f "delims=" %%a in ('cl.exe 2^>^&1') do (
    setlocal enabledelayedexpansion
    set "line=%%a"
    if "!line!" neq "!line:for x86=!" (
        echo The 32-bit Visual Studio tools are active, but Stable Diffusion cannot allocate enough memory in 32-bit mode.
        echo Please make sure the Visual Studio tools targeting x64 are active.
        echo You may have to launch "x64 Native Tools Command Prompt for VS".
        if "%explorer_launched%" == "1" pause
        exit /b 1
    )
    endlocal
)

cl.exe /fp:fast /Ox /EHsc /arch:AVX2 /GL /I. /I ggml\include\ggml /I ggml\include main.cpp stable-diffusion.cpp ggml\src\ggml.c /link /out:sd.exe