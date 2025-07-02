@echo off
setlocal EnableDelayedExpansion

echo Organizing directory structure for radiology-pipeline...

set "PROJECT_ROOT=C:\MO_Part\radiology-report-generation"

:: Create directory structure if missing
echo Checking and creating directories...
mkdir "%PROJECT_ROOT%\config\prompts" 2>nul
mkdir "%PROJECT_ROOT%\data\raw" 2>nul
mkdir "%PROJECT_ROOT%\data\processed" 2>nul
mkdir "%PROJECT_ROOT%\data\ground_truth" 2>nul
mkdir "%PROJECT_ROOT%\data\outputs\primary_reports" 2>nul
mkdir "%PROJECT_ROOT%\data\outputs\evaluation_reports" 2>nul
mkdir "%PROJECT_ROOT%\src\preprocessing" 2>nul
mkdir "%PROJECT_ROOT%\src\generation" 2>nul
mkdir "%PROJECT_ROOT%\src\evaluation" 2>nul
mkdir "%PROJECT_ROOT%\src\utils" 2>nul
mkdir "%PROJECT_ROOT%\scripts" 2>nul

:: Check and move config.yaml
echo Checking for config.yaml...
set "FOUND_CONFIG="
for /r "%PROJECT_ROOT%" %%F in (config.yaml) do (
    if exist "%%F" (
        set "FOUND_CONFIG=%%F"
        if not "%%F"=="%PROJECT_ROOT%\config\config.yaml" (
            echo Moving config.yaml from %%F to %PROJECT_ROOT%\config\config.yaml
            move "%%F" "%PROJECT_ROOT%\config\config.yaml" >nul
            if errorlevel 1 (
                echo Failed to move config.yaml
            ) else (
                echo Successfully moved config.yaml
            )
        ) else (
            echo config.yaml already in correct location: %PROJECT_ROOT%\config\config.yaml
        )
    )
)
if not defined FOUND_CONFIG (
    echo Warning: config.yaml not found in project directory
)

:: Check and move primary_llm_prompt.txt
echo Checking for primary_llm_prompt.txt...
set "FOUND_PROMPT="
for /r "%PROJECT_ROOT%" %%F in (primary_llm_prompt.txt) do (
    if exist "%%F" (
        set "FOUND_PROMPT=%%F"
        if not "%%F"=="%PROJECT_ROOT%\config\prompts\primary_llm_prompt.txt" (
            echo Moving primary_llm_prompt.txt from %%F to %PROJECT_ROOT%\config\prompts\primary_llm_prompt.txt
            move "%%F" "%PROJECT_ROOT%\config\prompts\primary_llm_prompt.txt" >nul
            if errorlevel 1 (
                echo Failed to move primary_llm_prompt.txt
            ) else (
                echo Successfully moved primary_llm_prompt.txt
            )
        ) else (
            echo primary_llm_prompt.txt already in correct location: %PROJECT_ROOT%\config\prompts\primary_llm_prompt.txt
        )
    )
)
if not defined FOUND_PROMPT (
    echo Warning: primary_llm_prompt.txt not found in project directory
)

:: Check and move requirements.txt
echo Checking for requirements.txt...
set "FOUND_REQUIREMENTS="
for /r "%PROJECT_ROOT%" %%F in (requirements.txt) do (
    if exist "%%F" (
        set "FOUND_REQUIREMENTS=%%F"
        if not "%%F"=="%PROJECT_ROOT%\requirements.txt" (
            echo Moving requirements.txt from %%F to %PROJECT_ROOT%\requirements.txt
            move "%%F" "%PROJECT_ROOT%\requirements.txt" >nul
            if errorlevel 1 (
                echo Failed to move requirements.txt
            ) else (
                echo Successfully moved requirements.txt
            )
        ) else (
            echo requirements.txt already in correct location: %PROJECT_ROOT%\requirements.txt
        )
    )
)
if not defined FOUND_REQUIREMENTS (
    echo Warning: requirements.txt not found in project directory
)

:: Verify src\evaluation and src\utils directories
echo Verifying src directories...
if exist "%PROJECT_ROOT%\src\evaluation" (
    echo src\evaluation exists
) else (
    echo Warning: src\evaluation directory missing
)
if exist "%PROJECT_ROOT%\src\utils" (
    echo src\utils exists
) else (
    echo Warning: src\utils directory missing
)

echo Directory structure organization complete.
pause