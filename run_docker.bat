@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo Persona Document Intelligence - Docker Runner
echo ===================================================

:: Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker and try again.
    exit /b 1
)
echo ✅ Docker is running

:: Get command argument
set "command=%~1"
if "%command%"=="" set "command=run"

if "%command%"=="run" (
    call :run_pipeline
) else if "%command%"=="dev" (
    call :run_dev
) else if "%command%"=="logs" (
    call :show_logs
) else if "%command%"=="clean" (
    call :cleanup
) else if "%command%"=="help" (
    call :show_help
) else (
    echo ❌ Unknown command: %command%
    echo Use '%~nx0 help' for usage information
    exit /b 1
)

goto :eof

:run_pipeline
echo 🔨 Building Docker image...
docker-compose build
if errorlevel 1 (
    echo ❌ Failed to build Docker image
    exit /b 1
)

echo 📂 Checking input documents...
if not exist "data\input_documents" (
    echo ⚠️  Warning: data\input_documents directory not found
    echo    Please create the directory and add PDF documents
    exit /b 1
)

dir /b "data\input_documents\*.pdf" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Warning: No PDF documents found in data\input_documents\
    echo    Please add PDF documents to data\input_documents\ directory
    exit /b 1
)

for /f %%i in ('dir /b "data\input_documents\*.pdf" 2^>nul ^| find /c /v ""') do set doc_count=%%i
echo 📄 Found !doc_count! PDF documents

echo 🚀 Running persona document intelligence pipeline...
docker-compose up
if errorlevel 1 (
    echo ❌ Pipeline failed
    exit /b 1
)

echo ✅ Pipeline completed!
echo 📊 Results saved to: data\output\persona_document_intelligence_results.json
goto :eof

:run_dev
echo 🔨 Building Docker image for development...
docker-compose --profile dev build

echo 🚀 Running in development mode...
docker-compose --profile dev up
goto :eof

:show_logs
echo 📋 Showing container logs...
docker-compose logs -f
goto :eof

:cleanup
echo 🧹 Cleaning up Docker containers and images...
docker-compose down --remove-orphans
docker image prune -f
echo ✅ Cleanup completed
goto :eof

:show_help
echo Usage: %~nx0 [command]
echo.
echo Commands:
echo   run     Run the persona document intelligence pipeline (default)
echo   dev     Run in development mode with live code updates
echo   logs    Show container logs
echo   clean   Clean up Docker containers and images
echo   help    Show this help message
echo.
echo Example:
echo   %~nx0 run     # Run the pipeline
echo   %~nx0 dev     # Development mode
goto :eof
