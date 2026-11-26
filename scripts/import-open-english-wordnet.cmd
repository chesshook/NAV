@echo off
setlocal
cd /d "%~dp0\.."
if not exist data mkdir data
set "WN_FILE=data\english-wordnet-2025-plus.xml.gz"
if not exist "%WN_FILE%" (
  echo Downloading Open English WordNet 2025+...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri 'https://en-word.net/static/english-wordnet-2025-plus.xml.gz' -OutFile '%WN_FILE%'"
  if errorlevel 1 (
    echo Download failed. Check the network connection and try again.
    exit /b 1
  )
)
echo Importing the lexical network into SQLite. This is a one-time operation per dictionary version.
py -3 -m mcrl --lexicon-db data\semantic_network.sqlite --import-wordnet-lmf "%WN_FILE%"
if errorlevel 1 (
  echo Semantic dictionary import failed.
  exit /b 1
)
echo.
echo Semantic network ready at data\semantic_network.sqlite
echo To inspect a sentence:
echo   py -3 -m mcrl --lexicon-db data\semantic_network.sqlite --semantic-inspect "The river bank was muddy."
