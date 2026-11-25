@echo off
setlocal
where llama >nul 2>nul
if errorlevel 1 (
  echo llama.cpp was not found in PATH.
  echo Install it with: winget install llama.cpp
  exit /b 1
)
echo Starting the local Qwen3.5-2B model on 127.0.0.1:8080
echo First startup downloads the GGUF model from Hugging Face.
echo CPU mode is selected with -ngl 0. Keep this window open.
llama serve -hf gguf/Qwen3.5-2B-GGUF:Q4_K_M --host 127.0.0.1 --port 8080 -c 8192 -ngl 0
