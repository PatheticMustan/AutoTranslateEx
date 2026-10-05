# AutoTranslateEx

A Chrome extension and a local Python server that translate the Chinese speech bubbles on `mycomic.com` chapter pages into English and draw the translation over the original art. OCR and translation run locally (RapidOCR + llama.cpp / CTranslate2).

Requires Windows 10/11, Python 3.10+, and Chrome. A GPU (NVIDIA or Intel Arc) is optional but recommended.

## Install

```powershell
cd server
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m atx.models   # downloads llama.cpp, OCR and translation models
```

Load the extension:

1. Open `chrome://extensions/` and turn on **Developer mode**.
2. Click **Load unpacked** and select the `extension/` folder.

## Run

```powershell
cd server
.\.venv\Scripts\python.exe -m atx.app
```

The server listens on `http://127.0.0.1:8765`. Open a chapter on `mycomic.com/chapters/<id>`, and pages are translated as you scroll. Use the extension popup to turn it on or off, pick a tier (Very quick / Quick / Accurate), and edit character names.

## Test

```powershell
cd server
.\.venv\Scripts\python.exe -m pytest tests
```
