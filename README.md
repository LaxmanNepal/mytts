# Laxman MyTTS 🇳🇵

A Google Colab text-to-speech workspace for generating **Nepali speech in Laxman's own voice** using zero-shot voice cloning.

## V1 engine

V1 uses the `Oshara/xtts-v2-nepali` XTTS-v2 Nepali fine-tune. The model adds Nepali (`ne`) to XTTS-v2 and accepts a short reference recording for zero-shot voice cloning. The model card reports 24 kHz output and recommends the epoch-10 checkpoint for generalisation.

Model: https://huggingface.co/Oshara/xtts-v2-nepali

## Your reference voice

The project uses:

`voice/my sample.wav`

Raw file URL:

`https://github.com/LaxmanNepal/mytts/raw/refs/heads/main/voice/my%20sample.wav`

The Colab notebook downloads this file automatically, so you do not need to upload the reference audio every time.

## Google Colab

Open:

`colab/Laxman_TTS.ipynb`

Recommended runtime: **GPU**. A T4 is a good starting point.

The notebook provides:

- Automatic GPU detection
- Automatic download of your GitHub reference voice
- Nepali text input
- Voice-cloned WAV generation
- Sentence chunking for longer text
- Audio preview
- WAV download
- Adjustable temperature and repetition penalty
- Basic reference-audio validation

## Important

Use only your own voice or a voice for which you have explicit permission. Do not use the system to impersonate another person or present generated audio as an authentic recording.

## Roadmap

- V1: Nepali zero-shot cloning in Colab
- V1.1: Better text normalisation and long-form narration
- V1.2: MP3 export and silence controls
- V2: Gradio Studio UI with presets
- V3: Compare multiple Nepali cloning engines
- V4: Optional fine-tuning/training workflow if a properly transcribed dataset is prepared
