from pathlib import Path
import re
import torch
import torchaudio
from huggingface_hub import snapshot_download
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

MODEL_ID = "Oshara/xtts-v2-nepali"
MODEL_SUBDIR = "epoch-10"
SAMPLE_RATE = 24000


def download_model():
    root = snapshot_download(MODEL_ID, allow_patterns=[f"{MODEL_SUBDIR}/*"])
    return Path(root) / MODEL_SUBDIR


def load_model(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = download_model()
    config = XttsConfig()
    config.load_json(str(model_dir / "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_path=str(model_dir / "model.pth"),
        vocab_path=str(model_dir / "vocab.json"),
        speaker_file_path=str(model_dir / "speakers_xtts.pth"),
        eval=True,
    )
    if device == "cuda":
        model.cuda()
    else:
        model.cpu()
    return model, config, device


def normalise_nepali(text: str) -> str:
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_text(text: str, max_chars: int = 240):
    text = normalise_nepali(text)
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[।!?])\s+", text)
    chunks, current = [], ""
    for sentence in sentences:
        if not sentence:
            continue
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current.strip())
    return chunks


def synthesize(model, config, reference_wav: str, text: str, output_wav: str,
               temperature: float = 0.65, repetition_penalty: float = 5.0):
    chunks = split_text(text)
    outputs = []
    for chunk in chunks:
        result = model.synthesize(
            chunk,
            config,
            speaker_wav=reference_wav,
            language="ne",
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        outputs.append(torch.tensor(result["wav"]).float())
    gap = torch.zeros(int(SAMPLE_RATE * 0.18))
    pieces = []
    for i, audio in enumerate(outputs):
        pieces.append(audio)
        if i < len(outputs) - 1:
            pieces.append(gap)
    final = torch.cat(pieces).unsqueeze(0)
    torchaudio.save(output_wav, final, SAMPLE_RATE)
    return output_wav
