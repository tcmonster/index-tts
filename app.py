# app.py
import contextlib
import json
import os
import threading
import uuid
import wave
from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app_cache import PromptCache, warmup_roles
from indextts.infer_v2 import IndexTTS2, QwenEmotion

QWEN_CHAT_TEMPLATE = """{% for message in messages %}
{% if message['role'] == 'system' %}
{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}
{% elif message['role'] == 'user' %}
{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}
{% elif message['role'] == 'assistant' %}
{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}
{% else %}
{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
{{ '<|im_start|>assistant\n' }}
{% endif %}"""


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROLES_PATH = os.path.join(BASE_DIR, "roles.json")
EMO_TEXT_CACHE: Dict[str, List[float]] = {}
EMO_TEXT_LOCK = threading.Lock()
prompt_cache = PromptCache(max_speakers=8, max_emotions=8)


def ensure_qwen_chat_template(tokenizer):
    if tokenizer is None:
        return
    if getattr(tokenizer, "chat_template", None):
        return
    tokenizer.chat_template = QWEN_CHAT_TEMPLATE
    print("Configured fallback chat_template for QwenEmotion tokenizer.")


def _resolve_audio_path(path: str) -> str:
    if not path:
        raise FileNotFoundError("未提供音频路径")
    candidate = path if os.path.isabs(path) else os.path.join(BASE_DIR, path)
    candidate = os.path.abspath(candidate)
    if not os.path.isfile(candidate):
        raise FileNotFoundError(f"音频文件不存在: {path}")
    return candidate


def _resolve_output_dir(path: str) -> str:
    candidate = path if os.path.isabs(path) else os.path.join(BASE_DIR, path)
    return os.path.abspath(candidate)


def _normalize_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _build_group_key(speaker_audio: str, emotion_audio: Optional[str], emotion_text: Optional[str]) -> Tuple[str, Tuple[str, str]]:
    if emotion_audio:
        return speaker_audio, ("audio", emotion_audio)
    text_key = emotion_text or ""
    return speaker_audio, ("text", text_key)


def get_or_cache_emotion_vector(emotion_text: Optional[str]) -> Optional[List[float]]:
    normalized = _normalize_text(emotion_text)
    if not normalized:
        return None
    with EMO_TEXT_LOCK:
        cached = EMO_TEXT_CACHE.get(normalized)
    if cached is not None:
        return cached
    emo_dict = tts.qwen_emo.inference(normalized)
    vector = list(emo_dict.values())
    with EMO_TEXT_LOCK:
        EMO_TEXT_CACHE[normalized] = vector
    return vector


def combine_wav_files(inputs: List[str], output_path: str) -> str:
    if not inputs:
        raise ValueError("没有可合并的音频")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with contextlib.ExitStack() as stack:
        readers = [stack.enter_context(wave.open(path, "rb")) for path in inputs]
        params = readers[0].getparams()
        for reader in readers[1:]:
            if reader.getparams()[:3] != params[:3]:
                raise ValueError("音频参数不一致，无法合并")
        with wave.open(output_path, "wb") as writer:
            writer.setparams(params)
            for reader in readers:
                writer.writeframes(reader.readframes(reader.getnframes()))
    return output_path


def prepare_request_item(idx: int, item: "Item", output_dir: str) -> Dict:
    if item.role:
        role_cfg = ROLES.get(item.role)
        if not role_cfg:
            raise ValueError(f"未知角色: {item.role}")
        speaker_audio = _resolve_audio_path(role_cfg["speaker_audio"])
        emotion_audio_override = role_cfg.get("emotion_audio")
        emotion_text = _normalize_text(item.emotion_text or role_cfg.get("default_emotion_text"))
        emo_alpha = item.emo_alpha if item.emo_alpha is not None else role_cfg.get("default_emo_alpha", 1.0)
    else:
        if not item.speaker_audio:
            raise ValueError("未指定角色，须传 speaker_audio")
        speaker_audio = _resolve_audio_path(item.speaker_audio)
        emotion_audio_override = None
        emotion_text = _normalize_text(item.emotion_text)
        emo_alpha = item.emo_alpha if item.emo_alpha is not None else 1.0

    # Emotion priority: explicit audio > role override audio > emotion text > fallback to speaker
    if item.emotion_audio:
        emotion_audio = _resolve_audio_path(item.emotion_audio)
        emotion_vector = None
        emotion_text_for_group = None
    elif emotion_audio_override:
        emotion_audio = _resolve_audio_path(emotion_audio_override)
        emotion_vector = None
        emotion_text_for_group = None
    elif emotion_text:
        emotion_audio = None
        emotion_vector = get_or_cache_emotion_vector(emotion_text)
        emotion_text_for_group = emotion_text
    else:
        emotion_audio = speaker_audio
        emotion_vector = None
        emotion_text_for_group = None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = item.role or "custom"
    fn = item.output_filename or f"{base_name}_{timestamp}_{uuid.uuid4().hex}.wav"
    output_path = os.path.join(output_dir, fn)

    group_key = _build_group_key(speaker_audio, emotion_audio, emotion_text_for_group)
    return {
        "idx": idx,
        "role": item.role,
        "text": item.text,
        "speaker_audio": speaker_audio,
        "emotion_audio": emotion_audio,
        "emotion_text": emotion_text_for_group,
        "emo_vector": emotion_vector,
        "emo_alpha": emo_alpha if emo_alpha is not None else 1.0,
        "duration_tokens": item.duration_tokens,
        "output_path": output_path,
        "group_key": group_key,
    }


def group_prepared_items(prepared: List[Dict]) -> Tuple["OrderedDict[Tuple[str, Tuple[str, str]], List[Dict]]", List[Tuple[str, Tuple[str, str]]]]:
    grouped: "OrderedDict[Tuple[str, Tuple[str, str]], List[Dict]]" = OrderedDict()
    order: List[Tuple[str, Tuple[str, str]]] = []
    for entry in prepared:
        key = entry["group_key"]
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(entry)
    return grouped, order


# —— 加载角色配置 —— #
with open(ROLES_PATH, "r", encoding="utf-8") as f:
    ROLES = json.load(f)

# —— 模型初始化 —— #
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

model_dir = "checkpoints/indextts2"
cfg_path = os.path.join(model_dir, "config.yaml")

# 情绪子目录
emo_subdir = "qwen0.6bemo4-merge"
emo_path = os.path.join(model_dir, emo_subdir)
print("Setting emotion tokenizer path:", emo_path)

tts = IndexTTS2(
    cfg_path=cfg_path,
    model_dir=model_dir,
    use_fp16=False,
    use_cuda_kernel=False,
    use_deepspeed=False
)

# 覆盖情绪子模块
tts.qwen_emo = QwenEmotion(emo_path)
ensure_qwen_chat_template(getattr(tts.qwen_emo, "tokenizer", None))

# 若模型有子模块需移至 device
try:
    tts.s2mel = tts.s2mel.to(device)
    tts.gpt = tts.gpt.to(device)
except AttributeError:
    pass

tts.device = device
print("Model initialized.")

try:
    warmup_roles(tts, ROLES, prompt_cache, _resolve_audio_path, verbose=True)
except Exception as exc:
    print(f"[prompt_cache] Warmup skipped: {exc}")

# —— FastAPI 服务 —— #
app = FastAPI(title="Batch TTS Service")

class Item(BaseModel):
    role: Optional[str] = None
    speaker_audio: Optional[str] = None
    emotion_audio: Optional[str] = None
    emotion_text: Optional[str] = None
    text: str
    duration_tokens: Optional[int] = None
    emo_alpha: Optional[float] = 1.0
    output_filename: Optional[str] = None

class BatchRequest(BaseModel):
    items: List[Item]
    output_dir: str
    combine: Optional[bool] = False

@app.post("/tts/batch")
def tts_batch(req: BatchRequest):
    output_dir = _resolve_output_dir(req.output_dir)
    if not os.path.isdir(output_dir):
        raise HTTPException(status_code=400, detail=f"输出目录 {req.output_dir} 不存在")

    prepared_items = []
    for idx, item in enumerate(req.items):
        try:
            prepared = prepare_request_item(idx, item, output_dir)
            prepared_items.append(prepared)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    grouped, order = group_prepared_items(prepared_items)
    responses = []

    for key in order:
        for entry in grouped[key]:
            prompt_cache.inject_speaker(tts, entry["speaker_audio"])
            if entry["emotion_audio"]:
                prompt_cache.inject_emotion(tts, entry["emotion_audio"])

            kwargs = {
                "spk_audio_prompt": entry["speaker_audio"],
                "text": entry["text"],
                "output_path": entry["output_path"],
                "verbose": False,
            }
            if entry["duration_tokens"]:
                kwargs["duration_tokens"] = entry["duration_tokens"]
            if entry["emo_vector"] is not None:
                kwargs["emo_vector"] = entry["emo_vector"]
                kwargs["emo_alpha"] = entry["emo_alpha"]
            elif entry["emotion_audio"]:
                kwargs["emo_audio_prompt"] = entry["emotion_audio"]
                kwargs["emo_alpha"] = entry["emo_alpha"]

            try:
                tts.infer(**kwargs)
                prompt_cache.capture_from_tts(tts)
                responses.append({
                    "idx": entry["idx"],
                    "role": entry["role"],
                    "output": entry["output_path"],
                })
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"生成失败: {exc}")

    ordered = sorted(responses, key=lambda x: x["idx"])
    ordered_payload = [{"role": r["role"], "output": r["output"]} for r in ordered]

    if req.combine:
        merged_fn = f"merged_{uuid.uuid4().hex}.wav"
        merged_path = os.path.join(output_dir, merged_fn)
        try:
            combine_wav_files([entry["output"] for entry in ordered], merged_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"合并失败: {exc}")
        return {"status": "success", "merged_file": merged_path, "details": ordered_payload}

    return {"status": "success", "results": ordered_payload}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
