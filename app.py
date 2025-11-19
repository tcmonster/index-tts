# app.py
import json
import os
import uuid
import torch
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
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


def ensure_qwen_chat_template(tokenizer):
    if tokenizer is None:
        return
    if getattr(tokenizer, "chat_template", None):
        return
    tokenizer.chat_template = QWEN_CHAT_TEMPLATE
    print("Configured fallback chat_template for QwenEmotion tokenizer.")

# —— 加载角色配置 —— #
with open("roles.json", "r", encoding="utf-8") as f:
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
    # 校验输出目录
    if not os.path.isdir(req.output_dir):
        raise HTTPException(status_code=400, detail=f"输出目录 {req.output_dir} 不存在")

    responses = []
    for item in req.items:
        # 若指定角色
        if item.role:
            role_cfg = ROLES.get(item.role)
            if not role_cfg:
                raise HTTPException(status_code=400, detail=f"未知角色: {item.role}")
            speaker_audio = role_cfg["speaker_audio"]
            emotion_text = item.emotion_text or role_cfg["default_emotion_text"]
            emo_alpha = item.emo_alpha or role_cfg["default_emo_alpha"]
        else:
            # 否则必须提供 speaker_audio
            if not item.speaker_audio:
                raise HTTPException(status_code=400, detail="未指定角色，须传 speaker_audio")
            speaker_audio = item.speaker_audio
            emotion_text = item.emotion_text
            emo_alpha = item.emo_alpha

        # 设置输出文件名（带时间戳＋UUID）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = item.output_filename or f"{item.role or 'role'}_{timestamp}_{uuid.uuid4().hex}.wav"
        out_path = os.path.join(req.output_dir, fn)

        # 构造推理参数
        kwargs = {
            "spk_audio_prompt": speaker_audio,
            "text": item.text,
            "output_path": out_path,
            "verbose": False
        }
        if item.duration_tokens:
            kwargs["duration_tokens"] = item.duration_tokens
        if item.emotion_audio:
            kwargs["emo_audio_prompt"] = item.emotion_audio
            kwargs["emo_alpha"] = emo_alpha
        elif emotion_text:
            kwargs["use_emo_text"] = True
            kwargs["emo_text"] = emotion_text
            kwargs["emo_alpha"] = emo_alpha

        # 执行推理
        try:
            tts.infer(**kwargs)
            responses.append({"role": item.role, "output": out_path})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"生成失败: {e}")

    # 合并处理
    if req.combine:
        merged_fn = f"merged_{uuid.uuid4().hex}.wav"
        merged_path = os.path.join(req.output_dir, merged_fn)
        # 这里可用 ffmpeg 或其它方式合并 responses 中所有音频输出
        return {"status": "success", "merged_file": merged_path, "details": responses}

    return {"status": "success", "results": responses}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
