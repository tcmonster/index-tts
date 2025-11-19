import torch
import os
from indextts.infer_v2 import IndexTTS2

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

model_dir = "checkpoints/indextts2"
cfg_path = os.path.join(model_dir, "config.yaml")
# 子目录名
emo_path = "qwen0.6bemo4-merge"

# 打印确认路径
print("Expected emotion tokenizer path:", os.path.join(model_dir, emo_path))

tts = IndexTTS2(
    cfg_path=cfg_path,
    model_dir=model_dir,
    use_fp16=False,
    use_cuda_kernel=False,
    use_deepspeed=False
)

# 强制将子模块 tokenizer 指定到子目录
from indextts.infer_v2 import QwenEmotion
tts.qwen_emo = QwenEmotion(os.path.join(model_dir, emo_path))  # 强制覆盖

tts.s2mel = tts.s2mel.to(device)
tts.gpt = tts.gpt.to(device)
tts.model_dir = model_dir
tts.device = device

text = "你好，这是一段测试文本。"
wav_path = "output.wav"
tts.infer(
    spk_audio_prompt="examples/voice_01.wav",
    text=text,
    output_path=wav_path,
    verbose=True
)

print("生成完毕，输出位于", wav_path)