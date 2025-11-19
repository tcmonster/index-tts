## IndexTTS2 批量语音生成 API 使用文档

本文介绍如何在本地启动 `app.py` 所提供的 FastAPI 服务，并通过 `/tts/batch` 接口批量生成语音。内容覆盖环境准备、角色配置、请求 / 响应说明以及故障排查提示。

---

### 1. 环境准备

1. **依赖安装**：确保已按项目 `README.md` 安装 Python 依赖与 IndexTTS2 权重，可使用 `pip install -e .` 或 `pip install -r requirements.txt`（如提供）。
2. **模型与资源**：将 IndexTTS2 模型权重置于 `checkpoints/indextts2/`，并确认情绪子模型目录（默认 `qwen0.6bemo4-merge/`）完整。
3. **音频资源**：在 `voice_prompts/` 中准备角色参考音色，格式为 WAV，采样率与模型一致。

> `app.py` 启动后会自动将 QwenEmotion 的 tokenizer 绑定一个 Jinja 模板 (`chat_template`)，避免 Hugging Face 报 “tokenizer.chat_template 缺失” 的错误，因此无需改动官方 `indextts` 代码。

---

### 2. 启动 API 服务

执行以下命令之一即可启动服务：

```bash
# 方式一：直接运行 app.py（默认 0.0.0.0:8000）
python app.py

# 方式二：使用 uvicorn（可选热重载）
uvicorn app:app --host 0.0.0.0 --port 8000
```

启动成功后，可访问 `http://127.0.0.1:8000/docs` 查看 Swagger UI 进行调试。

---

### 3. 角色配置 (`roles.json`)

`app.py` 会在启动时读取根目录下的 `roles.json`。每个角色包含：

```json
{
  "Narrator_male": {
    "speaker_audio": "voice_prompts/narrator_male.wav",
    "default_emotion_text": "neutral",
    "default_emo_alpha": 0.8
  },
  "Narrator_female": {
    "speaker_audio": "voice_prompts/narrator_female.wav",
    "default_emotion_text": "neutral",
    "default_emo_alpha": 1.0
  }
}
```

请求中指定 `role` 时，会自动使用对应的 `speaker_audio` 与默认情绪。若未指定角色，需显式提供 `speaker_audio`（自定义音色）。

---

### 4. 接口说明

- **URL**：`POST /tts/batch`
- **内容类型**：`application/json`

#### 4.1 请求体结构

```json
{
  "output_dir": "outputs",
  "items": [
    {
      "role": "Narrator_male",
      "speaker_audio": "voice_prompts/custom.wav",
      "emotion_audio": "voice_prompts/emo.wav",
      "emotion_text": "happy",
      "text": "你好，我是角色 A。",
      "duration_tokens": 300,
      "emo_alpha": 0.9,
      "output_filename": "roleA.wav"
    }
  ],
  "combine": false
}
```

字段 | 类型 | 必填 | 说明
---|---|---|---
`output_dir` | string | 是 | 输出目录，必须存在且具备写权限。
`items` | array | 是 | 每个元素对应一次语音生成任务。
`items[].role` | string | 否 | 角色名称。存在角色时可省略 `speaker_audio`。
`items[].speaker_audio` | string | 条件 | 未指定 `role` 时必填，指向参考音色 WAV。
`items[].emotion_audio` | string | 否 | 情绪参考音频，优先级高于 `emotion_text`。
`items[].emotion_text` | string | 否 | 情绪文字提示（如 `neutral`、`happy` 等）。
`items[].text` | string | 是 | 要合成的文本。
`items[].duration_tokens` | int | 否 | 控制时长的 token 数。
`items[].emo_alpha` | float | 否 | 情绪强度，默认 1.0。
`items[].output_filename` | string | 否 | 指定输出文件名，默认使用时间戳+UUID。
`combine` | bool | 否 | 若为 `true`，所有结果合并成单一音频并返回 `merged_file`。

#### 4.2 响应结构

- **成功（未合并）**

```json
{
  "status": "success",
  "results": [
    {"role": "Narrator_male", "output": "outputs/roleA.wav"}
  ]
}
```

- **成功（合并）**

```json
{
  "status": "success",
  "merged_file": "outputs/merged_xxx.wav",
  "details": [
    {"role": "Narrator_male", "output": "outputs/part1.wav"},
    {"role": "Narrator_female", "output": "outputs/part2.wav"}
  ]
}
```

- **失败**（示例）

```json
{
  "detail": "生成失败: <错误信息>"
}
```

---

### 5. 请求示例

#### 5.1 使用角色配置

```bash
curl -X POST "http://127.0.0.1:8000/tts/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "output_dir": "outputs",
    "items": [
      {
        "role": "Narrator_male",
        "text": "你好，我是角色A，欢迎来到我们的故事。",
        "output_filename": "roleA_welcome.wav"
      }
    ]
  }'
```

#### 5.2 自定义音色 + 情绪文字

```bash
curl -X POST "http://127.0.0.1:8000/tts/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "output_dir": "outputs",
    "items": [
      {
        "speaker_audio": "voice_prompts/custom_voice.wav",
        "text": "今天我要讲一个很开心的故事。",
        "emotion_text": "happy",
        "emo_alpha": 0.9,
        "output_filename": "custom_happy_story.wav"
      }
    ]
  }'
```

#### 5.3 生成后合并

```bash
curl -X POST "http://127.0.0.1:8000/tts/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "output_dir": "outputs",
    "items": [
      {
        "role": "Narrator_male",
        "text": "第一句。",
        "output_filename": "part1.wav"
      },
      {
        "role": "Narrator_female",
        "text": "第二句。",
        "output_filename": "part2.wav"
      }
    ],
    "combine": true
  }'
```

---

### 6. 常见问题 & 提示

- **输出目录不存在**：接口会直接报 400。请预先创建 `output_dir` 指定的文件夹。
- **角色找不到**：返回 400。确认 `roles.json` 中存在该角色且服务已重启以加载新配置。
- **情绪提示报错**：`app.py` 已在启动时自动修复 QwenEmotion 的 `chat_template`，若仍报错请检查 `checkpoints/indextts2/qwen0.6bemo4-merge/` 是否完整。
- **文本过长**：建议客户端自行分段再批量请求，避免显存不足或耗时过长。
- **文件权限**：确保服务进程对 `output_dir` 拥有写权限，尤其在 Docker 或远程主机上部署时。

---

如需进一步扩展（例如新增 Web UI、异步队列或使用 GPU/Deepspeed 版本），可在 FastAPI 基础上继续封装；该文档聚焦于当前 `app.py` 所提供的最小可用 API。
