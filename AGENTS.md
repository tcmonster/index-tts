# AGENTS：本地 API 开发与维护指南

本仓库以官方 `index-tts` 为上游，仅在其基础之上新增本地化 API 服务相关文件，确保后续可无痛与官方版本同步。以下内容说明自定义入口、开发规范以及与上游仓库协同的流程。

## 1. 目录角色与边界

| 分类 | 说明 |
| --- | --- |
| 官方代码（请视为只读） | `indextts/`, `archive/`, `assets/`, `checkpoints/`, `docs/`, `examples/`, `tests/`, `tools/`, `webui.py`, `pyproject.toml`, `uv.lock` 等上游文件夹与脚本。更新来自官方版本时，用原样覆盖。 |
| 本地扩展（允许新增/修改） | `app.py`, `roles.json`, `API.md`, `scripts/` 下的脚本、`voice_prompts/` 中的参考音频、`outputs/` 等运行产物。所有新功能请放在该区域或在其下再建子模块，禁止直接修改官方目录中的源码。 |

> **原则：**向官方代码提交 PR 时，也仅同步本地新增内容；如确需调整官方模块，优先通过包装器或 Monkey Patch 的方式在 `app.py` 或自建模块中完成。

## 2. 本地自定义入口

### 2.1 `app.py`

- 负责启动 FastAPI 服务（默认 `POST /tts/batch`），并在模块顶部加载一次 `IndexTTS2` 模型。该实例会被所有请求复用，避免重复占用显存。
- 初始化逻辑会：  
  1. 加载 `checkpoints/indextts2/config.yaml`；  
  2. 选择 `mps`（若可用）或 CPU；  
-  3. 重载情绪模型目录 `qwen0.6bemo4-merge/`，并使用 `ensure_qwen_chat_template` 给 tokenizer 补上 chat template（解决官方模型缺省模板的问题）。  
-  4. 初始化 `PromptCache`，对角色默认音色/情绪做一次预热（仅编码参考音频，不做推理），提升后续批量调用的命中率。
- `/tts/batch` 请求体映射到 `BatchRequest -> List[Item]`。服务会先根据 `(speaker_audio, emotion_source)` 对条目分组，再按分组顺序串行推理，确保命中的参考音频不会被频繁清空；推理结果依旧按原请求顺序返回。若传入 `combine=true`，会按原顺序将所有 WAV 直接级联成一个新文件，并把每条明细放在 `details` 字段，方便排查。
- 如需新增路由、鉴权、队列等扩展，请全部写在 `app.py` 或其同级新模块（例如 `app_utils/`），并通过 `from .app_utils import ...` 的方式引入，避免触碰 `indextts/` 源码。

### 2.2 `roles.json`

- 维护角色到音色/情绪默认值的映射，示例：

  ```json
  {
    "Narrator_male": {
      "speaker_audio": "voice_prompts/narrator_male.wav",
      "default_emotion_text": "neutral",
      "default_emo_alpha": 0.8
    }
  }
  ```
- 新增角色时务必：  
  1. 将参考音频放到 `voice_prompts/`；  
  2. 在 `roles.json` 中添加配置；  
  3. 重启 `app.py` 以重新加载角色列表；  
  4. 在 `API.md` 的「角色配置」部分补充说明。

### 2.3 `API.md`

- 描述 `/tts/batch` 的字段、响应示例及调试方法。若接口、字段名、可选值或业务流程有变化，必须同步更新此文档，确保调用方无需翻阅源码即可接入。
- 建议每次改动 API 行为后同时在 PR 描述中引用对应章节，方便审核。

### 2.4 `scripts/`

| 脚本 | 用途 |
| --- | --- |
| `scripts/test_mps.py` | 验证当前 PyTorch 是否启用 MPS，排查 Mac 上的训练/推理环境问题。 |
| `scripts/test_model.py` | 最小化调用 `IndexTTS2.infer`，用于检测权重、情绪模型路径与音频输出，开发新功能前后都应运行一次。 |
| `scripts/test_api.py` | 极简 FastAPI 示例，方便在不加载大模型的情况下验证反向代理、鉴权或部署脚本。 |

可以继续在 `scripts/` 下添加新的诊断或部署脚本，但不要移动或覆盖 `tools/`、`tests/` 等官方目录。

### 2.5 `app_cache.py`

- 维护 `PromptCache`（LRU 缓存），缓存 `IndexTTS2` 的 `cache_spk_*`、`cache_emo_*` 张量，并能在推理前自动注入、推理后捕获，避免重复读取与特征提取。
- 提供 `build_speaker_cache`/`build_emotion_cache` 用于启动时的角色预热，不会触碰官方源码。
- 如需调整缓存大小、淘汰策略或扩展新的 prompt 类型，只需在该文件内修改即可。

## 3. 开发流程

1. **环境准备**  
   - 安装 `uv`（或 `pip`）后执行 `uv sync --all-extras`，并运行 `git lfs install && git lfs pull` 以拉取权重。  
   - 将官方提供的 IndexTTS2 模型放到 `checkpoints/indextts2/`，情绪子目录需包含 `qwen0.6bemo4-merge/`。
2. **启动服务**  
   - 本地调试可用 `python app.py` 或 `uvicorn app:app --host 0.0.0.0 --port 8000`。  
   - 访问 `http://127.0.0.1:8000/docs` 使用 Swagger 调试。
3. **验证**  
   - 修改模型或硬件相关逻辑后先运行 `python scripts/test_mps.py`、`python scripts/test_model.py`。  
   - 修改接口后用 `curl`/`httpie` 或 `scripts/test_api.py` 做基本回归，再按照 `API.md` 的示例调用一次 `/tts/batch`。
4. **交付前 checklist**  
   - `roles.json`、`voice_prompts/`、`outputs/` 中的敏感或占空间的内容是否需要清理？  
   - `API.md` 是否覆盖了新增字段/返回值？  
   - 是否记录了需要手工执行的部署步骤（可写在 PR 或 README 的「本地 API」段落）？

## 4. 新功能规范（保持与官方兼容）

- **仅新增，不替换**：若要修改官方逻辑，请在 `app.py` 或新建模块中通过函数封装、子类或 Monkey Patch 的方式覆盖，不要直接编辑 `indextts/*`、`tests/*` 等文件。
- **集中管理配置**：与角色、音色、API 令牌相关的配置统一写在 `roles.json` 或新建 `config/*.yaml`，并在 `app.py` 动态读取。避免把硬编码散落在官方源文件里。
- **模型交互层**：如需对 `IndexTTS2` 增加缓存、异步推理、排队等功能，创建 `app_agent.py` / `services/tts_service.py` 等新模块来封装，再由 `app.py` 调用，确保上游更新后无需解决冲突。
- **文档同步**：新增路由、响应字段或脚本后必须同步更新 `API.md`（对外）及本文件（对内），让后续维护者明确扩展范围。
- **可恢复性**：任何对官方文件的临时改动都要通过补丁或脚本生成，不得直接打开官方文件编辑；需要的话在 `scripts/` 中新增自动化脚本来应用这些补丁。

## 5. 与官方仓库同步

1. 设置上游：

   ```bash
   git remote add upstream https://github.com/index-tts/index-tts.git
   ```

2. 每次同步前提交/暂存本地新增文件，然后执行：

   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main --no-commit
   ```

3. 如果官方目录有冲突，直接选择官方版本（因为本地不会修改这些文件）；本地代码只会在 `app.py`、`roles.json`、`API.md`、`scripts/` 等目录产生差异。
4. 合并后重新运行第 3 节中的验证步骤，并在 PR 描述中注明使用的官方版本 tag 或 commit。

## 6. 常见问题

- **QwenEmotion 缺少 chat template**：`app.py` 已在初始化时调用 `ensure_qwen_chat_template`，若仍报错，请确认情绪模型目录完整并清理旧的缓存。
- **Combine 需要同构 WAV**：服务在最后一步直接串联 WAV 数据，请确保同一批次的采样率、声道和位宽一致；否则 `combine_wav_files` 会抛错并返回 500。
- **批量交替角色**：请求会先按 `(speaker_audio, emotion_audio|emotion_text)` 分组后再推理，以提高命中率；服务完成后仍按原索引顺序返回，后续合并或下游流程无需改动。
- **显存/算力不足**：`app.py` 默认优先使用 `mps`，必要时可在 `app.py` 中新增环境变量（如 `FORCE_DEVICE`）的读取逻辑，但仍应放在自定义区域而不是修改 `indextts/infer_v2.py`。
- **角色未识别**：检查 `roles.json` 是否保存为 UTF-8，文件更新后需要重启服务；可在日志中打印 `ROLES.keys()` 来快速定位。

---

按照上述约定新增功能，即可保证本地 API 服务在保持灵活性的同时，与官方 IndexTTS 仓库的代码更新保持兼容。若需新增约定，请在本文件补充说明并在 PR 中提醒所有贡献者同步遵守。
