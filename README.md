# voice-lab：角色台词的重配与实时替换

一套 Windows 上的语音处理工具，做两件事：

1. **离线重配**：把一个角色的一组语音台词，用一位目标说话人的声音重新配出来，并尽量保留原句的语气、节奏和停顿。
2. **实时替换**：程序播放这些台词时，把原声换成重配版本。只处理系统输出的声音，不修改程序文件，也不读写进程内存。

主要技术点：
- 人声分离和目标说话人片段提取（分离、切片、转写、声纹与说话习惯筛选）；
- 两条生成路线：分句重配（IndexTTS-2）+ RVC，以及在目标说话人数据上微调的 GPT-SoVITS；
- 48 kHz 超分；
- 约 200 ms 延迟的流式识别与替换，可同时处理多个角色；
- 复盘工具，以及每次改动都有记录的改动日志。

## 合规与使用限制（请先读）

- **只用你拥有或已获得授权的声音。** 用真实人物（包括配音演员）的声音训练模型或生成语音，可能涉及声音权、人格权、著作权和平台协议。训练、生成之前，必须取得声音权利人的授权。
- **本仓库只有通用脚本。** 不含任何音频素材、训练好的模型或生成的成品，也不含作者本地实验所用的作品、角色、演员和片源信息。
- **生成的模型和音频只在本机做技术研究。** 不要发布、分享或出售，不要用于冒充任何人，也不要拿去误导他人。
- **不提供以下内容**：下载素材；解密、修改或重新打包游戏文件；注入进程、读写进程内存；绕过签名校验或反作弊。把成品重新编码、打包回游戏的工具不在公开版里。

## 一键启动

双击根目录的 `一键启动.cmd`，会打开一个本地窗口（`scripts/ui_server.py` + `scripts/ui/index.html`）。它只监听 127.0.0.1，每次启动都会生成随机令牌，用 Edge 应用窗口打开。`一键启动.cmd -Menu` 打开同样功能的文字菜单（`scripts/launch.ps1`）。

- **实时替换**：
  - 开始和停止，运行中显示替换句数、正在播放的句子、处理耗时、缓冲和设备；
  - **角色与声线**：每个角色一行，列出有成品的声线和版本，非主角色可以选“不替换”；
  - **队伍开关**：不在场的角色不参与匹配。
- **环境**：检查和深度检查，可以一键修复：补齐 uv、第三方仓库、Python 环境和模型。
- **试听页面**、**快照与恢复**：保存环境快照，先预览再恢复。

也可以带参数直接运行，例如 `-Check`、`-Setup`、`-Snapshot`、`-Restore -DryRun`。启动器不改 Windows 设置，只在实时替换时切换程序的输出设备；也不执行从网上下载的安装脚本。

## 它怎么做的

| 阶段 | 脚本 | 内容 |
|---|---|---|
| 素材 | `run_voice_data.ps1`、`prep_voice.py` | 提取音轨 → 分离人声（Mel-Band RoFormer）→ 切片（Silero VAD）→ 转写（Whisper large-v3）→ 找出目标说话人的片段（声纹 + 说话习惯弱标签分类器）→ 去混响、降噪 → 训练集与参考音频 |
| 数据检查 | `speaker_embed.py`、`clip_purity.py`、`transcript_check.py` | 比较声纹模型（ECAPA、CAM++、WavLM）；检查片段里有没有混进别人的话；用第二个转写模型复核转写 |
| 台词 | `prep_source.py`、`prep_set.py`、`scan_lines.py`、`segment_lines.py` | 读入要替换的台词，建台词集；分成普通台词、需精修、拟声词（闷哼、喘气等）；在原句停顿处分句 |
| 路线一：分句重配 | `run_d4.ps1` | 每个短句用 IndexTTS-2 生成多个候选（情绪匹配的参考 + 原句作情感提示），按语调、读音、时长、音区和衔接挑选，按原句时间拼接，再用 RVC 统一音色；`rvc_guard.py` 修正被 RVC 带偏的音高 |
| 路线二：微调 TTS | `gsv.py`、`tts_lines.py`、`tts_pick.py`、`tts_segments.py`、`keypoints.py` | 在目标说话人数据上微调 GPT-SoVITS；整句生成，每句按原句情绪挑风格参考，按音色、情绪、读音、语调关键点挑选，可以按原句的语速和停顿重新对时 |
| 评估 | `tts_testset.py`、`tts_eval.py`、`metrics.py` | 留出测试集，量音色相似度、读音错误率、时长、语调还原度；另有盲听页 |
| 拟声词 / 精修 | `run_nonverbal.ps1`、`run_refine.ps1` | 拟声词有原声、RVC、升调 RVC、TTS 四种版本，自动检查后给出选择页；问题句逐短句生成候选，在页面上按耳朵挑选 |
| 48 kHz | `superres.py`、`postprocess.py` | AudioSR 超分，去爆音，响度对齐原句 |
| 录屏替换 | `replace_in_recording.py` | 在录屏里找到台词（匹配滤波 + 能量解释验证），去掉原声，插入新版本 |
| 实时替换 | `live.ps1` | WASAPI 回环截获 → 流式识别 → 去除原声并插入 → 播放，见下文 |

## 实时替换

```powershell
.\scripts\live.ps1 -Voice myvoice                                    # 主角色换成 myvoice 的成品
.\scripts\live.ps1 -Voice myvoice -Also other:all:voice2:final_48k   # 同一次会话里再替换另一个角色
```

- **输出设备自动切换**（`app_route.py`，需先设置 `VL_GAME_PROCESS=<程序名>.exe`）：
  - 运行时，把程序的输出切到虚拟设备（Steam Streaming Speakers 或 VB-CABLE）；
  - 退出时切回原来的设备；
  - 程序先关了，就等它下次启动时再切回；
  - 加 `-NoRoute` 关掉自动切换。
- **多个角色**：`-Also <角色>:<台词集>:<声线>:<版本>` 可以重复加。附加角色的匹配门槛更高，因为它可能根本不在场。
- **按在场角色开关**：可以用窗口开关、热键 Ctrl+Alt+1…9，或者屏幕识别（`party_screen.py`，用你自己截的头像模板，只截屏幕上的一小块，不保存图片）。看不到队伍栏时，所有角色都打开。
- **判断规则**（`live_core.py`）：
  - 每个候选在 100～1500 ms 的几个检查点上重新判断；
  - 1 秒以上的台词，听到的越少，要求的匹配度越高；
  - 开头有长轻声前奏的台词，要听到主体才判断；
  - 这些规则都是按实测复盘的数据调出来的，改动前后的数字记在改动日志里。
- **热键**：
  - `~`：标记“这里听着不对”；
  - 空格：记录刚刚过去的那段语音；
  - Ctrl+Alt+B：旁路；
  - Ctrl+Alt+Q：退出。
- **复盘**：
  - 每次会话在 `logs/live/<时间>/` 留下事件、截获的原声和实际输出；
  - `live_review.py` 生成复盘页；
  - 深度检查会在录音上跑离线识别，逐句列出漏换、可能错换和偏晚。

## 多角色和多声线

角色、台词集、声线分别由 `VL_CHAR`、`VL_SET`、`VL_VOICE` 决定（`scripts/vl_paths.py`）。PowerShell 脚本的 `-Char`、`-Set`、`-Voice` 会自动设置它们。结果按组合分文件夹，互不覆盖：

| 组合 | `exp/`、`eval/`、`out/` 下的文件夹 |
|---|---|
| 默认角色 × 台词集 × 默认声线 | `<集>` |
| 默认角色 × 台词集 × 其他声线 | `<集>_<声线>` |
| 其他角色 × 台词集 × 声线 | `<角色>_<集>_<声线>` |

每个文件夹里的“组”是一种生成方式的完整结果，例如 `final_48k`、`final_gsv_a_48k`。每组都有 `final.csv`，记录每句来自哪里。启动器和实时工具按组选版本。

## 环境

- **系统与硬件**：Windows 10/11；NVIDIA 显卡 8 GB 显存以上（推荐 16 GB）；内存 16 GB，并配置足够的页面文件。重任务一次只跑一个。
- **独立的 Python 环境**（包版本见 `requirements/`）：
  - `envs/tools`：处理、评估、实时工具；
  - IndexTTS-2：`repos/index-tts`，用上游的 `uv` 环境；
  - Applio（RVC）：放在 `$VL_CACHE\Applio`，打上 `patches/applio-train-workers.patch`，可以减少训练时的内存占用；
  - GPT-SoVITS：放在 `$VL_CACHE\GPT-SoVITS`，环境 `$VL_CACHE\envs\gptsovits`（说明见 `gsv.py` 开头）；
  - AudioSR：`envs/audiosr`；
  - Seed-VC（可选）：`envs/seedvc`。
- **外部工具**：ffmpeg；vgmstream 用来读取 `.wem` 格式的输入台词。
- **路径**：
  - 环境变量 `VL_CACHE` 要指向一个**纯英文路径**，例如 `C:\voice-lab-cache`，因为有些组件打不开含非 ASCII 字符的路径；
  - 每个 PowerShell 窗口先执行 `. .\scripts\env.ps1`；
  - 从别的终端运行脚本时，先把 `HF_HOME` 设到仓库的 `cache/hf`，否则模型会下载到系统盘。

## 数据放置

```
data/source/wem/                   默认角色的台词（.wem / .wav），由你自己合法导出
data/<角色>/wem/Media/              其他角色的台词
data/<voice>/voice.json            目标说话人的设定，见 examples/voice.example.json
<录音文件夹>                         目标说话人的授权录音，原地读取，不复制
```

## 基本流程

```powershell
. .\scripts\env.ps1                                                 # $T = envs\tools\Scripts\python.exe
& $T scripts/prep_source.py                                         # 台词 -> wav + 转写
& $T scripts/prep_set.py --name all                                 # 建台词集
.\scripts\run_voice_data.ps1  -Voice myvoice -Src '<录音文件夹>'      # 目标说话人的素材（可分批追加）
.\scripts\run_voice_model.ps1 -Voice myvoice -Epochs 200             # 训练 RVC
.\scripts\run_voice_lines.ps1 -Voice myvoice                         # 路线一：全部台词，重配、拟声词、48k、导出
& $T scripts\compare_voices.py --voices myvoice                     # 对比试听页
```

路线二（GPT-SoVITS）：
```powershell
$env:VL_VOICE = 'myvoice'
& $T scripts/tts_testset.py                                         # 留出测试集
& "$env:VL_CACHE\envs\gptsovits\Scripts\python.exe" scripts/gsv.py prepare --name myvoice_gsv1
& "$env:VL_CACHE\envs\gptsovits\Scripts\python.exe" scripts/gsv.py train   --name myvoice_gsv1
& $T scripts/tts_eval.py <生成目录> --name <名字> --page               # 在测试集上打分
& $T scripts/tts_lines.py plan --all --work <工作目录>                 # 再 speak、pick --mode a，见 tts_lines.py 开头
```
训练用的轮数写在模型目录的 `use.json` 里，生成时一律按它取权重，避免误用过拟合的轮数。

**听审页面**在 `eval/<集>_<voice>/`，在仓库根目录运行 `python -m http.server` 后打开：
- `review.html`：逐句对比；
- `nonverbal.html`：拟声词选择页；
- `refine_<句>.html`：逐短句挑选。

## 改动日志

每次改动都留一个条目：代码、设定、成品的改动，以及最后什么都没改的实验。这样以后调优时能查到“试过什么、数字是多少”。

```powershell
& $T scripts\change_log.py start <名字> --title "…" --why "…" --files <要改的文件>   # 建条目，备份改动前的文件
& $T scripts\change_log.py diff <条目>                                             # 生成 diff.patch
& $T scripts\change_log.py attach <条目> <文件> [--to scripts]                     # 附上指标输出、一次性脚本
& $T scripts\change_log.py index                                                   # 重建 logs/changes/INDEX.md
```

条目写：起因、改动内容、运行命令、结果（改动前 → 改动后）、结论、后续可优化、回滚方法。

## 声线设定（voice.json）

| 字段 | 作用 |
|---|---|
| `label` | 页面上显示的名字 |
| `keyword` / `own_markers` | 只有目标说话人会说的口头习惯（正则），用来打锚点和弱标签 |
| `anchor_match` / `anchor_episodes` | 这些习惯只在哪些录音里专属于目标说话人 |
| `other_markers` | 其他说话人的说话习惯，作为反例 |
| `min_f0` | 音高下限，用来排除音域差别很大的说话人 |
| `nonverbal_pitch` | 拟声词转换的变调半音数 |
| `vad_min_silence_ms` / `vad_max_speech_s` | 切片粒度，切细可以把快速对话拆成单人片段 |
| `add_cuts_from` | 换了更好的同一份录音时，沿用旧录音的切点重切一遍 |
| `extra_band` | 分类器很有把握时，允许较低的声纹相似度 |
| `exclude_sources` | 跳过的文件 |
| `rvc_model` | 模型名，重新训练时换一个新名字 |

## 已知限制

- 只能处理导出过的台词；被音效完全盖住的句子，实时识别可能漏换。
- 自动的音高检查只能发现一部分跑音，最后要靠人耳判断，再用精修流程处理。
- RVC 和 GPT-SoVITS 的训练都不是逐位确定的，同样的数据训练两次，结果会略有不同。

## 第三方组件

本仓库只有调用脚本。模型和第三方代码需要按各自的许可证自行获取：
- IndexTTS-2、GPT-SoVITS（v2ProPlus）、Applio / RVC、AudioSR、Seed-VC（可选）
- audio-separator（Mel-Band RoFormer / UVR 模型）、faster-whisper（Whisper large-v3）、kotoba-whisper、Silero VAD
- 声纹：SpeechBrain ECAPA、3D-Speaker CAM++、WavLM；情绪：emotion2vec；音质：torchaudio SQUIM；语调分析：Parselmouth（Praat）；日语前端：pyopenjtalk
