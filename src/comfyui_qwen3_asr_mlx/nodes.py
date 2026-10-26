import os
import re
import json
from pathlib import Path
from inspect import cleandoc
import soundfile as sf
from mlx_audio.stt.utils    import load_model
from mlx_audio.stt.generate import generate_transcription

def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS,mmm format for SRT"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")

def format_vtt_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format for VTT"""
    return format_timestamp(seconds).replace(",", ".")

def format_lrc_timestamp(seconds:float) -> str:
    """LRC格式：mm:ss.xx"""
    mm = int(seconds // 60)
    ss = seconds % 60
    return f"{mm:02d}:{ss:05.2f}"

def save_as_lrc(all_segments, output_path:str, title:str="", artist:str="", album:str="", lrc_by:str=""):
    """
    增强版LRC生成：
    1.支持头部元标签 ti / ar / al / by，输入为空则跳过该标签；
    2.单个segment内部多行文本，在segment[start~end]时间区间均匀分配时间戳
    """
    lrc_lines = []
    # 头部元标签，非空才写入
    if title.strip():
        lrc_lines.append(f"[ti:{title.strip()}]")
    if artist.strip():
        lrc_lines.append(f"[ar:{artist.strip()}]")
    if album.strip():
        lrc_lines.append(f"[al:{album.strip()}]")
    if lrc_by.strip():
        lrc_lines.append(f"[by:{lrc_by.strip()}]")
    if lrc_lines:
        lrc_lines.append("") # 空行分隔元标签与歌词

    print(f"\n==== lrc banner ====\n")
    for s in lrc_lines:
        print(f"{s}")
    print(f"\n==== lrc banner ====\n")
    for seg in all_segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_text = seg["text"].rstrip("%").strip()
        sub_lines = [line.strip() for line in seg_text.splitlines() if line.strip()]
        if not sub_lines:
            continue
        duration_total = seg_end - seg_start
        line_count = len(sub_lines)
        if line_count == 1:
            t_str = format_lrc_timestamp(seg_start)
            lrc_lines.append(f"[{t_str}]{sub_lines[0]}")
        else:
            step = duration_total / line_count
            for idx, sub_text in enumerate(sub_lines):
                line_time = seg_start + step * idx
                t_str = format_lrc_timestamp(line_time)
                lrc_lines.append(f"[{t_str}]{sub_text}")
    lrc_content = "\n".join(lrc_lines)
    with open(f"{output_path}.lrc","w",encoding="utf-8") as f:
        f.write(lrc_content)

def save_as_txt_merged(all_segments, output_path:str):
    full_text = "\n".join(s["text"].strip() for s in all_segments)
    with open(f"{output_path}.txt","w",encoding="utf-8") as f:
        f.write(full_text)

def save_as_srt_merged(all_segments, output_path:str):
    lines = []
    for idx,seg in enumerate(all_segments, start=1):
        s = format_timestamp(seg["start"])
        e = format_timestamp(seg["end"])
        lines.append(f"{idx}\n{s} --> {e}\n{seg['text'].strip()}\n")
    with open(f"{output_path}.srt","w",encoding="utf-8") as f:
        f.write("\n".join(lines))

def save_as_vtt_merged(all_segments, output_path:str):
    lines = ["WEBVTT",""]
    for idx,seg in enumerate(all_segments, start=1):
        s = format_vtt_timestamp(seg["start"])
        e = format_vtt_timestamp(seg["end"])
        lines.append(f"{idx}\n{s} --> {e}\n{seg['text'].strip()}\n")
    with open(f"{output_path}.vtt","w",encoding="utf-8") as f:
        f.write("\n".join(lines))

def save_as_json_merged(all_segments, output_path:str):
    payload = {
        "segments": all_segments,
        "full_text": "\n".join(s["text"].strip() for s in all_segments)
    }
    with open(f"{output_path}.json","w",encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

# =========================
# 🔹 模型加载节点（带缓存）
# =========================
class Qwen3ASRMLXLoader:
    _model_cache = {}
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_dir": ("STRING", {"default": "/path/to/your/qwen3asr-folder"}),
            }
        }
    RETURN_TYPES = ("QWEN3_ASR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model_node"
    CATEGORY = "Qwen3-ASR-MLX"
    def load_model_node(self, model_dir):
        if model_dir in self._model_cache:
            return (self._model_cache[model_dir],)
        model = load_model(model_dir)
        self._model_cache[model_dir] = model
        return (model,)

# =========================
# 🔹 ASR(单文件)生成节点（原有保留，增加language_hint）
# =========================
class Qwen3ASRMLXGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("QWEN3_ASR_MODEL",),
                "audio_path": ("STRING", {"default": ""}),
                "output_dir": ("STRING", {"default": "asr_output"}),
                "file_prefix": ("STRING", {"default": "asr"}),
                "file_format": (['txt', 'srt', 'vtt', 'json'], {"default": "txt"}),
                "verbose" : ("BOOLEAN", {"default": True}),
            },
            "optional":{
                "language_hint": (["auto","zh","en","ja","ko","ru","fr"],{"default":"auto","tooltip":"语言提示hint；auto=模型自动检测，zh=中文，en=英文，改善识别漂移"}),
            }
        }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_path", "result")
    FUNCTION = "generate_node"
    CATEGORY = "Qwen3-ASR-MLX"
    OUTPUT_NODE = True
    def generate_node(
        self,
        model,
        audio_path,
        output_dir,
        file_prefix,
        file_format,
        verbose,
        language_hint="auto",
    ):
        os.makedirs(output_dir, exist_ok=True)
        output_path=os.path.join(output_dir, f'{file_prefix}')
        result = ''
        lang_arg = None if language_hint == "auto" else language_hint
        segments = generate_transcription(
            model=model,
            audio=audio_path,
            output_path=output_path,
            format=file_format,
            verbose=verbose,
            language=lang_arg,
        )
        with open(f"{output_path}.{file_format}", "r", encoding="utf-8") as f:
            result = f.read()
        return (f'{output_path}.{file_format}',result)

# =========================
# 🔹【新增】批量文件夹分片ASR节点 Qwen3ASRMLXBatchFolder
# 匹配 {prefix}_000.wav / {prefix}_001.mp3，按序号升序识别，累加时间戳，合并输出txt/srt/vtt/json/lrc
# =========================
class Qwen3ASRMLXBatchFolder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required":{
                "model": ("QWEN3_ASR_MODEL",),
                "audio_folder": ("STRING", {"default":"","tooltip":"分片音频存放绝对目录，tts_000.wav等在此目录"}),
                "file_prefix": ("STRING", {"default":"tts","tooltip":"分片文件前缀，例如tts匹配tts_000.wav"}),
                "output_dir": ("STRING", {"default":"asr_batch_output","tooltip":"合并字幕输出目录"}),
                "out_filename": ("STRING", {"default":"merged_subtitle","tooltip":"合并输出文件名（不带后缀）"}),
                "file_format": (["txt","srt","vtt","json","lrc"],{"default":"lrc"}),
                "verbose": ("BOOLEAN",{"default":True}),
                "trigger": ("*",{"tooltip":"时序触发，等待上游音频全部写入磁盘后再执行ASR"})
            },
            "optional":{
                "language_hint": (["auto","zh","en","ja","ko","ru","fr"],{"default":"auto","tooltip":"语言提示hint；auto=模型自动检测，zh=中文，en=英文，改善识别漂移，中文TTS音频建议选zh"}),
                "title_for_lrc": ("STRING", {"default":"","tooltip":"LRC元标签ti(歌曲标题)；仅file_format=lrc生效，留空不输出标签"}),
                "artist_for_lrc": ("STRING", {"default":"","tooltip":"LRC元标签ar(艺术家)；仅file_format=lrc生效，留空不输出标签"}),
                "album_for_lrc": ("STRING", {"default":"","tooltip":"LRC元标签al(专辑)；仅file_format=lrc生效，留空不输出标签"}),
                "lrc_by": ("STRING", {"default":"Qwen3‑ASR‑MLX","tooltip":"LRC元标签by，生成工具标识；仅file_format=lrc生效"}),
                }
        }
    RETURN_TYPES = ("STRING","STRING")
    RETURN_NAMES = ("merged_file_path","merged_text")
    FUNCTION = "batch_run"
    CATEGORY = "Qwen3-ASR-MLX"
    OUTPUT_NODE = True
    def batch_run(self, model, audio_folder, file_prefix, output_dir, out_filename, file_format, verbose, trigger,
        language_hint="auto",
        title_for_lrc="", artist_for_lrc="", album_for_lrc="", lrc_by=""):
        target_dir = Path(audio_folder).absolute()
        if not target_dir.is_dir():
            raise NotADirectoryError(f"音频分片目录不存在：{target_dir}")
        os.makedirs(output_dir, exist_ok=True)
        out_base = os.path.join(output_dir, out_filename)
        # 正则匹配 prefix_数字. wav/mp3
        escaped_pre = re.escape(file_prefix)
        pat = re.compile(rf"^{escaped_pre}_(\d+)\.(wav|mp3)$", re.IGNORECASE)
        matched = []
        for fname in os.listdir(target_dir):
            m = pat.match(fname)
            if m:
                seq = int(m.group(1))
                matched.append((seq, fname))
        if len(matched) ==0:
            raise FileNotFoundError(f"目录 {target_dir} 没有找到 {file_prefix}_xxx.wav / mp3")
        matched.sort(key=lambda x:x[0])
        file_list = [str(target_dir/fn) for _,fn in matched]
        print(f"\n==== Qwen3ASRMLXBatchFolder 共找到 {len(file_list)} 个分片音频 ====")
        for f in file_list:
            print(f" > {Path(f).name}")
        total_offset = 0.0 # 时间偏移：累计前面所有音频总时长（秒）
        all_merged_segments = []
        lang_arg = None if language_hint == "auto" else language_hint
        for idx, audio_file in enumerate(file_list):
            # 获取当前分片音频实际时长
            info = sf.info(audio_file)
            dur = info.duration
            print(f"--处理分片 {Path(audio_file).name} | 分片时长={dur:.2f}s | 时间偏移offset={total_offset:.2f}s")
            # ✅ 使用output_dir下真实临时文件
            temp_base = os.path.join(output_dir, f"_temp_asr_part_{idx}")
            temp_json = f"{temp_base}.json"
            seg_obj = generate_transcription(
                model=model,
                audio=audio_file,
                output_path=temp_base,
                format="json",
                verbose=verbose,
                language=lang_arg,
            )
            # 优先取内存返回对象seg_obj；万一接口异常降级读取临时json文件
            seg_items = []
            if hasattr(seg_obj, "segments"):
                seg_items = seg_obj.segments
            elif hasattr(seg_obj, "sentences"):
                seg_items = seg_obj.sentences
            else:
                # fallback：读取临时json
                with open(temp_json,"r",encoding="utf-8") as f:
                    jdata = json.load(f)
                seg_items = jdata.get("segments", [])
            for s in seg_items:
                d = {}
                if isinstance(s,dict):
                    d["start"] = s.get("start",0.0)+total_offset
                    d["end"] = s.get("end",0.0)+total_offset
                    d["text"] = s.get("text","").strip()
                else:
                    # 对象属性（AlignedSentence）
                    d["start"] = getattr(s,"start",0.0)+total_offset
                    d["end"] = getattr(s,"end",0.0)+total_offset
                    d["text"] = getattr(s,"text","").strip()
                if d["text"]:
                    all_merged_segments.append(d)
            # ✅ 删除临时分片json，清理垃圾
            try:
                os.unlink(temp_json)
            except OSError:
                pass
            # 更新全局时间偏移
            total_offset += dur
        print(f"✅全部分片识别完成；合并总时长 {total_offset:.2f}s；有效字幕段：{len(all_merged_segments)}")
        # 根据选择格式写出合并字幕文件
        if file_format == "txt":
            save_as_txt_merged(all_merged_segments, out_base)
        elif file_format == "srt":
            save_as_srt_merged(all_merged_segments, out_base)
        elif file_format == "vtt":
            save_as_vtt_merged(all_merged_segments, out_base)
        elif file_format == "json":
            save_as_json_merged(all_merged_segments, out_base)
        elif file_format == "lrc":
            save_as_lrc(all_merged_segments, out_base,
                        title=title_for_lrc,
                        artist=artist_for_lrc,
                        album=album_for_lrc,
                        lrc_by=lrc_by)
        final_path = f"{out_base}.{file_format}"
        with open(final_path,"r",encoding="utf-8") as fp:
            merged_text = fp.read()
        print(f"✅合并字幕输出：{final_path}\n")
        return (final_path, merged_text)


NODE_CLASS_MAPPINGS = {
    "Qwen3ASRMLXLoader": Qwen3ASRMLXLoader,
    "Qwen3ASRMLXGenerate": Qwen3ASRMLXGenerate,
    "Qwen3ASRMLXBatchFolder": Qwen3ASRMLXBatchFolder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3ASRMLXLoader": "Qwen3 ASR MLX Loader",
    "Qwen3ASRMLXGenerate": "Qwen3 ASR MLX Generate(单文件)",
    "Qwen3ASRMLXBatchFolder": "Qwen3 ASR MLX Batch 文件夹分片合并ASR"
}
