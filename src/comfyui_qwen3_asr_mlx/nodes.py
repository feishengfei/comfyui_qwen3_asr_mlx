import os
import re
import json
from pathlib import Path
from inspect import cleandoc
import soundfile as sf
from mlx_audio.stt.utils    import load_model
from mlx_audio.stt.generate import generate_transcription

# ====================== 字幕后处理切分配置（大写常量） ==================
SUBTITLE_MAX_CHARS = 35        # 单条字幕最大字符数(中英文均按字符计数)
SUBTITLE_MAX_DURATION_S = 6.0  # 单条字幕最大时长(秒)
SUBTITLE_SPLIT_PUNCT = "，。！？；"       # 中文优先切分标点
SUBTITLE_SPLIT_PUNCT_EN = ",.!?;"        # 英文切分标点
TRIM_PUNCT_EXCEPT = "！？!?" # trim ending except zh/en exclamation & question mark
# =============================================================================

def split_by_text_ratio_interpolate(seg_items, max_chars: int, max_dur: float, puncts: str):
    """
    无align、无word时间戳：按标点/最大字符切分，按【字符数量占比】分配片段时长
    字符计数使用len()，天然兼容中文、英文混合文本
    :param seg_items: ASR输出segment列表，仅包含 start,end,text
    :param max_chars: 单片段最大字符数
    :param max_dur: 单片段最大时长(本函数仅作为配置保留，切分逻辑以标点+字符数为主)
    :param puncts: 切分标点集合字符串
    :return: list[dict], 局部时间(未叠加分片offset) [{"start":float,"end":float,"text":str}]
    """
    out = []
    punct_set = set(puncts)
    for seg in seg_items:
        if isinstance(seg, dict):
            seg_start = seg.get("start", 0.0)
            seg_end = seg.get("end", 0.0)
            full_text = seg.get("text", "").strip()
        else:
            seg_start = getattr(seg, "start", 0.0)
            seg_end = getattr(seg, "end", 0.0)
            full_text = getattr(seg, "text", "").strip()
        seg_total_dur = seg_end - seg_start
        # 边界保护：空文本或时长接近0，直接原样返回
        if not full_text or seg_total_dur <= 1e-6:
            out.append({"start": seg_start, "end": seg_end, "text": full_text})
            continue
        pieces = []
        buf = []
        for ch in full_text:
            buf.append(ch)
            buf_str = "".join(buf)
            # 遇到切分标点 或 字符数达到上限，执行断开
            if ch in punct_set or len(buf_str) >= max_chars:
                piece = buf_str.strip()
                # 去除片段末尾的切分标点
                while len(piece) > 0 and piece[-1] in punct_set and piece[-1] not in TRIM_PUNCT_EXCEPT:
                    piece = piece[:-1].strip()
                if piece:
                    pieces.append(piece)
                buf.clear()
        # 缓冲区剩余内容
        if buf:
            piece = "".join(buf).strip()
            if piece:
                pieces.append(piece)
        if len(pieces) == 0:
            out.append({"start": seg_start, "end": seg_end, "text": full_text})
            continue
        # 全部子片段总字符数，用于计算占比
        total_char = sum(len(p) for p in pieces)
        cur_t = seg_start
        for piece in pieces:
            char_cnt = len(piece)
            ratio = char_cnt / total_char
            piece_dur = seg_total_dur * ratio
            p_end = cur_t + piece_dur
            out.append({
                "start": cur_t,
                "end": p_end,
                "text": piece
            })
            cur_t = p_end
    return out


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
# 新增开关 auto_split_subtitle：默认True，启用按字符占比插值自动切分字幕；关闭输出原始ASR segment
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
                "auto_split_subtitle": ("BOOLEAN",{"default":True,"tooltip":"开启：按标点+字符占比插值切分长字幕（无word‑level对齐）；关闭：输出ASR原始大segment，适合调试原始时间戳"}),
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
        auto_split_subtitle=True,
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

            # ========== 新增：字幕切分逻辑，auto_split_subtitle开关控制 ==========
            if auto_split_subtitle:
                all_punct = SUBTITLE_SPLIT_PUNCT + SUBTITLE_SPLIT_PUNCT_EN
                local_segs = split_by_text_ratio_interpolate(
                    seg_items,
                    max_chars=SUBTITLE_MAX_CHARS,
                    max_dur=SUBTITLE_MAX_DURATION_S,
                    puncts=all_punct
                )
                print(f"ℹ️ auto_split_subtitle开启，原始{len(seg_items)} → 切分后{len(local_segs)}条子目")
            else:
                # 不切分，原样转为dict
                local_segs = []
                for s in seg_items:
                    d = {}
                    if isinstance(s,dict):
                        d["start"] = s.get("start",0.0)
                        d["end"] = s.get("end",0.0)
                        d["text"] = s.get("text","").strip()
                    else:
                        d["start"] = getattr(s,"start",0.0)
                        d["end"] = getattr(s,"end",0.0)
                        d["text"] = getattr(s,"text","").strip()
                    if d["text"]:
                        local_segs.append(d)

            # 叠加分片全局时间偏移
            for d in local_segs:
                d["start"] = d["start"] + total_offset
                d["end"] = d["end"] + total_offset
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
