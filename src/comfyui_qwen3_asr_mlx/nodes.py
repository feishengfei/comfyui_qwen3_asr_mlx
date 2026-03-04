import os
from inspect import cleandoc

from mlx_audio.stt.utils    import load_model
from mlx_audio.stt.generate import generate_transcription

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
# 🔹 ASR(Automatic Speech Recognition) 生成节点
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
    ):
        os.makedirs(output_dir, exist_ok=True)

        output_path=os.path.join(output_dir, f'{file_prefix}')
        result = ''

        # 调用生成
        generate_transcription(
            model=model,
            audio=audio_path,
            output_path=output_path,
            format=file_format,
            verbose=verbose,
        )

        with open(f"{output_path}.{file_format}", "r", encoding="utf-8") as f:
            result = f.read()

        return (f'{output_path}.{file_format}',result)

# =========================
# 🔹 注册
# =========================

NODE_CLASS_MAPPINGS = {
    "Qwen3ASRMLXLoader": Qwen3ASRMLXLoader,
    "Qwen3ASRMLXGenerate": Qwen3ASRMLXGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3ASRMLXLoader": "Qwen3 ASR MLX Loader",
    "Qwen3ASRMLXGenerate": "Qwen3 ASR MLX Generate",
}
