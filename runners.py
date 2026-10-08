import json, re
from pathlib import Path
from typing import List

from ccd_config import CODES, PROMPT_CODES, NUM_FRAMES, IMAGE_SIZE, MAX_NEW_TOKENS


class RunnerBase:
    max_new_tokens = MAX_NEW_TOKENS
    name = "base"
    supports_audio = False

    def __init__(self, model_path: str, device: str = "cuda:0", dtype: str = "bfloat16"):
        self.model_path, self.device, self.dtype_name = model_path, device, dtype
        self.model = self.processor = None

    def load(self): ...

    def generate(self, prompt: str, frame_paths: List[Path], clip_path: Path, use_audio: bool = False) -> str:
        raise NotImplementedError

    def _torch(self):
        import torch
        return torch, {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[self.dtype_name]

    def _device_map(self):
        return "auto" if self.device == "auto" else {"": self.device}

    def _target_device(self):
        return self.model.device if self.device == "auto" else self.device

    @staticmethod
    def _load_frames(frame_paths, as_numpy=False):
        from PIL import Image
        import numpy as np
        imgs = []
        for p in frame_paths:
            im = Image.open(p).convert("RGB")
            w, h = im.size; s = IMAGE_SIZE / min(w, h)
            im = im.resize((round(w * s), round(h * s)), Image.BICUBIC)
            imgs.append(np.asarray(im) if as_numpy else im)
        return np.stack(imgs) if as_numpy else imgs


class LlavaNextVideoRunner(RunnerBase):
    name = "llava_next_video"

    def load(self):
        torch, dtype = self._torch()
        from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
        self.processor = LlavaNextVideoProcessor.from_pretrained(self.model_path)
        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype=dtype, low_cpu_mem_usage=True, device_map=self._device_map()).eval()

    def generate(self, prompt, frame_paths, clip_path, use_audio=False):
        torch, _ = self._torch()
        video = self._load_frames(frame_paths, as_numpy=True)                 # (16, H, W, 3)
        conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "video"}]}]
        chat = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(text=chat, videos=video, return_tensors="pt").to(self._target_device())
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        out = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(out, skip_special_tokens=True)[0]


class Gemma3nRunner(RunnerBase):
    name = "gemma3n"
    supports_audio = True

    def load(self):
        torch, dtype = self._torch()
        from transformers import AutoProcessor, Gemma3nForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = Gemma3nForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype=dtype, device_map=self._device_map()).eval()

    def generate(self, prompt, frame_paths, clip_path, use_audio=False):
        torch, _ = self._torch()
        content = [{"type": "image", "image": im} for im in self._load_frames(frame_paths)]
        if use_audio:
            for wav in self._wav_chunks(clip_path):
                content.append({"type": "audio", "audio": str(wav)})
        content.append({"type": "text", "text": prompt})
        inputs = self.processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                                    tokenize=True, return_dict=True, return_tensors="pt").to(self._target_device())
        n = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.processor.decode(out[0][n:], skip_special_tokens=True)

    @staticmethod
    def _wav_of(clip_path):
        import subprocess
        wav = Path(clip_path).with_suffix(".wav")
        if not wav.exists():
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(clip_path), "-vn", "-ac", "1", "-ar", "16000", str(wav)], check=True)
        return wav

    AUDIO_CHUNK_SEC = 30

    @classmethod
    def _wav_chunks(cls, clip_path):
        import json, math, subprocess
        wav = cls._wav_of(clip_path)
        meta = json.loads((Path(clip_path).parent / "meta.json").read_text())
        dur = max(0.1, meta["end"] - meta["start"])
        n = max(1, math.ceil(dur / (cls.AUDIO_CHUNK_SEC + 1)))
        if n == 1:
            return [wav]
        part_len = dur / n
        chunks = []
        for i in range(n):
            part = wav.with_name(f"{wav.stem}_a{i}.wav")
            if not part.exists():
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{i * part_len:.3f}", "-i", str(wav),
                                "-t", f"{part_len:.3f}", "-ac", "1", "-ar", "16000", str(part)], check=True)
            chunks.append(part)
        return chunks


class VideoLLaMA2Runner(RunnerBase):
    name = "videollama2"

    def load(self):
        from videollama2 import model_init
        from videollama2.utils import disable_torch_init
        disable_torch_init()
        self.model, self.processor, self.tokenizer = model_init(self.model_path)
        self.supports_audio = "AV" in self.model_path

    def generate(self, prompt, frame_paths, clip_path, use_audio=False):
        from videollama2 import mm_infer
        video_tensor = self.processor["video"]([str(p) for p in frame_paths], num_frames=NUM_FRAMES)
        if use_audio and self.supports_audio:
            audio = self._full_fbank(clip_path)
            return mm_infer({"video": video_tensor, "audio": audio}, prompt, model=self.model, tokenizer=self.tokenizer,
                            modal="video", do_sample=False, max_new_tokens=self.max_new_tokens)
        return mm_infer(video_tensor, prompt, model=self.model, tokenizer=self.tokenizer,
                        modal="video", do_sample=False, max_new_tokens=self.max_new_tokens)

    @staticmethod
    def _full_fbank(clip_path):
        import torch, torchaudio.compliance.kaldi as ta_kaldi
        from videollama2.mm_utils import load_audio_from_video
        wav, sr = load_audio_from_video(str(clip_path))
        w = torch.from_numpy(wav).unsqueeze(0) * 2 ** 15
        fb = ta_kaldi.fbank(w, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10).to(torch.bfloat16)
        return fb.unsqueeze(0)


class VideoLLaMA3Runner(RunnerBase):
    name = "videollama3"

    def load(self):
        torch, dtype = self._torch()
        from transformers import AutoModelForCausalLM, AutoProcessor
        import transformers.image_utils as _iu
        if not hasattr(_iu, "VideoInput"):
            from transformers.video_utils import VideoInput as _VI
            _iu.VideoInput = _VI
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True, torch_dtype=dtype,
                                                          device_map=self._device_map(), attn_implementation="sdpa").eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)

    def generate(self, prompt, frame_paths, clip_path, use_audio=False):
        torch, dtype = self._torch()
        import numpy as np
        meta = json.loads((Path(clip_path).parent / "meta.json").read_text())
        dur = max(1.0, meta["end"] - meta["start"])
        frames = [np.asarray(im) for im in self._load_frames(frame_paths)]
        n = len(frames)
        timestamps = [round(i * dur / n, 1) for i in range(n)]
        conversation = [{"role": "user", "content": [
            {"type": "video", "video": frames, "num_frames": n, "timestamps": timestamps},
            {"type": "text", "text": prompt}]}]
        inputs = self.processor(conversation=conversation, return_tensors="pt")
        dev = self._target_device()
        inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()


BACKENDS = {r.name: r for r in (LlavaNextVideoRunner, Gemma3nRunner, VideoLLaMA2Runner, VideoLLaMA3Runner)}


def get_runner(backend: str, model_path: str, device: str, dtype: str) -> RunnerBase:
    r = BACKENDS[backend](model_path, device, dtype)
    r.load()
    return r


_INT = r"(-?\d+(?:\.\d+)?)"

def parse_counts(text: str):
    if not text:
        return None
    s = re.sub(r"```(?:json)?", "", text).strip()
    for cand in reversed(re.findall(r"\{[^{}]*\}", s, flags=re.S)):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and sum(c in obj for c in CODES) >= 5:
            return {c: max(0, int(round(float(obj.get(c, 0) or 0)))) for c in PROMPT_CODES}
    found = {}
    for c in PROMPT_CODES:
        m = re.search(rf'\b{c}\b["\']?\s*[:=]\s*{_INT}', s)
        if m: found[c] = max(0, int(round(float(m.group(1)))))
    if len(found) >= 5:
        return {c: found.get(c, 0) for c in PROMPT_CODES}
    return None


def parse_trace(text: str):
    rows, bad = [], 0
    if not text:
        return rows, bad
    s = re.sub(r"```(?:json)?", "", text)
    for cand in re.findall(r"\{[^{}]*\}", s, flags=re.S):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            bad += 1
            continue
        if not isinstance(obj, dict) or sum(c in obj for c in PROMPT_CODES) < 6:
            bad += 1
            continue
        conf = {}
        for c in PROMPT_CODES:
            try:
                conf[c] = min(1.0, max(0.0, float(obj.get(c, 0) or 0)))
            except (TypeError, ValueError):
                conf[c] = 0.0
        try:
            turn = int(obj.get("turn", len(rows) + 1))
        except (TypeError, ValueError):
            turn = len(rows) + 1
        rows.append(dict(turn=turn, text=str(obj.get("text", "")), conf=conf))
    return rows, bad
