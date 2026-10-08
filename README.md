# Can AI Understand Interactions Between Intimate Partners

Code for *Can AI Understand Interactions Between Intimate Partners*.

We evaluate eight open-source vision-language models on the Couple Conflict Dataset (CCD). For each 15-minute discussion video, models predict the occurrence counts of RMICS behavior codes. Per-code evaluation uses MAE, RMSE, Pearson's *r*, and Spearman's *ρ* over 304 videos per code. Pooled metrics across the 10 evaluated behavior codes are also reported.

The CCD videos are IRB-restricted and are not included in this repository. A fully synthetic sample is provided in `data/synthetic_sample/` for end-to-end pipeline testing.

## Files

| File | Purpose |
|---|---|
| `ccd_config.py` | Paths, RMICS codes, model registry, and protocol settings |
| `prompt.py` | Zero-shot, few-shot, and trace prompts |
| `few_shot_examples.json` | 14 RMICS examples used for few-shot prompting |
| `runners.py` | Model backends and output parsing |
| `run_eval.py` | Main evaluation pipeline |
| `metrics.py` | Per-code metrics, pooled metrics, and trivial baselines |

## Setup

Two environments are used because the VideoLLaMA2 family requires an older `transformers` version.

```bash
# VideoLLaMA3, LLaVA-NeXT-Video, Gemma 3n
conda create -n ccd-vlm python=3.10
conda activate ccd-vlm
pip install torch torchvision torchaudio "transformers>=4.56" accelerate pandas numpy scipy pillow ffmpeg-python imageio

# VideoLLaMA2 family
conda create -n ccd-videollama2 python=3.10
conda activate ccd-videollama2
pip install torch==2.2.0 transformers==4.42.3 accelerate pandas numpy scipy pillow
git clone -b audio_visual https://github.com/DAMO-NLP-SG/VideoLLaMA2 third_party/VideoLLaMA2
pip install -e third_party/VideoLLaMA2 --no-deps
```

## Input Format

```text
data/<name>/manifest.csv
data/<name>/windows/<CID>/video_<k>/windows.json
data/<name>/windows/<CID>/video_<k>/w01/
    frames/f01.jpg ... f16.jpg
    clip.mp4
    transcript.txt
    meta.json
```

Each 15-minute discussion is divided into 60-second windows. 16 frames are uniformly sampled from each window. Window-level predicted behavior counts are summed to obtain video-level counts.

## Run

```bash
python prompt.py

python run_eval.py --model VideoLLaMA3-7B
python run_eval.py --model VideoLLaMA3-7B --few-shot
python run_eval.py --model Gemma-3n-E4B-it --use-audio
python run_eval.py --model Gemma-3n-E4B-it --trace

python metrics.py --collect
```

Supported models:

```text
VideoLLaMA2-7B
VideoLLaMA2-7B-Base
VideoLLaMA2.1-7B-AV
VideoLLaMA2.1-7B-Base
VideoLLaMA3-2B
VideoLLaMA3-7B
LLaVA-NeXT-Video-7B
Gemma-3n-E4B-it
```

VideoLLaMA2-family models use the `ccd-videollama2` environment. The remaining models use `ccd-vlm`.

## Outputs

Results are saved under:

```text
runs/<model>__<zero|few>[__audio][__trace]/
```

| File | Content |
|---|---|
| `raw_outputs.jsonl` | Raw model outputs and parsed window-level predictions (trace mode: the parsed turns) |
| `traces.csv` | Trace mode only: one row per speaker turn with its text and the confidence for every code |
| `predictions.csv` | Video-level gold and predicted behavior counts |
| `per_code_metrics.csv` | Per-code MAE, RMSE, Pearson *r*, Spearman *ρ*, and two-sided *p*-values |
| `pooled_metrics.csv` | Pooled metrics and trivial baselines |
| `summary.json` | Run summary |