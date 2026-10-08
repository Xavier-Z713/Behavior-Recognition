from pathlib import Path
import json, os

CODE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = Path(os.environ.get("CCD_DATA_DIR", CODE_DIR / "data" / "synthetic_sample"))
MANIFEST   = Path(os.environ.get("CCD_MANIFEST", DATA_DIR / "manifest.csv"))
WINDOW_DIR = Path(os.environ.get("CCD_WINDOW_DIR", DATA_DIR / "windows"))
RUNS_DIR   = Path(os.environ.get("CCD_RUNS_DIR", CODE_DIR / "runs"))

WINDOW_SEC     = int(os.environ.get("CCD_WINDOW_SEC", 60))
SNAP_TOL_SEC   = int(os.environ.get("CCD_SNAP_TOL_SEC", 10))
NUM_FRAMES     = 16
IMAGE_SIZE     = 336
MAX_NEW_TOKENS = 256
FEW_SHOT_K     = None

CODES = ["PA", "DA", "HO", "DY", "WI", "AC", "RA", "SD", "HM", "PD"]
PROMPT_CODES = CODES + ["OT"]
NEGATIVE, POSITIVE, NEUTRAL = ["PA", "DA", "HO", "DY", "WI"], ["AC", "RA", "SD", "HM"], ["PD"]

CODE_NAMES = {
    "PA": "Psychological Abuse",
    "DA": "Distress-Maintaining Attributions",
    "HO": "Hostility",
    "DY": "Dysphoric Affect",
    "WI": "Withdrawal",
    "AC": "Acceptance",
    "RA": "Relationship-Enhancing Attributions",
    "SD": "Self-Disclosure",
    "HM": "Humor",
    "PD": "Problem Description",
    "OT": "Other",
}

CODE_DESCRIPTIONS = {
    "PA": "Psychological Abuse: A communication intended to cause psychological pain to another person, or a communication perceived as having that intent.",
    "DA": "Distress-Maintaining Attributions: (a) the speaker offers an explanation for a negative event that is blameworthy or intentional on the part of the partner/self, or (b) the speaker offers an explanation for a non-negative event that the partner/self caused involuntarily or unintentionally.",
    "HO": "Hostility: All angry or irritated negative affect and statements with strong negative content, excluding behavior that is codeable as psychological abuse.",
    "DY": "Dysphoric Affect: Sad or depressed expressed emotional states.",
    "WI": "Withdrawal: Behaviors that imply pulling back from the interaction, walling off the partner, or not listening to the speaker.",

    "AC": "Acceptance: Active listening skills that help the partner feel understood and validated, including paraphrasing (restating partner's statement in one's own words), reflecting feelings (voicing what one thought the partner's underlying feelings were), giving positive feedback, and expressing caring, concern, or understanding of the partner's experience.",
    "RA": "Relationship-Enhancing Attributions: an explanation for the causes of (a) a neutral or positive event that implicates the self or partner as having acted intentionally, or (b) a negative event that exempts the partner/self from having caused it in a blameworthy manner.",
    "SD": "Self-Disclosure: Statements about the speaker's feelings, wishes or beliefs. SD can also include acceptance of responsibility not phrased as an attribution.",
    "HM": "Humor: Statements that are clearly intended to be humorous. HM also includes genuine smiling and laughing (not nervous smiling and laughing). Sarcastic humor directed at the partner is coded as hostility, not humor.",

    "PD": "Problem Description: all constructive approaches to discussing or solving problems, including elaborated disagreements.",

    "OT": "Other: discussing something other than a personal or relationship topic. OT is most often coded when the experimental situation itself is discussed. OT is coded conservatively; the statement must be clearly out of bounds. If the couple strays from the appointed topic, but is talking about anything relevant to their lives or marriage, use another code.",
}

MODEL_REGISTRY = {
    "VideoLLaMA2-7B":          dict(backend="videollama2", hf_path="DAMO-NLP-SG/VideoLLaMA2-7B"),
    "VideoLLaMA2-7B-Base":     dict(backend="videollama2", hf_path="DAMO-NLP-SG/VideoLLaMA2-7B-Base"),
    "VideoLLaMA2.1-7B-AV":     dict(backend="videollama2", hf_path="DAMO-NLP-SG/VideoLLaMA2.1-7B-AV"),
    "VideoLLaMA2.1-7B-Base":   dict(backend="videollama2", hf_path="DAMO-NLP-SG/VideoLLaMA2.1-7B-16F-Base"),
    "VideoLLaMA3-2B":          dict(backend="videollama3", hf_path="DAMO-NLP-SG/VideoLLaMA3-2B"),
    "VideoLLaMA3-7B":          dict(backend="videollama3", hf_path="DAMO-NLP-SG/VideoLLaMA3-7B"),
    "LLaVA-NeXT-Video-7B":     dict(backend="llava_next_video", hf_path="llava-hf/LLaVA-NeXT-Video-7B-hf"),
    "Gemma-3n-E4B-it":         dict(backend="gemma3n", hf_path="google/gemma-3n-E4B-it"),
}

def load_few_shot():
    with open(CODE_DIR / "few_shot_examples.json", encoding="utf-8") as fh:
        ex = json.load(fh)
    return ex if FEW_SHOT_K is None else ex[:FEW_SHOT_K]
