import hashlib
import json

from ccd_config import CODES, PROMPT_CODES, CODE_DESCRIPTIONS, NEGATIVE, POSITIVE, NEUTRAL, load_few_shot

_SCHEMA_JSON = "{" + ", ".join(f'"{c}": <count>' for c in PROMPT_CODES) + "}"


_TRACE_SCHEMA = '{"turn": 1, "text": "...", ' + ", ".join(f'"{c}": <confidence>' for c in PROMPT_CODES) + "}"
_TRACE_OUTPUT = """Output
List every speaker turn you identified in this segment, in order, one turn per line. Each line must be one JSON object with these fields and nothing else: "turn" (the turn number), "text" (the words of that turn exactly as they appear in the transcript), and one confidence between 0.00 and 1.00 for each code, giving how likely it is that this turn receives that code, in this order:
""" + _TRACE_SCHEMA + """
Output only these lines. Do not add a summary, placeholder turns, or turns that do not occur in the transcript."""


def build_prompt(transcript: str, window_index: int, n_windows: int, start_s: float, end_s: float,
                 few_shot: bool = False, inputs_note: str = None, trace: bool = False) -> str:
    inputs_note = inputs_note or "16 frames sampled uniformly across the segment, and the speech transcript of the segment"
    codes_block = "\n".join(f"- {c}: {CODE_DESCRIPTIONS[c]}" for c in PROMPT_CODES)
    mmss = lambda s: f"{int(s // 60):02d}:{int(s % 60):02d}"

    few_shot_block = ""
    if few_shot:
        parts = []
        for i, ex in enumerate(load_few_shot(), 1):
            parts.append(f"Example {i}\nTranscript: {ex['transcript']}\nCounts: {json.dumps(ex['target'])}")
        few_shot_block = ("\nExamples\n"
                          "The following examples are coded utterances from the RMICS manual. Each example is one speaker turn with its "
                          "reference counts. Text in square brackets describes nonverbal behavior of the speaker or the listener, "
                          "which is not written in the segment. "
                          "You must infer it from the frames. When the listener shows codable behavior, the listener receives a code "
                          "in addition to the speaker's code, and both are included in the counts. The segment you must code is different from these examples.\n\n"
                          + "\n\n".join(parts) + "\n")

    transcript_block = transcript.strip() if transcript and transcript.strip() else "[No speech was transcribed in this segment.]"

    output_block = _TRACE_OUTPUT if trace else f"""Output
Return only a JSON object with exactly these eleven integer fields, in this order. Replace each <count> with the total number of times that code was assigned in this segment, counting speaker and listener assignments together (0 if the code was never assigned):
{_SCHEMA_JSON}"""

    return f"""You are an expert observational coder trained in the Rapid Marital Interaction Coding System (RMICS).

You are shown one segment of a discussion between two partners. You receive {inputs_note}. Speaker changes are not marked in the transcript, so you should infer them from the content and the frames.

Codes
{codes_block}

RMICS coding rules
- The coding unit is the speaker turn. A turn lasts until the other partner takes the floor. For each turn, you should code both speaker and listener.
- Assign the single code that best characterizes the turn. If more than one code applies, use this precedence: negative codes take precedence over positive codes, which take precedence over the neutral code, and the neutral code over OT.
- Use the transcript for what is said and the frames for facial expression, gaze, gesture, and posture. Nonverbal behavior can change the code of a turn.

Task
Identify each speaker turn, then assign one RMICS code to the speaker and, when the listener shows codable behavior, one code to the listener in each turn. A listener who emits no codable behavior receives no code.
{few_shot_block}
Transcript of this segment
{transcript_block}

{output_block}"""


def prompt_hash(few_shot: bool, trace: bool = False, inputs_note: str = None) -> str:
    text = build_prompt("<TRANSCRIPT>", 1, 15, 0, 60, few_shot=few_shot, inputs_note=inputs_note, trace=trace)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


if __name__ == "__main__":
    demo = "<transcript of the 60-second segment goes here>"
    print("=" * 100 + "\nZERO-SHOT PROMPT   (hash " + prompt_hash(False) + ")\n" + "=" * 100)
    print(build_prompt(demo, 7, 15, 360, 420, few_shot=False))
    print("\n" + "=" * 100 + "\nFEW-SHOT PROMPT   (hash " + prompt_hash(True) + ")\n" + "=" * 100)
    print(build_prompt(demo, 7, 15, 360, 420, few_shot=True))
