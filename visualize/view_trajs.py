"""
AReaL Trajectory Viewer

Gradio-based viewer for AReaL rollout trajectories.

Supports two formats:
  1. Episode JSON (one file per episode):
     {"task_id": str, "episode_idx": int, "n_steps": int, "reward": float,
      "head_version": int, "tail_version": int,
      "screenshot_paths": [...], "evaluation": [...], "task_info": {...},
      "steps": [{"step_idx": int, "prompt": str, "completion": str,
                  "observation": {...}, "response": {...}, "reward": {...}}]}

  2. Legacy JSONL (one step per line, backward compatible):
     {"task_id": int, "prompt": str, "completion": str, ...}

Usage:
  python visualize/view_trajs.py /path/to/rollout/dir
  python visualize/view_trajs.py /path/to/rollout/dir --version 0
  python visualize/view_trajs.py /path/to/rollout/dir --port 7861
  python visualize/view_trajs.py /path/to/rollout/dir --tasks-path /path/to/train.jsonl
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="View AReaL rollout trajectories")
parser.add_argument(
    "rollout_dir",
    help="Root rollout directory (contains version sub-dirs with .json/.jsonl files)",
)
parser.add_argument(
    "--version",
    type=int,
    default=None,
    help="Load only a specific version sub-directory (default: load all)",
)
parser.add_argument(
    "--port", type=int, default=7860, help="Gradio server port (default: 7860)"
)
parser.add_argument(
    "--show-prompt",
    action="store_true",
    help="Display raw prompt for each step",
)
parser.add_argument(
    "--tasks-path",
    type=str,
    default=None,
    help="Path to tasks JSONL file for metadata (difficulty, domain, etc.)",
)
parser.add_argument(
    "--critic-path",
    type=str,
    default=None,
    help="Path to critic checkpoint directory (auto-detected if not specified)",
)
parser.add_argument(
    "--no-critic",
    action="store_true",
    help="Disable critic loading even if checkpoint exists",
)
parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="Device for critic model (default: cpu, use 'cuda' for GPU)",
)
parser.add_argument(
    "--remap",
    type=str,
    default=None,
    help="Remap screenshot paths: 'old_prefix:new_prefix' (e.g. '/data/foo:/tmp/local')",
)
args = parser.parse_args()

# Path remapping for screenshots copied to a different machine
_REMAP_FROM, _REMAP_TO = None, None
if args.remap:
    _REMAP_FROM, _REMAP_TO = args.remap.split(":", 1)


def _remap_path(p: str) -> str:
    if _REMAP_FROM and p.startswith(_REMAP_FROM):
        return _REMAP_TO + p[len(_REMAP_FROM):]
    return p

# ---------------------------------------------------------------------------
# Critic model loading (auto-detect from rollout dir if not specified)
# ---------------------------------------------------------------------------

critic_model = None
critic_tokenizer = None
critic_image_processor = None


def _auto_detect_critic_path(rollout_dir: str) -> str | None:
    """Auto-detect latest critic HF checkpoint from experiment directory."""
    # rollout_dir is like .../logs/root/<exp>/<trial>/rollout
    # critic checkpoints are at .../checkpoints/root/<exp>/<trial>/critic/epoch*
    rd = Path(rollout_dir)
    # Navigate: rollout -> trial -> exp -> root -> logs -> root(parent)
    # Then: checkpoints/root/<exp>/<trial>/critic/
    trial_dir = rd.parent  # .../logs/root/<exp>/<trial>
    trial_name = trial_dir.name
    exp_name = trial_dir.parent.name
    fileroot = trial_dir.parent.parent.parent.parent  # .../experiments
    critic_root = fileroot / "checkpoints" / "root" / exp_name / trial_name / "critic"
    if not critic_root.exists():
        return None
    # Find latest epoch checkpoint (HF format, has config.json)
    candidates = sorted(
        [d for d in critic_root.iterdir() if d.is_dir() and (d / "config.json").exists()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else None


if not args.no_critic and args.critic_path is None:
    detected = _auto_detect_critic_path(args.rollout_dir)
    if detected:
        args.critic_path = detected
        print(f"Auto-detected critic checkpoint: {detected}")

if args.critic_path:
    import torch
    from safetensors.torch import load_file as load_safetensors
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    critic_path = Path(args.critic_path)
    print(f"Loading critic from: {critic_path}")

    # 1. Read checkpoint weights to detect value head architecture
    full_state_dict = {}
    for ckpt_file in critic_path.glob("*.safetensors"):
        full_state_dict.update(load_safetensors(str(ckpt_file)))
    lm_head_keys = [k for k in full_state_dict if k.startswith("lm_head")]
    print(f"  Checkpoint lm_head keys: {lm_head_keys}")

    # 2. Load tokenizer and image processor from checkpoint dir
    critic_tokenizer = AutoTokenizer.from_pretrained(
        str(critic_path), trust_remote_code=True
    )
    critic_processor = AutoProcessor.from_pretrained(
        str(critic_path), trust_remote_code=True
    )
    critic_image_processor = critic_processor.image_processor

    # 3. Find the base model name from config.json (e.g. Qwen3-VL)
    #    and load ONLY the architecture (no weights) to avoid shape mismatch
    import json as _json
    config_file = critic_path / "config.json"
    with open(config_file) as f:
        model_config = _json.load(f)
    base_model_type = model_config.get("model_type", "")
    print(f"  Model type: {base_model_type}")

    # Load architecture on CPU (buffers like rotary inv_freq need real values)
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(str(critic_path), trust_remote_code=True)
    critic_model = AutoModelForImageTextToText.from_config(
        cfg, torch_dtype=torch.bfloat16,
    )
    hidden_size = cfg.text_config.hidden_size if hasattr(cfg, "text_config") else cfg.hidden_size

    # 4. Replace lm_head to match checkpoint architecture
    if "lm_head.0.weight" in full_state_dict:
        mlp_hidden = full_state_dict["lm_head.0.weight"].shape[0]
        critic_model.lm_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, mlp_hidden, bias=True, dtype=torch.bfloat16),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden, mlp_hidden, bias=True, dtype=torch.bfloat16),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden, 1, bias=False, dtype=torch.bfloat16),
        )
        print(f"  Rebuilt MLP value head (hidden={mlp_hidden})")
    elif "lm_head.weight" in full_state_dict:
        out_features = full_state_dict["lm_head.weight"].shape[0]
        critic_model.lm_head = torch.nn.Linear(
            hidden_size, out_features, bias=False, dtype=torch.bfloat16
        )
        print(f"  Rebuilt Linear value head (out={out_features})")

    # 5. Load checkpoint weights (assign=True replaces tensors in-place)
    missing, unexpected = critic_model.load_state_dict(
        full_state_dict, strict=False, assign=True
    )
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:3]}...")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
    if not missing and not unexpected:
        print(f"  All {len(full_state_dict)} tensors loaded successfully")
    del full_state_dict  # free ~16GB CPU memory

    device = args.device
    critic_model = critic_model.to(device).eval()
    print(f"Critic model on {device}")


def compute_step_values(record: dict) -> list[float] | None:
    """Compute critic V(t) for each step in a trajectory.

    Uses each step's stored prompt (which is the full cumulative prompt
    the critic sees during training) and runs a forward pass. For VLM critics,
    loads the actual screenshots and passes pixel_values to the model.
    """
    if critic_model is None or critic_tokenizer is None:
        return None

    import torch

    # Structured steps format: each step has its own cumulative prompt
    structured_steps = record.get("steps")
    if structured_steps and isinstance(structured_steps, list):
        step_prompts = [s.get("prompt", "") for s in structured_steps]
    else:
        # Legacy format: reconstruct from top-level prompt + completion
        full_text = record.get("prompt", "") + record.get("completion", "")
        turns = parse_turns(full_text)

        step_prompts = []
        cumulative = ""
        i = 0
        while i < len(turns):
            turn = turns[i]
            if turn["role"] == "system":
                cumulative += f"{IM_START}system\n{turn['content']}{IM_END}\n"
                i += 1
            elif turn["role"] == "user":
                prompt_up_to_here = (
                    cumulative + f"{IM_START}user\n{turn['content']}{IM_END}\n"
                )
                step_prompts.append(prompt_up_to_here)
                if i + 1 < len(turns) and turns[i + 1]["role"] == "assistant":
                    cumulative = (
                        prompt_up_to_here
                        + f"{IM_START}assistant\n{turns[i + 1]['content']}{IM_END}\n"
                    )
                    i += 2
                else:
                    cumulative = prompt_up_to_here
                    i += 1
            else:
                i += 1

    if not step_prompts:
        return None

    # Load screenshot images for VLM pixel_values
    screenshot_paths = [_remap_path(p) for p in record.get("screenshot_paths", [])]
    screenshot_images = []
    for sp in screenshot_paths:
        if os.path.exists(sp):
            screenshot_images.append(Image.open(sp).convert("RGB"))
        else:
            # Try to find screenshot relative to rollout dir
            screenshot_images.append(None)

    # Compute V(t) for each step
    values = []
    device = next(critic_model.parameters()).device
    for step_idx, prompt_text in enumerate(step_prompts):
        inputs = critic_tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=16384
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Prepare pixel_values if we have screenshots and an image processor
        pixel_kwargs = {}
        if critic_image_processor is not None and screenshot_images:
            # Count vision blocks in the prompt to determine how many images to pass.
            # The prompt may use a sliding window (e.g., max 5 screenshots), so the
            # number of <|vision_start|> blocks can be fewer than step_idx + 1.
            n_vision_blocks = prompt_text.count("<|vision_start|>")
            # Take the LAST n_vision_blocks screenshots (sliding window keeps recent)
            start = max(0, step_idx + 1 - n_vision_blocks)
            step_images = [
                img
                for img in screenshot_images[start : step_idx + 1]
                if img is not None
            ]
            if step_images:
                img_out = critic_image_processor(step_images, return_tensors="pt")
                pixel_kwargs["pixel_values"] = img_out["pixel_values"].to(device)
                pixel_kwargs["image_grid_thw"] = img_out["image_grid_thw"].to(device)

        with torch.no_grad():
            outputs = critic_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **pixel_kwargs,
            )
            # Value at last token position (equivalent to prompt_pos in training)
            v = outputs.logits[0, -1, 0].item()
            values.append(v)

    return values


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
IMAGE_PAD = "<|image_pad|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_turns(text: str) -> list[dict]:
    """Parse Qwen chat-template text into a list of {role, content} turns.

    Handles the format:
      <|im_start|>role\ncontent<|im_end|>
    """
    turns = []
    parts = text.split(IM_START)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if IM_END in part:
            part = part[: part.index(IM_END)]
        lines = part.split("\n", 1)
        role = lines[0].strip()
        content = lines[1].strip() if len(lines) > 1 else ""
        turns.append({"role": role, "content": content})
    return turns


def clean_content(content: str, strip_image_pad: bool = True) -> str:
    """Clean content for display by removing image pad tokens etc."""
    if strip_image_pad:
        content = content.replace(IMAGE_PAD, "")
        content = content.replace(VISION_START, "")
        content = content.replace(VISION_END, "")
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


def extract_thinking_and_answer(content: str) -> tuple[str, str]:
    """Extract thinking and answer parts from model response.

    Returns (thinking, answer) tuple.
    """
    think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        answer = content[think_match.end() :].strip()
        return thinking, answer
    return "", content.strip()


# ---------------------------------------------------------------------------
# Step extraction
# ---------------------------------------------------------------------------


def extract_steps(record: dict) -> tuple[str, list[dict]]:
    """Parse prompt+completion into (system_msg, steps).

    Prefers structured ``"steps"`` field from new JSONL format.
    Falls back to text parsing for old records.

    Each step is {"observation": str, "action": str} where observation
    comes from a user turn and action from the following assistant turn.
    """
    # Prefer structured steps (new format with per-step data)
    structured = record.get("steps")
    if structured and isinstance(structured, list):
        steps = []
        for s in structured:
            resp = s.get("response") or {}
            obs = s.get("observation") or {}
            steps.append(
                {
                    "observation": obs.get("ac_tree", ""),
                    "action": resp.get("raw_response", ""),
                }
            )
        return "", steps

    full_text = record.get("prompt", "") + record.get("completion", "")
    turns = parse_turns(full_text)

    system_msg = ""
    steps = []

    i = 0
    while i < len(turns):
        turn = turns[i]
        if turn["role"] == "system":
            system_msg = turn["content"]
            i += 1
        elif turn["role"] == "user":
            obs = turn["content"]
            action = ""
            if i + 1 < len(turns) and turns[i + 1]["role"] == "assistant":
                action = turns[i + 1]["content"]
                i += 2
            else:
                i += 1
            steps.append({"observation": obs, "action": action})
        elif turn["role"] == "assistant":
            # Standalone assistant turn (shouldn't happen normally)
            steps.append({"observation": "", "action": turn["content"]})
            i += 1
        else:
            i += 1

    return system_msg, steps


def extract_task_description(system_msg: str) -> str:
    """Extract task description from system message."""
    if not system_msg:
        return "N/A"

    cleaned = clean_content(system_msg)

    # Common patterns in WebGym system prompts
    for pattern in [
        r"(?:Your task is|Task):\s*(.*?)(?:\n\n|\Z)",
        r"(?:you need to|please)\s+(.*?)(?:\.\s|\n\n|\Z)",
    ]:
        m = re.search(pattern, cleaned, re.IGNORECASE | re.DOTALL)
        if m:
            task_text = m.group(1).strip()
            if len(task_text) > 20:
                return task_text[:500]

    # Fall back to full system message (truncated)
    return cleaned[:500] + ("..." if len(cleaned) > 500 else "")


def extract_ac_tree(obs_content: str) -> str:
    """Extract accessibility tree from observation content."""
    if not obs_content:
        return "N/A (empty observation)"

    cleaned = clean_content(obs_content)

    # Look for accessibility tree markers
    for marker in [
        "Accessibility tree:",
        "accessibility tree:",
        "AC Tree:",
        "Accessibility Tree:",
    ]:
        idx = cleaned.find(marker)
        if idx >= 0:
            return cleaned[idx:]

    # No AC tree found — this is expected for screenshot-only prompts
    return "N/A (screenshot-only mode, no accessibility tree in prompt)"


def extract_answer_text(steps: list[dict]) -> str:
    """Extract the agent's final answer from the last step's action."""
    if not steps:
        return "No steps in trajectory"

    last_action = steps[-1].get("action", "")
    if not last_action:
        return "No action in last step"

    _, answer_part = extract_thinking_and_answer(last_action)

    # Try tool_call JSON format: {"action": "answer", "text": "..."}
    tool_call_match = re.search(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>", answer_part, re.DOTALL
    )
    if tool_call_match:
        try:
            tc = json.loads(tool_call_match.group(1))
            args = tc.get("arguments", tc)
            if args.get("action") == "answer" and "text" in args:
                return args["text"].strip()
        except (json.JSONDecodeError, AttributeError):
            pass

    # Look for answer() function call (greedy match to handle quotes in text)
    answer_match = re.search(
        r"answer\s*\(\s*[\"'](.*)[\"']\s*\)", answer_part, re.DOTALL
    )
    if answer_match:
        return answer_match.group(1).strip()

    # Look for ANSWER keyword
    if "ANSWER" in answer_part.upper():
        idx = answer_part.upper().find("ANSWER") + len("ANSWER")
        return answer_part[idx:].strip().strip("[]").strip()

    # Return the answer part (truncated)
    return answer_part[:500] if answer_part else "No answer found"


# ---------------------------------------------------------------------------
# Image compositing — loads actual screenshots saved during rollout
# Matches scaled_webgym/analysis/view_trajs.py style: screenshot on top,
# action text (thinking + answer tokens) drawn below, colored borders.
# ---------------------------------------------------------------------------

# Try to load a font (NotoSans preferred, then DejaVuSans)
_FONT_SIZE = 20
_FONT = None
_FONT_BOLD = None
for _font_path, _bold_path in [
    # NotoSans (same as scaled_webgym font_manager)
    (
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    ),
    # DejaVuSans (proportional, close to NotoSans)
    (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ),
    # DejaVuSansMono (last resort before default)
    (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    ),
]:
    try:
        _FONT = ImageFont.truetype(_font_path, _FONT_SIZE)
        try:
            _FONT_BOLD = ImageFont.truetype(_bold_path, _FONT_SIZE)
        except OSError:
            _FONT_BOLD = _FONT
        break
    except OSError:
        continue
if _FONT is None:
    _FONT = ImageFont.load_default()
    _FONT_BOLD = _FONT

BORDER = 5
LINE_SPACING = 26  # pixels between text lines
TEXT_MARGIN = 10  # left margin for text
MAX_TEXT_WIDTH = 90  # chars per line for word-wrapping (wider due to smaller font)


def _load_screenshot(path: str) -> Image.Image | None:
    """Load a screenshot from disk, returning None on failure."""
    try:
        if os.path.isfile(path):
            return Image.open(path).convert("RGB")
    except Exception:
        pass
    return None


def _wrap_text(text: str, max_width: int = MAX_TEXT_WIDTH) -> list[str]:
    """Word-wrap text preserving existing newlines."""
    result_lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            result_lines.append("")
            continue
        line = ""
        for word in paragraph.split(" "):
            if line and len(line) + 1 + len(word) > max_width:
                result_lines.append(line)
                line = word
            else:
                line = f"{line} {word}" if line else word
        if line:
            result_lines.append(line)
    return result_lines


def _draw_colored_text(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    fill: tuple = (0, 0, 0),
) -> int:
    """Draw text and return the new y position."""
    draw.text((x, y), text, fill=fill, font=_FONT)
    return y + LINE_SPACING


# Color tag definitions matching scaled_webgym
_COLOR_TAGS = [
    ("<<BLUE>>", "<</BLUE>>", (0, 0, 255)),
    ("<<GREEN>>", "<</GREEN>>", (0, 128, 0)),
    ("<<RED>>", "<</RED>>", (255, 0, 0)),
]


def _draw_line_with_color_tags(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
) -> int:
    """Draw a line of text, parsing <<BLUE>>/<</BLUE>> and <<GREEN>>/<</GREEN>> tags.

    Text outside tags is drawn in black. Returns the new y position.
    Matches scaled_webgym inline color tag rendering.
    """
    has_tags = any(start in text and end in text for start, end, _ in _COLOR_TAGS)
    if not has_tags:
        draw.text((x, y), text, fill=(0, 0, 0), font=_FONT)
        return y + LINE_SPACING

    current_x = x
    remaining = text

    while any(start in remaining and end in remaining for start, end, _ in _COLOR_TAGS):
        # Find the earliest color marker
        earliest_pos = len(remaining)
        earliest_tag = None
        for start_m, end_m, color in _COLOR_TAGS:
            if start_m in remaining and end_m in remaining:
                sp = remaining.find(start_m)
                ep = remaining.find(end_m)
                if 0 <= sp < earliest_pos and ep > sp:
                    earliest_pos = sp
                    earliest_tag = (start_m, end_m, color)

        if earliest_tag is None:
            break

        start_m, end_m, color = earliest_tag
        sp = remaining.find(start_m)
        ep = remaining.find(end_m)

        # Draw text before the tag in black
        if sp > 0:
            before = remaining[:sp]
            draw.text((current_x, y), before, fill=(0, 0, 0), font=_FONT)
            current_x += (
                _FONT.getlength(before)
                if hasattr(_FONT, "getlength")
                else len(before) * 10
            )

        # Draw tagged text in color
        colored = remaining[sp + len(start_m) : ep]
        draw.text((current_x, y), colored, fill=color, font=_FONT)
        current_x += (
            _FONT.getlength(colored)
            if hasattr(_FONT, "getlength")
            else len(colored) * 10
        )

        remaining = remaining[ep + len(end_m) :]

    # Draw any remaining text in black
    if remaining:
        draw.text((current_x, y), remaining, fill=(0, 0, 0), font=_FONT)

    return y + LINE_SPACING


def _extract_action_coordinates(answer_text: str) -> list[tuple[int, int]]:
    """Extract normalized (0-1000) click/hover/type coordinates from answer tokens.

    Supports two formats:
      1. Text-based:  Click[x, y], Type[x, y][...], Hover[x, y], Scroll[x, y][...],
                      HoverAndScroll[x, y][...]
      2. JSON tool_call: <tool_call>{"name":"computer_use","arguments":{"coordinate":[x,y],...}}</tool_call>
    """
    coords: list[tuple[int, int]] = []

    # 1. Text-based actions: Action[x, y]
    for m in re.finditer(
        r"(?:Click|Type|Hover|Scroll|HoverAndScroll)\s*\[(\d+),\s*(\d+)\]",
        answer_text,
        re.IGNORECASE,
    ):
        coords.append((int(m.group(1)), int(m.group(2))))

    # 2. JSON tool_call with "coordinate": [x, y]
    for m in re.finditer(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>", answer_text, re.DOTALL
    ):
        try:
            payload = json.loads(m.group(1))
            coord = payload.get("arguments", {}).get("coordinate")
            if isinstance(coord, list) and len(coord) >= 2:
                coords.append((int(coord[0]), int(coord[1])))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    return coords


def _draw_action_coordinates(
    img: Image.Image, coords: list[tuple[int, int]]
) -> Image.Image:
    """Draw red dots on the image at normalized (0-1000) coordinates.

    Matches scaled_webgym draw_action_coordinates style.
    """
    if not coords:
        return img
    img = img.copy()
    draw = ImageDraw.Draw(img)
    for x, y in coords:
        # Scale from 0-1000 normalized to actual pixel coordinates
        scaled_x = int(x * img.width / 1000)
        scaled_y = int(y * img.height / 1000)
        # Red circle
        radius = 8
        draw.ellipse(
            [
                scaled_x - radius,
                scaled_y - radius,
                scaled_x + radius,
                scaled_y + radius,
            ],
            fill=(255, 0, 0),
            outline=(128, 0, 0),
            width=2,
        )
        # White cross in center
        cross = 3
        draw.line(
            [scaled_x - cross, scaled_y, scaled_x + cross, scaled_y],
            fill=(255, 255, 255),
            width=2,
        )
        draw.line(
            [scaled_x, scaled_y - cross, scaled_x, scaled_y + cross],
            fill=(255, 255, 255),
            width=2,
        )
    return img


def _build_step_card(
    img: Image.Image,
    step_idx: int,
    action_raw: str,
    step_data: dict | None = None,
    state_value: float | None = None,
) -> Image.Image:
    """Build a single step card: screenshot on top, action info below, border.

    Text layout:
      Step {n}
      <<BLUE>>Submit:<</BLUE>> {value}
      <<GREEN>>Thinking Tokens:<</GREEN>>
      {thinking content}
      <<BLUE>>Answer Tokens:<</BLUE>>
      {answer content}

    Note: ``step_data["reward"]["submission_judgment"]`` is the upstream
    GPT-based screenshot filter's CoT, not the actor's reasoning, so it is
    deliberately not rendered here.
    """
    # Parse action into thinking and answer
    thinking, answer = extract_thinking_and_answer(action_raw)

    # Draw click/hover coordinates on the screenshot
    coords = _extract_action_coordinates(answer)
    if coords:
        img = _draw_action_coordinates(img, coords)

    # Get structured reward data if available
    rew = (step_data.get("reward") or {}) if step_data else {}
    submit_value = (
        rew.get("submit", False)
        if rew
        else bool(re.search(r"answer\s*\(", answer, re.IGNORECASE))
    )

    combined_text = f"Step {step_idx}"
    combined_text += f"\n<<BLUE>>Submit:<</BLUE>> {submit_value}"
    if state_value is not None:
        combined_text += f"\n<<RED>>State Value:<</RED>> {state_value:.4f}"

    # Blank line separator before action text
    combined_text += "\n"

    # Action text with thinking/answer tokens
    if thinking and thinking.strip():
        combined_text += (
            f"\n<<GREEN>>Thinking Tokens:<</GREEN>>"
            f"\n{thinking.strip()}"
            f"\n\n<<BLUE>>Answer Tokens:<</BLUE>>"
            f"\n{answer}"
        )
    elif answer:
        combined_text += f"\n<<BLUE>>Answer Tokens:<</BLUE>>\n{answer}"
    else:
        combined_text += "\nNo response available"

    # Wrap text
    lines = _wrap_text(combined_text, max_width=MAX_TEXT_WIDTH)

    # Calculate text area height (matches scaled_webgym: lines * 35 + 100)
    text_area_h = len(lines) * LINE_SPACING + 100

    # Create combined image: screenshot + text below
    combined_img = Image.new(
        "RGB", (img.width, img.height + text_area_h), (255, 255, 255)
    )
    combined_img.paste(img, (0, 0))

    # Draw text below using color tag parser
    draw = ImageDraw.Draw(combined_img)
    y = img.height + 10
    for line in lines:
        y = _draw_line_with_color_tags(draw, TEXT_MARGIN, y, line)

    # Always red border (matches scaled_webgym)
    bordered = Image.new(
        "RGB",
        (combined_img.width + 2 * BORDER, combined_img.height + 2 * BORDER),
        (255, 0, 0),
    )
    bordered.paste(combined_img, (BORDER, BORDER))
    return bordered


def build_screenshot_composite(
    screenshot_paths: list[str],
    steps: list[dict],
    structured_steps: list[dict] | None = None,
    step_values: list[float] | None = None,
) -> Image.Image | None:
    """Build a composite image from actual saved screenshots.

    Each step shows the screenshot with action info drawn below it
    (matching scaled_webgym/analysis/view_trajs.py style).
    Laid out in a 2-per-row grid.
    """
    if not screenshot_paths:
        return None

    cards: list[Image.Image] = []
    for i, spath in enumerate(screenshot_paths):
        img = _load_screenshot(spath)
        if img is None:
            img = Image.new("RGB", (800, 600), (200, 200, 200))
            draw = ImageDraw.Draw(img)
            draw.text(
                (10, 10),
                f"Could not load image:\n{spath}",
                fill=(0, 0, 0),
                font=_FONT,
            )
        action_raw = steps[i].get("action", "") if i < len(steps) else ""
        step_data = (
            structured_steps[i]
            if structured_steps and i < len(structured_steps)
            else None
        )
        sv = step_values[i] if step_values and i < len(step_values) else None
        cards.append(
            _build_step_card(img, i, action_raw, step_data=step_data, state_value=sv)
        )

    if not cards:
        return None

    # Lay out 2 per row
    per_row = 2
    rows: list[Image.Image] = []
    max_w = 0
    for r in range(0, len(cards), per_row):
        row_cards = cards[r : r + per_row]
        w = sum(c.width for c in row_cards)
        h = max(c.height for c in row_cards)
        row_img = Image.new("RGB", (w, h), (255, 255, 255))
        x = 0
        for c in row_cards:
            row_img.paste(c, (x, 0))
            x += c.width
        rows.append(row_img)
        max_w = max(max_w, w)

    total_h = sum(r.height for r in rows)
    composite = Image.new("RGB", (max_w, total_h), (255, 255, 255))
    y = 0
    for row_img in rows:
        composite.paste(row_img, (0, y))
        y += row_img.height
    return composite


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_trajectories(rollout_dir: str, version: int | None = None) -> list[dict]:
    """Load trajectory records from the rollout directory.

    Supports two formats:
      - Legacy JSONL: rollout_dir/{version}/*.jsonl (one step per line)
      - Episode JSON: rollout_dir/{version}/*.json (one episode per file)
    """
    rollout_path = Path(rollout_dir)
    if not rollout_path.exists():
        print(f"Error: rollout directory not found: {rollout_dir}")
        sys.exit(1)

    records: list[dict] = []

    if version is not None:
        version_dirs = [rollout_path / str(version)]
        if not version_dirs[0].exists():
            print(f"Error: version directory not found: {version_dirs[0]}")
            sys.exit(1)
    else:
        version_dirs = sorted(
            [d for d in rollout_path.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda d: int(d.name),
        )
        if not version_dirs:
            if list(rollout_path.glob("*.jsonl")) or list(rollout_path.glob("*.json")):
                version_dirs = [rollout_path]
            else:
                print(
                    f"Error: no version directories or trajectory files in {rollout_dir}"
                )
                sys.exit(1)

    for vdir in version_dirs:
        version_num = int(vdir.name) if vdir.name.isdigit() else -1

        # Load legacy JSONL files (one step per line)
        for jsonl_file in sorted(vdir.glob("*.jsonl")):
            with open(jsonl_file) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        record["_source_file"] = str(jsonl_file)
                        record["_version_dir"] = version_num
                        records.append(record)
                    except json.JSONDecodeError as e:
                        print(f"Warning: bad JSON in {jsonl_file}:{line_num}: {e}")

        # Load episode JSON files (one episode per file)
        for json_file in sorted(vdir.glob("*.json")):
            try:
                with open(json_file) as f:
                    record = json.load(f)
                record["_source_file"] = str(json_file)
                record["_version_dir"] = version_num
                records.append(record)
            except (json.JSONDecodeError, OSError) as e:
                print(f"Warning: bad JSON in {json_file}: {e}")

    return records


# Load data
print(f"Loading trajectories from: {args.rollout_dir}")
data = load_trajectories(args.rollout_dir, args.version)

if not data:
    print("No trajectories found. Check the rollout directory path.")
    sys.exit(1)

# Sort by (version, task_id, episode_idx)
data.sort(
    key=lambda r: (
        r.get("tail_version", 0),
        str(r.get("task_id", "")),
        r.get("episode_idx", r.get("sample_idx", 0)),
    )
)

# Compute summary stats
total = len(data)
successful = sum(1 for r in data if r.get("reward", 0) > 0)
successful_traj_ids = [i for i, r in enumerate(data) if r.get("reward", 0) > 0]
versions = sorted(set(r.get("tail_version", -1) for r in data))
task_ids = sorted(set(r.get("task_id", -1) for r in data))
avg_reward = sum(r.get("reward", 0) for r in data) / total if total else 0

print(
    f"Loaded {total} trajectories ({successful} successful, avg reward {avg_reward:.4f})"
)
print(f"Versions: {versions}")
print(f"Unique task IDs: {len(task_ids)}")
# ---------------------------------------------------------------------------
# Task metadata loading
# ---------------------------------------------------------------------------

task_metadata: dict[int, dict] = {}

if args.tasks_path:
    tasks_path = Path(args.tasks_path)
    if tasks_path.exists():
        print(f"Loading task metadata from: {args.tasks_path}")
        with open(tasks_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    task = json.loads(line)
                    tid = task.get("task_id")
                    if tid is not None:
                        try:
                            task_metadata[int(tid)] = task
                        except (ValueError, TypeError):
                            # Non-integer task_id (e.g. '185832_d1'), store as string
                            task_metadata[str(tid)] = task
                except json.JSONDecodeError:
                    pass
        print(f"Loaded metadata for {len(task_metadata)} tasks")
    else:
        print(f"Warning: tasks file not found: {args.tasks_path}")


def _resolve_task(task_id: int | str) -> dict | None:
    """Look up task metadata, trying int and str keys."""
    t = task_metadata.get(task_id)
    if t:
        return t
    t = task_metadata.get(str(task_id))
    if t:
        return t
    try:
        t = task_metadata.get(int(task_id))
        if t:
            return t
    except (ValueError, TypeError):
        pass
    return None


def get_task_metadata(task_id: int | str) -> tuple[str, str, str, str, str]:
    """Get (difficulty, domain, subdomain, website, evaluator_reference) for a task."""
    task = _resolve_task(task_id)
    if not task:
        return "N/A", "N/A", "N/A", "N/A", "N/A"
    difficulty = str(task.get("difficulty", "N/A"))
    domain = str(task.get("domain", "N/A"))
    subdomain = str(task.get("subdomain", "N/A"))
    website = str(task.get("website", "N/A"))

    # Format evaluator_reference nicely
    eval_ref = task.get("evaluator_reference", "N/A")
    if isinstance(eval_ref, list) and eval_ref:
        formatted_items = []
        for item in eval_ref:
            if isinstance(item, dict):
                item_id = item.get("id", "?")
                desc = item.get("description", "N/A")
                facts = item.get("facts", [])
                item_str = f"[{item_id}] {desc}"
                if facts and isinstance(facts, list):
                    for fact in facts:
                        item_str += f"\n    - {fact}"
                formatted_items.append(item_str)
            else:
                formatted_items.append(str(item))
        evaluator_reference = "\n\n".join(formatted_items)
    else:
        evaluator_reference = str(eval_ref)

    return difficulty, domain, subdomain, website, evaluator_reference


# ---------------------------------------------------------------------------
# Evaluation formatting (matching scaled_webgym/analysis/view_trajs.py)
# ---------------------------------------------------------------------------


def format_evaluation_info(eval_info: list, evaluator_ref: list | None = None) -> str:
    """Format evaluation info based on evaluator_reference structure.

    Handles both legacy format and new format with separate Criterion A/B:
    - [Criterion B - Anti-Hallucination]: Task-level response verification
    - [Criterion A - Fact N]: Per-fact verification (checked for each rubric)
    - [Reference Answer Evaluation]: Reference answer comparison
    """
    if not eval_info or not isinstance(eval_info, list):
        return ""

    formatted_parts = []
    criterion_a_evals = []
    criterion_b_eval = None
    reference_answer_eval = None
    legacy_evals = []

    # Separate evaluations by type
    for evaluation in eval_info:
        eval_str = str(evaluation)
        if eval_str.startswith("[Criterion B - Anti-Hallucination]"):
            criterion_b_eval = eval_str
        elif eval_str.startswith("[Criterion A - Fact"):
            criterion_a_evals.append(eval_str)
        elif eval_str.startswith("[Reference Answer Evaluation]"):
            reference_answer_eval = eval_str
        else:
            legacy_evals.append(eval_str)

    # New format with Criterion A/B
    if criterion_b_eval or criterion_a_evals:
        if criterion_b_eval:
            formatted_parts.append("=" * 60)
            formatted_parts.append(
                "TASK-LEVEL EVALUATION (Criterion B - Anti-Hallucination)"
            )
            formatted_parts.append(
                "Checks if the agent's response is supported by screenshots"
            )
            formatted_parts.append("=" * 60)
            criterion_b_content = criterion_b_eval.replace(
                "[Criterion B - Anti-Hallucination] ", ""
            )
            formatted_parts.append(criterion_b_content)
            formatted_parts.append("")

        if criterion_a_evals:
            formatted_parts.append("=" * 60)
            formatted_parts.append(
                "PER-FACT EVALUATIONS (Criterion A - Fact Verification)"
            )
            formatted_parts.append(
                "Checks if screenshots contain evidence for each fact"
            )
            formatted_parts.append("=" * 60)

            for eval_str in criterion_a_evals:
                match = re.match(
                    r"\[Criterion A - Fact (\d+)\] (.*)", eval_str, re.DOTALL
                )
                if match:
                    fact_num = match.group(1)
                    content = match.group(2)

                    rubric_desc = ""
                    if evaluator_ref and isinstance(evaluator_ref, list):
                        idx = int(fact_num) - 1
                        if 0 <= idx < len(evaluator_ref):
                            rubric = evaluator_ref[idx]
                            if isinstance(rubric, dict):
                                rubric_desc = rubric.get("description", "")
                            else:
                                rubric_desc = str(rubric)

                    formatted_parts.append(f"\n--- Fact {fact_num} ---")
                    if rubric_desc:
                        truncated = (
                            f"{rubric_desc[:100]}..."
                            if len(rubric_desc) > 100
                            else rubric_desc
                        )
                        formatted_parts.append(f"Rubric: {truncated}")
                    formatted_parts.append(content)
                else:
                    formatted_parts.append(eval_str)
            formatted_parts.append("")

        if reference_answer_eval:
            formatted_parts.append("=" * 60)
            formatted_parts.append("REFERENCE ANSWER EVALUATION")
            formatted_parts.append("=" * 60)
            ref_content = reference_answer_eval.replace(
                "[Reference Answer Evaluation] ", ""
            )
            formatted_parts.append(ref_content)

        return "\n".join(formatted_parts).strip()

    # Legacy format: evaluations without Criterion A/B prefixes
    if legacy_evals:
        if not evaluator_ref or not isinstance(evaluator_ref, list):
            formatted = [
                f"Evaluation {i}:\n{s}\n" for i, s in enumerate(legacy_evals, 1)
            ]
            return "\n".join(formatted).strip()

        # New format with 'facts' key in evaluator_ref
        first_item = evaluator_ref[0] if evaluator_ref else None
        has_facts = (
            isinstance(first_item, dict) and "facts" in first_item
            if first_item
            else False
        )

        if has_facts:
            formatted_eval = []
            eval_idx = 0
            for group_idx, group in enumerate(evaluator_ref, 1):
                group_desc = (
                    group.get("description", "N/A")
                    if isinstance(group, dict)
                    else str(group)
                )
                formatted_eval.append(f"Group {group_idx}: {group_desc}")
                facts = group.get("facts", []) if isinstance(group, dict) else []
                for fact_num in range(1, len(facts) + 1):
                    if eval_idx < len(legacy_evals):
                        fact_desc = (
                            facts[fact_num - 1] if fact_num <= len(facts) else "N/A"
                        )
                        formatted_eval.append(f"Fact {fact_num}: {fact_desc}")
                        formatted_eval.append(str(legacy_evals[eval_idx]))
                        formatted_eval.append("")
                        eval_idx += 1
                formatted_eval.append("")
            return "\n".join(formatted_eval).strip()

        # Old format: group embedded in description
        group_to_facts: dict[int, list] = {}
        group_descriptions: dict[int, str] = {}
        for idx, item in enumerate(evaluator_ref):
            desc = item.get("description", "") if isinstance(item, dict) else str(item)
            match = re.match(r"\[Group (\d+): ([^\]]+)\] (.+)", desc)
            if match:
                gnum = int(match.group(1))
                gdesc = match.group(2).strip()
                fdesc = match.group(3).strip()
                if gnum not in group_to_facts:
                    group_to_facts[gnum] = []
                    group_descriptions[gnum] = gdesc
                group_to_facts[gnum].append((idx, fdesc))

        if group_to_facts:
            formatted_eval = []
            for gnum in sorted(group_to_facts.keys()):
                gdesc = group_descriptions.get(gnum, "N/A")
                formatted_eval.append(f"Group {gnum}: {gdesc}")
                for fnum, (fidx, fdesc) in enumerate(group_to_facts[gnum], 1):
                    if fidx < len(legacy_evals):
                        formatted_eval.append(f"Fact {fnum}: {fdesc}")
                        formatted_eval.append(str(legacy_evals[fidx]))
                        formatted_eval.append("")
                formatted_eval.append("")
            return "\n".join(formatted_eval).strip()

        # Fallback: just number them
        formatted = [f"Evaluation {i}:\n{s}\n" for i, s in enumerate(legacy_evals, 1)]
        return "\n".join(formatted).strip()

    return ""


# ---------------------------------------------------------------------------
# Display functions
# ---------------------------------------------------------------------------


def display_trajectory(traj_id, step_id=0):
    """Display trajectory overview.

    Returns 12 values matching the old scaled_webgym layout:
      [task, answer, reward, blocked, eval_info, image,
       ac_tree, difficulty, domain, subdomain, website, evaluator_ref]
    """
    traj_id = int(traj_id)
    step_id = int(step_id)

    if traj_id < 0 or traj_id >= len(data):
        return [
            "Invalid trajectory ID",
            "",
            "",
            "",
            "",
            "",
            None,
            "Invalid trajectory ID",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
        ]

    record = data[traj_id]
    system_msg, steps = extract_steps(record)
    structured_steps = record.get("steps")

    # Task — prefer embedded task_info, then external metadata
    task_id_val = record.get("task_id", "?")
    task_info_embedded = record.get("task_info")
    if task_info_embedded and isinstance(task_info_embedded, dict):
        task_name = task_info_embedded.get("task_name", "")
        task = (
            f"[Task ID: {task_id_val}] {task_name}"
            if task_name
            else f"[Task ID: {task_id_val}]"
        )
    else:
        meta = _resolve_task(task_id_val)
        if meta:
            task_name = meta.get("task_name", "")
            task = (
                f"[Task ID: {task_id_val}] {task_name}"
                if task_name
                else f"[Task ID: {task_id_val}]"
            )
        else:
            task_desc = extract_task_description(system_msg)
            task = f"[Task ID: {task_id_val}] {task_desc}"

    # Answer
    answer = extract_answer_text(steps)

    # Reward
    reward = f"Traj reward: {record.get('reward', 0)}"

    # Blocked status
    is_blocked = record.get("is_blocked")
    if is_blocked is True:
        blocked = "YES - Website blocked the agent"
    elif is_blocked is False:
        blocked = "NO - Website accessible"
    else:
        blocked = "N/A (no evaluator data)"

    # Eval info: show evaluator responses matching scaled_webgym layout
    evaluation = record.get("evaluation")
    # Get raw evaluator_reference list for grouping context
    eval_ref_raw = None
    meta_for_eval = _resolve_task(task_id_val)
    if meta_for_eval:
        eval_ref_raw = meta_for_eval.get("evaluator_reference")

    if evaluation and isinstance(evaluation, list):
        eval_info = format_evaluation_info(evaluation, eval_ref_raw)
    else:
        # No evaluation data — show trajectory metadata
        eval_info = (
            f"Reward: {record.get('reward', 0)}\n"
            f"Head Version: {record.get('head_version', '?')}\n"
            f"Tail Version: {record.get('tail_version', '?')}\n"
            f"Episode Index: {record.get('episode_idx', record.get('sample_idx', '?'))}\n"
            f"Num Steps: {len(steps)}\n"
            f"Source: {os.path.basename(record.get('_source_file', '?'))}\n"
        )
        if evaluation is None:
            eval_info += (
                "\n(No evaluator data — evaluator may have failed or was not used)"
            )

    # Build composite from actual screenshots saved during rollout
    screenshot_paths = [_remap_path(p) for p in record.get("screenshot_paths", [])]
    # Fallback: extract image paths from structured steps
    if not screenshot_paths and structured_steps and isinstance(structured_steps, list):
        screenshot_paths = [
            _remap_path((s.get("observation") or {}).get("image_path", ""))
            for s in structured_steps
            if (s.get("observation") or {}).get("image_path")
        ]
    # Critic step values (if critic model loaded)
    step_values = compute_step_values(record)
    if step_values is not None:
        critic_values_str = ", ".join(f"{v:.4f}" for v in step_values)
        critic_values_str = f"[{critic_values_str}]"
    else:
        critic_values_str = "N/A (no --critic-path)"

    images = build_screenshot_composite(
        screenshot_paths,
        steps,
        structured_steps=structured_steps,
        step_values=step_values,
    )

    # Add screenshot availability info
    if not screenshot_paths:
        screenshot_info = "No screenshots (pre-screenshot record)"
    else:
        existing = sum(1 for p in screenshot_paths if os.path.isfile(p))
        screenshot_info = f"Screenshots: {existing}/{len(screenshot_paths)} files found"
        if existing < len(screenshot_paths):
            missing = [p for p in screenshot_paths if not os.path.isfile(p)]
            screenshot_info += f"\nMissing: {missing[:3]}"
    eval_info += "\n" + screenshot_info

    # AC tree for selected step — prefer structured data
    if structured_steps and isinstance(structured_steps, list):
        if 0 <= step_id < len(structured_steps):
            ac_tree = (structured_steps[step_id].get("observation") or {}).get(
                "ac_tree", ""
            )
        elif structured_steps:
            ac_tree = (structured_steps[0].get("observation") or {}).get("ac_tree", "")
        else:
            ac_tree = "No steps in trajectory"
        if not ac_tree:
            ac_tree = "N/A (screenshot-only mode, no accessibility tree in prompt)"
    elif 0 <= step_id < len(steps):
        ac_tree = extract_ac_tree(steps[step_id]["observation"])
    elif steps:
        ac_tree = extract_ac_tree(steps[0]["observation"])
    else:
        ac_tree = "No steps in trajectory"

    # Task metadata: prefer embedded task_info, fallback to external file
    if task_info_embedded and isinstance(task_info_embedded, dict):
        difficulty = str(task_info_embedded.get("difficulty", "N/A"))
        domain = str(task_info_embedded.get("domain", "N/A"))
        subdomain = str(task_info_embedded.get("subdomain", "N/A"))
        website = str(task_info_embedded.get("website", "N/A"))

        eval_ref_raw_val = task_info_embedded.get("evaluator_reference", "N/A")
        # evaluator_reference may be a JSON string from the dataset loader
        if isinstance(eval_ref_raw_val, str):
            try:
                eval_ref_parsed = json.loads(eval_ref_raw_val)
                if isinstance(eval_ref_parsed, list):
                    eval_ref_raw_val = eval_ref_parsed
            except (json.JSONDecodeError, TypeError):
                pass

        if isinstance(eval_ref_raw_val, list) and eval_ref_raw_val:
            formatted_items = []
            for item in eval_ref_raw_val:
                if isinstance(item, dict):
                    item_id = item.get("id", "?")
                    desc = item.get("description", "N/A")
                    facts = item.get("facts", [])
                    item_str = f"[{item_id}] {desc}"
                    if facts and isinstance(facts, list):
                        for fact in facts:
                            item_str += f"\n    - {fact}"
                    formatted_items.append(item_str)
                else:
                    formatted_items.append(str(item))
            evaluator_ref = "\n\n".join(formatted_items)
        else:
            evaluator_ref = str(eval_ref_raw_val)
    else:
        difficulty, domain, subdomain, website, evaluator_ref = get_task_metadata(
            task_id_val
        )

    return [
        task,
        answer,
        reward,
        blocked,
        critic_values_str,
        eval_info,
        images,
        ac_tree,
        difficulty,
        domain,
        subdomain,
        website,
        evaluator_ref,
    ]


def get_step_info(traj_id, step_id):
    """Get step-specific info: (ac_tree, raw_prompt, raw_response).

    Prefers structured ``"steps"`` data when available for richer display
    (JSON raw_prompt, submission judgment, per-step metadata).
    """
    traj_id = int(traj_id)
    step_id = int(step_id)

    if traj_id < 0 or traj_id >= len(data):
        return "Invalid trajectory ID", "Invalid", "Invalid"

    record = data[traj_id]

    # Check for structured steps (new format)
    structured_steps = record.get("steps")
    if structured_steps and isinstance(structured_steps, list):
        if step_id < 0 or step_id >= len(structured_steps):
            return (
                f"Invalid step ID. Trajectory has {len(structured_steps)} steps "
                f"(0-{len(structured_steps) - 1})",
                "",
                "",
            )

        step_data = structured_steps[step_id]
        obs = step_data.get("observation") or {}
        resp = step_data.get("response") or {}
        rew = step_data.get("reward") or {}

        # AC tree
        ac_tree = obs.get("ac_tree", "")
        if not ac_tree:
            ac_tree = "N/A (screenshot-only mode, no accessibility tree in prompt)"

        # Raw prompt: JSON messages array from the actual API call
        raw_prompt_json = resp.get("raw_prompt", "")
        if raw_prompt_json:
            try:
                prompt_data = json.loads(raw_prompt_json)
                raw_prompt = json.dumps(prompt_data, indent=2, ensure_ascii=False)
                # Truncate base64 data URLs for readability
                raw_prompt = re.sub(
                    r"(data:image/[^;]+;base64,)[A-Za-z0-9+/=]{100,}",
                    r"\1<base64_truncated>",
                    raw_prompt,
                )
                # Truncate file:// image paths content display
                raw_prompt = re.sub(
                    r'(file://[^\s"]+)',
                    lambda m: (
                        m.group(1)[:120] + "..."
                        if len(m.group(1)) > 120
                        else m.group(1)
                    ),
                    raw_prompt,
                )
            except (json.JSONDecodeError, TypeError):
                raw_prompt = raw_prompt_json
        else:
            raw_prompt = "(no raw_prompt stored for this step)"

        # Raw response: actor's full output (thinking + answer when present)
        raw_resp_text = resp.get("raw_response", "")
        if raw_resp_text:
            thinking, answer_text = extract_thinking_and_answer(raw_resp_text)
            raw_response = "=== RAW MODEL RESPONSE ===\n"
            if thinking:
                raw_response += f"[Thinking]\n{thinking}\n\n[Answer]\n{answer_text}"
            else:
                raw_response += raw_resp_text.strip()
        else:
            raw_response = "No response for this step"

        # Append reward info
        step_reward = rew.get("reward", 0)
        is_blocked = rew.get("is_blocked", False)
        submit = rew.get("submit", False)
        raw_response += (
            f"\n\n=== STEP REWARD INFO ===\n"
            f"Reward: {step_reward}, Submit: {submit}, Blocked: {is_blocked}"
        )

        return ac_tree, raw_prompt, raw_response

    # Fallback: text-parsed steps (old format)
    _, steps = extract_steps(record)

    if step_id < 0 or step_id >= len(steps):
        return (
            f"Invalid step ID. Trajectory has {len(steps)} steps (0-{len(steps) - 1})",
            "",
            "",
        )

    step = steps[step_id]

    # AC tree
    ac_tree = extract_ac_tree(step["observation"])

    # Raw prompt: full conversation up to and including this step's observation
    system_msg_full, _ = extract_steps(record)
    prompt_parts = []
    if system_msg_full:
        prompt_parts.append(f"=== SYSTEM ===\n{clean_content(system_msg_full)}")
    for s in range(step_id + 1):
        prev = steps[s]
        prompt_parts.append(
            f"=== USER (Step {s}) ===\n{clean_content(prev['observation'])}"
        )
        if s < step_id and prev["action"]:
            prompt_parts.append(
                f"=== ASSISTANT (Step {s}) ===\n{clean_content(prev['action'])}"
            )
    raw_prompt = "\n\n".join(prompt_parts) if prompt_parts else "(empty prompt)"

    # Raw response
    action = step["action"]
    if action:
        thinking, answer_text = extract_thinking_and_answer(action)

        raw_response = "=== RAW MODEL RESPONSE ===\n"
        if thinking:
            raw_response += f"[Thinking]\n{thinking}\n\n[Answer]\n{answer_text}"
        else:
            raw_response += clean_content(action)
    else:
        raw_response = "No response for this step"

    return ac_tree, raw_prompt, raw_response


# ---------------------------------------------------------------------------
# Gradio UI (matching scaled_webgym/analysis/view_trajs.py layout)
# ---------------------------------------------------------------------------

# Truncate successful_traj_ids for display
display_ids = successful_traj_ids[:50]
display_ids_text = str(display_ids)
if len(successful_traj_ids) > 50:
    display_ids_text += f" ... ({len(successful_traj_ids)} total)"

# Default starting trajectory: first successful one, else 0
default_traj_id = successful_traj_ids[0] if successful_traj_ids else 0

with gr.Blocks() as interface:
    gr.Markdown("# WebGym Analysis Dashboard - AReaL Rollouts")
    gr.Markdown(
        f"Total trajectories: {total}, "
        f"Successful: {successful} ({100 * successful / total:.1f}%)"
    )

    gr.Markdown("## Trajectory Viewer")
    gr.Markdown(f"Some successful trajectory IDs: {display_ids_text}")
    with gr.Row():
        traj_id_input = gr.Number(label="Trajectory ID", value=default_traj_id)
        step_id_input = gr.Number(label="Step ID", value=0)
        reward_output = gr.Textbox(label="Reward")
        blocked_output = gr.Textbox(label="Website Blocked?")
        critic_values_output = gr.Textbox(label="Critic V(t)")

    with gr.Row():
        task_output = gr.Textbox(label="Task")

    # Task metadata row
    with gr.Row():
        difficulty_output = gr.Textbox(label="Difficulty")
        benchmark_name_output = gr.Textbox(label="Domain")
        subdomain_output = gr.Textbox(label="Subdomain")

    with gr.Row():
        website_output = gr.Textbox(label="Website")
        evaluator_reference_output = gr.Textbox(label="Evaluator Reference")

    with gr.Row():
        ac_tree_output = gr.Textbox(label="Accessibility Tree", lines=10, max_lines=20)

    with gr.Row():
        images_output = gr.Image(label="Steps Overview")

    with gr.Row():
        answer_output = gr.Textbox(label="Answer")
        eval_info_output = gr.Textbox(
            label="Evaluation Details", lines=15, max_lines=40
        )

    gr.Markdown("## Step-Specific Details")
    gr.Markdown(
        "Shows the raw API prompt (JSON messages), model response, "
        "and submission judgment for the selected step."
    )

    with gr.Row():
        raw_prompt_output = gr.Textbox(label="Raw Model Prompt", lines=15, max_lines=30)

    with gr.Row():
        raw_response_output = gr.Textbox(
            label="Raw Model Response",
            lines=15,
            max_lines=30,
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def update_all(traj_id, step_id):
        """Update trajectory display + step info when traj_id changes."""
        traj_results = display_trajectory(traj_id, step_id)
        step_results = get_step_info(traj_id, step_id)
        return (*traj_results, *step_results)

    # Update when trajectory ID is submitted (Enter key)
    traj_id_input.submit(
        fn=update_all,
        inputs=[traj_id_input, step_id_input],
        outputs=[
            task_output,
            answer_output,
            reward_output,
            blocked_output,
            critic_values_output,
            eval_info_output,
            images_output,
            ac_tree_output,  # from display_trajectory
            difficulty_output,
            benchmark_name_output,
            subdomain_output,
            website_output,
            evaluator_reference_output,
            # Step-specific outputs (ac_tree overwrites the one above)
            ac_tree_output,
            raw_prompt_output,
            raw_response_output,
        ],
    )

    # Update when step ID is submitted (only step-specific info)
    step_id_input.submit(
        fn=get_step_info,
        inputs=[traj_id_input, step_id_input],
        outputs=[
            ac_tree_output,
            raw_prompt_output,
            raw_response_output,
        ],
    )

    all_outputs = [
        task_output,
        answer_output,
        reward_output,
        blocked_output,
        critic_values_output,
        eval_info_output,
        images_output,
        ac_tree_output,
        difficulty_output,
        benchmark_name_output,
        subdomain_output,
        website_output,
        evaluator_reference_output,
        ac_tree_output,
        raw_prompt_output,
        raw_response_output,
    ]

    # Auto-load the default trajectory on page load
    interface.load(
        fn=update_all,
        inputs=[traj_id_input, step_id_input],
        outputs=all_outputs,
    )

    # Also update on change (not just submit/Enter)
    traj_id_input.change(
        fn=update_all,
        inputs=[traj_id_input, step_id_input],
        outputs=all_outputs,
    )

    step_id_input.change(
        fn=get_step_info,
        inputs=[traj_id_input, step_id_input],
        outputs=[
            ac_tree_output,
            raw_prompt_output,
            raw_response_output,
        ],
    )

print(f"\nStarting Gradio server on port {args.port}...")
# share defaults off: the public gradio tunnel (frpc) cannot reach its server
# from locked-down cluster nodes and only adds a startup timeout + warning.
# Access via VS Code port-forwarding instead. Set GRADIO_SHARE=1 to force it on.
_share = os.environ.get("GRADIO_SHARE", "0") == "1"
interface.launch(share=_share, server_name="0.0.0.0", server_port=args.port)
