import gradio as gr
from PIL import Image, ImageFont, ImageDraw
import torch
import re
import os
import pandas as pd
import unicodedata
import argparse

# Import the new font manager
import sys
import os
sys.path.append(os.path.dirname(__file__))
from font_manager import FontManager, setup_fonts_with_auto_download

# Parse command line arguments
parser = argparse.ArgumentParser(description='View WebGym trajectories')
parser.add_argument('split', choices=['train', 'test'],
                    help='Which split to view (train or test)')
parser.add_argument('--data-path', type=str, required=True,
                    help='Data directory path containing task files (e.g., /data/shared).')
parser.add_argument('--log-path', type=str, required=True,
                    help='Log directory path containing trajectories (e.g., /data/exp1).')
parser.add_argument('--show-prompt', action='store_true',
                    help='Display both prompt and response for each step')
parser.add_argument('--position', choices=['first', 'last'], default=None,
                    help='Load first or last N iterations (default: load all)')
parser.add_argument('--num-iterations', type=int, default=None,
                    help='Number of iterations to load (used with --position)')
args = parser.parse_args()

# Validate data path
if not os.path.isdir(args.data_path):
    print(f"Error: Data directory does not exist: {args.data_path}")
    exit(1)

# Validate log path
if not os.path.isdir(args.log_path):
    print(f"Error: Log directory does not exist: {args.log_path}")
    exit(1)

args.data_dir = args.data_path
args.logs_dir = args.log_path
print(f"📂 Using data directory: {args.data_dir}")
print(f"📂 Using log directory: {args.logs_dir}")

# Set up font directory - you can customize this path
FONT_DIR = os.path.join(args.logs_dir, "fonts")

if args.split == 'train':
    task_path = os.path.join(args.data_dir, 'train.jsonl')
else:
    task_path = os.path.join(args.data_dir, 'test.jsonl')

print(f"Loading {args.split} trajectories from: {args.logs_dir}/{args.split}_trajectories/")
print(f"Loading task metadata from: {task_path}")

# Import and use incremental loading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from webgym.utils import load_all_trajectories
import torch

# Determine which iteration files to load
if args.position and args.num_iterations:
    traj_dir = os.path.join(args.logs_dir, f'{args.split}_trajectories')

    # Find all iteration files
    pattern = re.compile(rf'{args.split}_trajectories\.pt\.iteration(\d+)$')
    iteration_files = []

    for filename in os.listdir(traj_dir):
        match = pattern.match(filename)
        if match:
            iteration_num = int(match.group(1))
            filepath = os.path.join(traj_dir, filename)
            iteration_files.append((iteration_num, filepath))

    # Sort by iteration number
    iteration_files.sort(key=lambda x: x[0])
    total_iterations = len(iteration_files)

    # Select which files to load
    if args.position == 'first':
        selected_files = iteration_files[:args.num_iterations]
        print(f"Loading first {len(selected_files)} of {total_iterations} iteration files...")
    else:  # 'last'
        selected_files = iteration_files[-args.num_iterations:]
        print(f"Loading last {len(selected_files)} of {total_iterations} iteration files...")

    # Load only selected iteration files
    data = []
    for iteration_num, filepath in selected_files:
        loaded_data = torch.load(filepath, weights_only=False)
        # Extract trajectories from dict format
        if isinstance(loaded_data, dict) and 'trajectories' in loaded_data:
            trajectories = loaded_data['trajectories']
        else:
            raise ValueError(f"Invalid trajectory file format in {filepath}. Expected dict with 'trajectories' key.")
        data.extend(trajectories)
        print(f"  ✓ Loaded iteration {iteration_num}: {len(trajectories)} trajectories")

    print(f"✓ Total trajectories loaded: {len(data)} from {len(selected_files)} files")

elif args.position or args.num_iterations:
    print("Warning: Both --position and --num-iterations must be specified together. Loading all trajectories.")
    data = load_all_trajectories(base_dir=args.logs_dir, split=args.split)
else:
    # Load all trajectories
    data = load_all_trajectories(base_dir=args.logs_dir, split=args.split)

if not data:
    print(f"Error: No {args.split} trajectories found in {args.logs_dir}/{args.split}_trajectories/")
    sys.exit(1)


# Initialize the new font manager and get fonts
FALLBACK_FONTS = setup_fonts_with_auto_download(
    font_directory=FONT_DIR,
    font_size=28
)

# If no fonts were found, fall back to system default
if not FALLBACK_FONTS:
    print("Warning: No Unicode fonts found. Falling back to system default font.")
    try:
        default_font = ImageFont.load_default()
        FALLBACK_FONTS = [default_font]
    except Exception as e:
        print(f"Error loading default font: {e}")
        FALLBACK_FONTS = []

print(f"Loaded {len(FALLBACK_FONTS)} fonts for fallback system")
for i, font in enumerate(FALLBACK_FONTS):
    print(f"  Font {i}: {getattr(font, 'path', 'Unknown path')}")

# Create a mapping from font categories to loaded fonts for easier access
FONT_CATEGORY_MAP = {}
for font in FALLBACK_FONTS:
    font_path = getattr(font, 'path', '')
    font_name = os.path.basename(font_path).lower()
    
    if 'notosans-regular.ttf' in font_name:
        FONT_CATEGORY_MAP['base'] = font
    elif 'notosanscjksc' in font_name:
        FONT_CATEGORY_MAP['cjk_sc'] = font
    elif 'notosanscjkjp' in font_name:
        FONT_CATEGORY_MAP['cjk_jp'] = font
    elif 'notosanscjkkr' in font_name:
        FONT_CATEGORY_MAP['cjk_kr'] = font
    elif 'notosansarabic' in font_name:
        FONT_CATEGORY_MAP['arabic'] = font
    elif 'notosansdevanagari' in font_name:
        FONT_CATEGORY_MAP['devanagari'] = font
    elif 'notosanssymbols' in font_name:
        FONT_CATEGORY_MAP['symbols2'] = font
    elif 'notocoloremoji' in font_name:
        FONT_CATEGORY_MAP['emoji'] = font

# Very simple "pick a font by Unicode range" (Pillow lacks true font fallback)
def pick_font_for_char(ch):
    if not FALLBACK_FONTS:
        return None
    
    o = ord(ch)
    # Emoji (very rough)
    if 0x1F300 <= o <= 0x1FAFF:
        return FONT_CATEGORY_MAP.get('emoji', FALLBACK_FONTS[-1])
    # CJK Unified Ideographs
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0x20000 <= o <= 0x2A6DF:
        # prefer SC JP KR in that order
        for key in ("cjk_sc", "cjk_jp", "cjk_kr"):
            if key in FONT_CATEGORY_MAP:
                return FONT_CATEGORY_MAP[key]
        return FALLBACK_FONTS[-1]
    # Arabic
    if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0x08A0 <= o <= 0x08FF:
        return FONT_CATEGORY_MAP.get('arabic', FALLBACK_FONTS[-1])
    # Devanagari
    if 0x0900 <= o <= 0x097F:
        return FONT_CATEGORY_MAP.get('devanagari', FALLBACK_FONTS[-1])
    # Cyrillic
    if 0x0400 <= o <= 0x04FF:
        return FONT_CATEGORY_MAP.get('base', FALLBACK_FONTS[-1])
    # default to base, then last-resort
    return FONT_CATEGORY_MAP.get('base', FALLBACK_FONTS[-1])

def draw_text_with_fallback(draw, xy, text, fill=(0,0,0), line_spacing=35):
    if not FALLBACK_FONTS:
        # Emergency fallback to system default
        try:
            default_font = ImageFont.load_default()
            draw.text(xy, text, font=default_font, fill=fill)
            return
        except Exception:
            # Last resort: just draw the text without font
            draw.text(xy, text, fill=fill)
            return
    
    x, y = xy
    line = ""
    for ch in text:
        if ch == "\n":
            # flush the buffered line in one go (use base font for measurement)
            font = pick_font_for_char("A")
            if font:
                draw.text((x, y), line, font=font, fill=fill)
            y += line_spacing
            line = ""
            continue
        font = pick_font_for_char(ch)
        if font:
            draw.text((x, y), ch, font=font, fill=fill)
            try:
                adv = draw.textlength(ch, font=font)
            except Exception:
                adv = font.getbbox(ch)[2]
            x += adv
        else:
            # If no font found, just advance by character width
            x += 10  # Approximate character width
    if line:
        font = pick_font_for_char("A")
        if font:
            draw.text((x, y), line, font=font, fill=fill)

# Load the JSONL file with task metadata
try:
    task_metadata_df = pd.read_json(task_path, lines=True)
    print(f"Loaded task metadata with {len(task_metadata_df)} tasks")
    print("Available columns:", task_metadata_df.columns.tolist())

    # Create composite task_id column for matching (format: subdomain_website_difficulty_taskname)
    if 'subdomain' in task_metadata_df.columns and 'website' in task_metadata_df.columns:
        task_metadata_df['composite_task_id'] = (
            task_metadata_df['subdomain'].astype(str) + '_' +
            task_metadata_df['website'].astype(str) + '_' +
            task_metadata_df['difficulty'].astype(str) + '_' +
            task_metadata_df['task_name'].astype(str)
        )
        print(f"Created composite_task_id column. Sample: {task_metadata_df['composite_task_id'].iloc[0][:100] if len(task_metadata_df) > 0 else 'N/A'}...")
    else:
        print("Warning: Required columns for composite task_id not found!")
except Exception as e:
    print(f"Error loading task metadata: {e}")
    task_metadata_df = pd.DataFrame()

# Find successful trajectories (reward > 0)
successful_traj_ids = []
for i, traj in enumerate(data):
    if traj:
        if len(traj) > 0:
            last_step = traj[-1]
            if hasattr(last_step.get('reward', None), 'reward'):
                if last_step['reward'].reward > 0:
                    successful_traj_ids.append(i)

print("successful trajectories:", successful_traj_ids)
print("total successful trajectories:", len(successful_traj_ids))
print("total trajectories:", len(data))

import re

def wrap_text_preserving_whitespace(text, max_width=70):
    # Split text into tokens while preserving all whitespace.
    # The pattern '(\s+)' splits on any whitespace but also keeps it.
    tokens = re.split(r'(\s+)', text)
    lines = []
    current_line = ""

    for token in tokens:
        # If the token contains a newline, we break it up further.
        if "\n" in token:
            # This splits on newline but leaves any text before/after it.
            parts = token.split("\n")
            for i, part in enumerate(parts):
                # Try to add the current part to the current line.
                if len(current_line) + len(part) <= max_width:
                    current_line += part
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = part

                # For all but the last sub-token, a newline was encountered.
                # So, we finish the line.
                if i < len(parts) - 1:
                    lines.append(current_line)
                    current_line = ""
            continue

        # For tokens with only spaces or tabs (or any whitespace without newlines),
        # check if appending would exceed the max_width.
        if len(current_line) + len(token) <= max_width:
            current_line += token
        else:
            # If the token itself is longer than max_width, or current line is non-empty,
            # finish the current line first.
            if current_line:
                lines.append(current_line)
            current_line = token

    if current_line:
        lines.append(current_line)
    
    return lines

def safe_get_attr(obj, attr_name, default=None):
    """Generic safe attribute getter that works with objects, dicts, or the pattern: getattr() or obj.get()"""
    try:
        if obj is None:
            return default
        # Try attribute access first
        if hasattr(obj, attr_name):
            return getattr(obj, attr_name, default)
        # Try dict-like access
        elif hasattr(obj, 'get'):
            return obj.get(attr_name, default)
        elif hasattr(obj, '__getitem__'):
            return obj[attr_name]
        return default
    except (KeyError, AttributeError, TypeError, NotImplementedError):
        return default

def safe_get_nested(obj, *attr_path, default=None):
    """Safely get nested attributes. Example: safe_get_nested(step, 'observation', 'task', 'task_name')"""
    result = obj
    for attr in attr_path:
        result = safe_get_attr(result, attr, None)
        if result is None:
            return default
    return result if result is not None else default

# Kept for backwards compatibility, now just wraps safe_get_attr
def safe_get_observation_attribute(observation, attr_name, default=""):
    """Safely get an attribute from an observation object."""
    return safe_get_attr(observation, attr_name, default)

def safe_get_task_info(observation, default=""):
    """Safely get composite task ID from observation."""
    task = safe_get_attr(observation, 'task')
    if not task:
        return default

    # Try to construct composite task_id: subdomain_website_difficulty_taskname
    subdomain = safe_get_attr(task, 'subdomain')
    website = safe_get_attr(task, 'website')
    difficulty = safe_get_attr(task, 'difficulty')
    task_name = safe_get_attr(task, 'task_name')

    if subdomain and website and difficulty is not None and task_name:
        return f"{subdomain}_{website}_{difficulty}_{task_name}"

    # Fallback: try to get task_id directly
    return safe_get_attr(task, 'task_id') or str(task) if task else default

def get_task_metadata(task_id):
    """Get additional task metadata from the JSONL file."""
    if task_metadata_df.empty or task_id is None:
        return "N/A", "N/A", "N/A", "N/A", "N/A"

    try:
        # Find matching task by composite_task_id
        matching_tasks = task_metadata_df[task_metadata_df['composite_task_id'] == task_id]

        if matching_tasks.empty:
            print(f"No matching task found for task_id: {task_id[:200] if isinstance(task_id, str) else task_id}")
            return "Not found", "Not found", "Not found", "Not found", "Not found"

        # Get the first matching task
        task_row = matching_tasks.iloc[0]

        # Extract the requested fields
        difficulty = str(task_row.get('difficulty', 'N/A'))
        domain = str(task_row.get('domain', 'N/A'))
        subdomain = str(task_row.get('subdomain', 'N/A'))
        website = str(task_row.get('website', 'N/A'))

        # Format evaluator_reference nicely for display
        # Expected format: list of dicts with 'id', 'description', and 'facts'
        eval_ref = task_row.get('evaluator_reference', 'N/A')

        if isinstance(eval_ref, list) and eval_ref:
            formatted_items = []
            for item in eval_ref:
                item_id = item.get('id', '?')
                desc = item.get('description', 'N/A')
                facts = item.get('facts', [])

                # Format: "[ID] description\n    • fact1\n    • fact2"
                item_str = f"[{item_id}] {desc}"
                if facts and isinstance(facts, list):
                    for fact in facts:
                        item_str += f"\n    • {fact}"
                formatted_items.append(item_str)
            # Add blank line between items for better readability
            evaluator_reference = "\n\n".join(formatted_items)
        else:
            evaluator_reference = str(eval_ref)

        return difficulty, domain, subdomain, website, evaluator_reference

    except Exception as e:
        print(f"Error getting task metadata for {task_id}: {e}")
        return "Error", "Error", "Error", "Error", "Error"

def safe_get_image_path(observation, default_path=""):
    """Safely get image path from observation."""
    return safe_get_attr(observation, 'image_path') or safe_get_attr(observation, 'image') or default_path

def safe_get_response_object(step):
    """Safely get Response object from step."""
    return safe_get_attr(step, 'response')

def safe_get_ac_tree(observation, default=""):
    """Safely get accessibility tree from observation."""
    return safe_get_attr(observation, 'ac_tree') or safe_get_attr(observation, 'accessibility_tree') or default

def safe_get_submit_field(step, default=False):
    """Safely get submit field from step's reward (new location) or observation (legacy)."""
    # Try reward first (new location)
    submit_val = safe_get_nested(step, 'reward', 'submit')
    if submit_val is None:
        # Fallback to observation (legacy)
        submit_val = safe_get_nested(step, 'observation', 'submit')

    if submit_val is None:
        return default
    # Convert to boolean
    if isinstance(submit_val, str):
        return submit_val.lower() in ('true', '1', 'yes')
    return bool(submit_val)

def safe_get_submission_judgment(step, default=None):
    """Safely get submission_judgment field from step's reward."""
    return safe_get_nested(step, 'reward', 'submission_judgment', default=default)

def extract_action_coordinates(action):
    """Extract coordinates from an action object. Returns list of (x, y) tuples."""
    try:
        if action is None:
            return []
        
        coordinates = []
        
        # Check if it's an Action object with action attribute
        if hasattr(action, 'action'):
            action_info = getattr(action, 'action', {})
            
            if isinstance(action_info, dict):
                # Handle different coordinate formats
                action_key = action_info.get('key', '')
                action_args = action_info.get('arguments', {})
                
                if isinstance(action_args, dict):
                    # Coordinates in arguments
                    if 'coordinates' in action_args:
                        coords = action_args['coordinates']
                        if isinstance(coords, list) and len(coords) >= 2:
                            if len(coords) == 4:  # [xmin, ymin, xmax, ymax]
                                x = (coords[0] + coords[2]) // 2
                                y = (coords[1] + coords[3]) // 2
                                coordinates.append((x, y))
                            elif len(coords) == 2:  # [x, y]
                                coordinates.append((coords[0], coords[1]))

                    # Alternative: box_2d format
                    elif 'box_2d' in action_args:
                        box_2d = action_args['box_2d']
                        if isinstance(box_2d, list) and len(box_2d) > 0:
                            if isinstance(box_2d[0], list) and len(box_2d[0]) >= 4:
                                coords = box_2d[0]  # First bounding box
                                x = (coords[0] + coords[2]) // 2
                                y = (coords[1] + coords[3]) // 2
                                coordinates.append((x, y))
        
        return coordinates
    except Exception as e:
        print(f"Error extracting coordinates: {e}")
        return []

def draw_action_coordinates(image, coordinates):
    """Draw red dots on the image at the specified coordinates."""
    if not coordinates:
        return image
    
    # Create a copy to avoid modifying original
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    for x, y in coordinates:
        # Scale coordinates if they're in 0-999 range to image dimensions
        if x <= 999 and y <= 999:
            scaled_x = int(x * img_copy.width / 1000)
            scaled_y = int(y * img_copy.height / 1000)
        else:
            scaled_x, scaled_y = int(x), int(y)
        
        # Draw red circle
        radius = 8
        draw.ellipse([
            scaled_x - radius, scaled_y - radius,
            scaled_x + radius, scaled_y + radius
        ], fill=(255, 0, 0), outline=(128, 0, 0), width=2)
        
        # Draw small cross in center for precision
        cross_size = 3
        draw.line([
            scaled_x - cross_size, scaled_y,
            scaled_x + cross_size, scaled_y
        ], fill=(255, 255, 255), width=2)
        draw.line([
            scaled_x, scaled_y - cross_size,
            scaled_x, scaled_y + cross_size
        ], fill=(255, 255, 255), width=2)
    
    return img_copy

def safe_get_action_text(action, step, default=""):
    """Get answer tokens and thinking tokens separately (prompt is displayed separately in textbox)."""
    response_obj = safe_get_response_object(step)
    if response_obj:
        # Get the full raw response
        raw_response = getattr(response_obj, 'raw_response', '')

        # Get thinking tokens if available (stored separately in answering_tokens)
        answering_tokens = getattr(response_obj, 'answering_tokens', {})
        thinking_content = answering_tokens.get('thinking', '')

        # Remove thinking tokens from raw_response to get answer tokens only
        import re
        if thinking_content and thinking_content.strip():
            # Remove <think>...</think> blocks
            answer_tokens = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL)
            # Also handle case where thinking doesn't have opening tag (auto-start)
            answer_tokens = re.sub(r'^.*?</think>\s*', '', answer_tokens, flags=re.DOTALL)
            answer_tokens = answer_tokens.strip()
        else:
            answer_tokens = raw_response.strip()

        # Build display text with thinking tokens if available
        if thinking_content and thinking_content.strip():
            # Display thinking tokens in a separate section
            display_text = f"<<GREEN>>Thinking Tokens:<</GREEN>>\n{thinking_content.strip()}\n\n<<BLUE>>Answer Tokens:<</BLUE>>\n{answer_tokens}"
        elif answer_tokens:
            display_text = f"<<BLUE>>Answer Tokens:<</BLUE>>\n{answer_tokens}"
        else:
            return "No response available"

        return display_text
    return "No response available"

def safe_get_step_attribute(step, attr_name, default=None):
    """Safely get an attribute from a step. Now just wraps safe_get_attr."""
    return safe_get_attr(step, attr_name, default)

def safe_get_reward_info(step):
    """Extract reward, evaluation, and blocking info from a step."""
    reward_obj = safe_get_attr(step, 'reward')
    if reward_obj:
        reward = safe_get_attr(reward_obj, 'reward', 0)
        evaluation = safe_get_attr(reward_obj, 'evaluation', "")
        is_blocked = safe_get_attr(reward_obj, 'is_blocked', False)
        return reward, evaluation, is_blocked
    return 0, "", False

def format_evaluation_info_new_format(evaluator_ref, eval_info):
    """Format evaluation info for new format (with 'facts' key)."""
    formatted_eval = []
    eval_idx = 0

    for group_idx, group in enumerate(evaluator_ref, 1):
        group_desc = safe_get_attr(group, 'description', 'N/A')
        formatted_eval.append(f"Group {group_idx}: {group_desc}")

        facts = safe_get_attr(group, 'facts', [])
        num_facts = len(facts) if isinstance(facts, list) else 0

        for fact_num in range(1, num_facts + 1):
            if eval_idx < len(eval_info):
                fact_desc = facts[fact_num - 1] if fact_num <= len(facts) else 'N/A'
                formatted_eval.append(f"Fact {fact_num}: {fact_desc}")
                formatted_eval.append(str(eval_info[eval_idx]))
                formatted_eval.append("")
                eval_idx += 1

        formatted_eval.append("")

    return "\n".join(formatted_eval).strip()

def format_evaluation_info_old_format(evaluator_ref, eval_info):
    """Format evaluation info for old format (group embedded in description)."""
    import re
    group_to_facts = {}
    group_descriptions = {}

    for idx, item in enumerate(evaluator_ref):
        desc = safe_get_attr(item, 'description', '')
        match = re.match(r'\[Group (\d+): ([^\]]+)\] (.+)', desc)
        if match:
            group_num = int(match.group(1))
            group_desc = match.group(2).strip()
            fact_desc = match.group(3).strip()

            if group_num not in group_to_facts:
                group_to_facts[group_num] = []
                group_descriptions[group_num] = group_desc
            group_to_facts[group_num].append((idx, fact_desc))

    formatted_eval = []
    for group_num in sorted(group_to_facts.keys()):
        group_desc = group_descriptions.get(group_num, 'N/A')
        formatted_eval.append(f"Group {group_num}: {group_desc}")

        for fact_num, (fact_idx, fact_desc) in enumerate(group_to_facts[group_num], 1):
            if fact_idx < len(eval_info):
                formatted_eval.append(f"Fact {fact_num}: {fact_desc}")
                formatted_eval.append(str(eval_info[fact_idx]))
                formatted_eval.append("")

        formatted_eval.append("")

    return "\n".join(formatted_eval).strip()

def format_evaluation_info(eval_info, evaluator_ref_for_grouping):
    """Format evaluation info based on evaluator_reference structure.

    Handles both legacy format and new format with separate Criterion A/B:
    - [Criterion B - Anti-Hallucination]: Task-level response verification (checked once)
    - [Criterion A - Fact N]: Per-fact verification (checked for each rubric)
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
            # Legacy format (no prefix)
            legacy_evals.append(eval_str)

    # If we have the new format with Criterion A/B
    if criterion_b_eval or criterion_a_evals:
        # Show Criterion B first (task-level anti-hallucination check)
        if criterion_b_eval:
            formatted_parts.append("=" * 60)
            formatted_parts.append("📋 TASK-LEVEL EVALUATION (Criterion B - Anti-Hallucination)")
            formatted_parts.append("Checks if the agent's response is supported by screenshots")
            formatted_parts.append("=" * 60)
            # Remove the prefix for cleaner display
            criterion_b_content = criterion_b_eval.replace("[Criterion B - Anti-Hallucination] ", "")
            formatted_parts.append(criterion_b_content)
            formatted_parts.append("")

        # Show Criterion A evaluations (per-fact verification)
        if criterion_a_evals:
            formatted_parts.append("=" * 60)
            formatted_parts.append("📋 PER-FACT EVALUATIONS (Criterion A - Fact Verification)")
            formatted_parts.append("Checks if screenshots contain evidence for each fact")
            formatted_parts.append("=" * 60)

            for i, eval_str in enumerate(criterion_a_evals, 1):
                # Extract fact number and content
                # Format: "[Criterion A - Fact N] content"
                import re
                match = re.match(r'\[Criterion A - Fact (\d+)\] (.*)', eval_str, re.DOTALL)
                if match:
                    fact_num = match.group(1)
                    content = match.group(2)

                    # Get corresponding rubric description if available
                    rubric_desc = ""
                    if evaluator_ref_for_grouping and isinstance(evaluator_ref_for_grouping, list):
                        idx = int(fact_num) - 1
                        if 0 <= idx < len(evaluator_ref_for_grouping):
                            rubric = evaluator_ref_for_grouping[idx]
                            rubric_desc = safe_get_attr(rubric, 'description', '') if isinstance(rubric, dict) else str(rubric)

                    formatted_parts.append(f"\n--- Fact {fact_num} ---")
                    if rubric_desc:
                        formatted_parts.append(f"Rubric: {rubric_desc[:100]}..." if len(rubric_desc) > 100 else f"Rubric: {rubric_desc}")
                    formatted_parts.append(content)
                else:
                    formatted_parts.append(eval_str)
            formatted_parts.append("")

        # Show Reference Answer evaluation if present
        if reference_answer_eval:
            formatted_parts.append("=" * 60)
            formatted_parts.append("📋 REFERENCE ANSWER EVALUATION")
            formatted_parts.append("=" * 60)
            ref_content = reference_answer_eval.replace("[Reference Answer Evaluation] ", "")
            formatted_parts.append(ref_content)

        return "\n".join(formatted_parts).strip()

    # Legacy format handling (no Criterion A/B prefixes)
    if legacy_evals:
        if not evaluator_ref_for_grouping or not isinstance(evaluator_ref_for_grouping, list):
            # No evaluator_reference, display sequentially
            formatted = [f"Evaluation {i}:\n{str(evaluation)}\n" for i, evaluation in enumerate(legacy_evals, 1)]
            return "\n".join(formatted).strip()

        # Check format: NEW (has 'facts') vs OLD (no 'facts')
        first_item = evaluator_ref_for_grouping[0]
        has_facts = isinstance(first_item, dict) and 'facts' in first_item if first_item else False

        if has_facts:
            return format_evaluation_info_new_format(evaluator_ref_for_grouping, legacy_evals)
        else:
            return format_evaluation_info_old_format(evaluator_ref_for_grouping, legacy_evals)

    return ""

def get_ac_tree_only(traj_id, step_id):
    """Get only the accessibility tree for a specific trajectory and step."""
    traj_id = int(traj_id)
    step_id = int(step_id)
    
    if 0 <= traj_id < len(data):
        traj = data[traj_id]
        
        if len(traj) == 0:
            return "Empty trajectory"
        
        # Get accessibility tree for the specified step
        if 0 <= step_id < len(traj):
            step = traj[step_id]
            observation = safe_get_step_attribute(step, 'observation')
            ac_tree = safe_get_ac_tree(observation, f"No accessibility tree available for step {step_id}")
            return ac_tree
        else:
            return f"Invalid step ID. Trajectory has {len(traj)} steps (0-{len(traj)-1})"
    else:
        return "Invalid trajectory ID"

def display_trajectory(traj_id, step_id=0):
    traj_id = int(traj_id)
    step_id = int(step_id)
    
    if 0 <= traj_id < len(data):
        traj = data[traj_id]
        
        if len(traj) == 0:
            return ["Empty trajectory", "", "", "", "", "No images available", "Empty trajectory", "N/A", "N/A", "N/A", "N/A", "N/A"]
        
        # Get task information from first step
        first_observation = safe_get_step_attribute(traj[0], 'observation')
        composite_task_id = safe_get_task_info(first_observation, "Unknown task")

        # Extract just the task_name for display
        task_obj = safe_get_observation_attribute(first_observation, 'task')
        task_name = "Unknown task"
        if task_obj:
            task_name = getattr(task_obj, 'task_name', None) or (task_obj.get('task_name', 'Unknown task') if hasattr(task_obj, 'get') else 'Unknown task')

        # Get additional task metadata using composite task ID
        difficulty, domain, subdomain, website, evaluator_reference = get_task_metadata(composite_task_id)
        
        # Get answer and reward from last step
        last_step = traj[-1]

        # Check if action is an "answer" action by examining the action object
        action_obj = safe_get_step_attribute(last_step, 'action')
        answer = None

        # Try to extract answer from action object
        if action_obj and hasattr(action_obj, 'action'):
            action_info = getattr(action_obj, 'action', {})
            if isinstance(action_info, dict):
                action_key = action_info.get('key', '')
                action_args = action_info.get('arguments', {})

                if action_key == 'answer':
                    # Extract answer content from arguments
                    answer = action_args.get('content', '')

        # Fallback: check raw response for "ANSWER" keyword
        if not answer:
            action_text = safe_get_action_text(action_obj, last_step, "No answer")
            if "ANSWER" in action_text.upper():
                # Try to extract answer after ANSWER keyword
                parts = action_text.split("ANSWER")
                if len(parts) > 1:
                    answer = parts[-1].strip()
                    # Clean up any remaining brackets or formatting
                    answer = answer.strip("[]").strip()

        # Default message if no answer found
        if not answer or not answer.strip():
            answer = "Not answered in this trajectory (agent thinks it needs more steps to finish this task)."
        
        # Extract reward, evaluation, and blocking info from last step only
        reward_val, eval_info, is_blocked = safe_get_reward_info(last_step)

        # Format evaluation info nicely
        # Get evaluator_reference from trajectory for grouping eval_info
        evaluator_ref_for_grouping = safe_get_nested(traj[0], 'observation', 'task', 'evaluator_reference')
        eval_info = format_evaluation_info(eval_info, evaluator_ref_for_grouping)

        # Format blocking status
        blocked_status = "🚫 YES - Website blocked the agent" if is_blocked else "✅ NO - Website accessible"

        # Also check for trajectory_reward on first step (if added during post-processing)
        trajectory_reward = safe_get_step_attribute(traj[0], 'trajectory_reward', None)
        if trajectory_reward is not None:
            reward = f"Traj reward: {trajectory_reward}"
        else:
            # Use the last step reward as trajectory reward
            reward = f"Traj reward: {reward_val}"

        # Get accessibility tree for the specified step
        ac_tree = get_ac_tree_only(traj_id, step_id)

        images = []
        for step_id_img in range(len(traj)):
            step = traj[step_id_img]
            observation = safe_get_step_attribute(step, 'observation')
            
            # Get image path
            image_path = safe_get_image_path(observation)
            
            # Try different path variations if the original doesn't exist
            if image_path and not os.path.exists(image_path):
                # Try the path modifications from the original script
                alt_path1 = image_path.replace("/scratch/azureml/cr/j/69aa558683564d0dbaf7c667b335e060/cap/data-capability/wd/output_dir", "/data/haobai/mnt/haobai/logs/webgym/")
                alt_path2 = image_path.replace("/home", "/data")
                
                if os.path.exists(alt_path1):
                    image_path = alt_path1
                elif os.path.exists(alt_path2):
                    image_path = alt_path2
            
            if image_path and os.path.exists(image_path):
                try:
                    img = Image.open(image_path)
                    img = img.convert("RGB")
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    # Create a placeholder image
                    img = Image.new('RGB', (800, 600), (200, 200, 200))
                    draw = ImageDraw.Draw(img)
                    draw_text_with_fallback(draw, (10, 10), f"Could not load image:\n{image_path}", fill=(0, 0, 0))
            else:
                # Create a placeholder image if no image path or file doesn't exist
                img = Image.new('RGB', (800, 600), (200, 200, 200))
                draw = ImageDraw.Draw(img)
                draw_text_with_fallback(draw, (10, 10), f"No image available for step {step_id_img}", fill=(0, 0, 0))
                print(f"Image path: {image_path}")

            # Get submit field from reward (new) or observation (legacy)
            submit_value = safe_get_submit_field(step)

            # Get submission judgment if available
            submission_judgment = safe_get_submission_judgment(step)
            submission_judgment_text = ""
            if submission_judgment:
                # Truncate if too long (show first 200 chars)
                judgment_preview = submission_judgment[:200] + "..." if len(submission_judgment) > 200 else submission_judgment
                submission_judgment_text = f"\n<<GREEN>>Submission Judgment:<</GREEN>> {judgment_preview}"

            # Check if screenshot is different from next step
            same_as_next = safe_get_step_attribute(step, 'same_as_next_screenshot', None)
            if same_as_next is None:
                # Legacy trajectory without this field
                screenshot_status = "<<BLUE>>Screenshot Different from Next:<</BLUE>> Unknown (legacy trajectory)"
            elif same_as_next:
                # Screenshot is the same as next step (no visual change)
                screenshot_status = "<<BLUE>>Screenshot Different from Next:<</BLUE>> NO (page unchanged)"
            else:
                # Screenshot is different from next step (visual change occurred)
                screenshot_status = "<<BLUE>>Screenshot Different from Next:<</BLUE>> YES (page changed)"

            # Prepare action text (now includes Response object data)
            action = safe_get_step_attribute(step, 'action')
            action_text = safe_get_action_text(action, step, f"No action for step {step_id_img}")

            action_coordinates = extract_action_coordinates(action)
            if action_coordinates:
                img = draw_action_coordinates(img, action_coordinates)

            # Combine submit field, submission judgment, screenshot status, and action text
            combined_text = f"Step {step_id_img}\n<<BLUE>>Submit:<</BLUE>> {submit_value}{submission_judgment_text}\n{screenshot_status}\n\n{action_text}"
            
            lines = wrap_text_preserving_whitespace(combined_text, max_width=70)

            # Font fallback system handles font selection automatically
            
            text_bg_height = len(lines) * 35 + 100  # Increased height for additional text

            img_with_text_bg = Image.new('RGB', (img.width, img.height + text_bg_height), (255, 255, 255))
            img_with_text_bg.paste(img, (0, 0))

            draw = ImageDraw.Draw(img_with_text_bg)
            text_y = img.height + 10  # Add some padding
            
            for line in lines:
                # Left-align text instead of centering
                text_x = 10  # Fixed left margin
                
                # Check if line contains blue or green markers and handle appropriately
                if ("<<BLUE>>" in line and "<</BLUE>>" in line) or ("<<GREEN>>" in line and "<</GREEN>>" in line):
                    # Parse color markers
                    current_x = text_x
                    remaining_line = line
                    
                    # Handle both blue and green markers
                    color_patterns = [
                        ("<<BLUE>>", "<</BLUE>>", (0, 0, 255)),   # Blue
                        ("<<GREEN>>", "<</GREEN>>", (0, 128, 0))  # Green
                    ]
                    
                    while any(start_marker in remaining_line and end_marker in remaining_line 
                             for start_marker, end_marker, _ in color_patterns):
                        
                        # Find the earliest color marker
                        earliest_pos = len(remaining_line)
                        earliest_color = None
                        
                        for start_marker, end_marker, color in color_patterns:
                            if start_marker in remaining_line and end_marker in remaining_line:
                                start_pos = remaining_line.find(start_marker)
                                end_pos = remaining_line.find(end_marker)
                                if start_pos >= 0 and end_pos > start_pos and start_pos < earliest_pos:
                                    earliest_pos = start_pos
                                    earliest_color = (start_marker, end_marker, color)
                        
                        if earliest_color is None:
                            break
                            
                        start_marker, end_marker, color = earliest_color
                        start_pos = remaining_line.find(start_marker)
                        end_pos = remaining_line.find(end_marker)
                        
                        # Draw text before color marker
                        if start_pos > 0:
                            before_text = remaining_line[:start_pos]
                            draw_text_with_fallback(draw, (current_x, text_y), before_text, fill=(0, 0, 0))
                            try:
                                # Use a base font for width measurement
                                base_font = pick_font_for_char("A")
                                if base_font:
                                    before_width = draw.textlength(before_text, font=base_font)
                                else:
                                    before_width = len(before_text) * 10
                            except:
                                before_width = len(before_text) * 10
                            current_x += before_width
                        
                        # Draw colored text
                        colored_text = remaining_line[start_pos + len(start_marker):end_pos]
                        draw_text_with_fallback(draw, (current_x, text_y), colored_text, fill=color)
                        try:
                            # Use a base font for width measurement
                            base_font = pick_font_for_char("A")
                            if base_font:
                                colored_width = draw.textlength(colored_text, font=base_font)
                            else:
                                colored_width = len(colored_text) * 10
                        except:
                            colored_width = len(colored_text) * 10
                        current_x += colored_width
                        
                        # Continue with remaining text
                        remaining_line = remaining_line[end_pos + len(end_marker):]
                    
                    # Draw any remaining text
                    if remaining_line:
                        draw_text_with_fallback(draw, (current_x, text_y), remaining_line, fill=(0, 0, 0))
                else:
                    # No color markers, draw normally
                    draw_text_with_fallback(draw, (text_x, text_y), line, fill=(0, 0, 0))
                text_y += 35

            # Add red border to the image with text
            border_width = 5
            img_with_border = Image.new('RGB', (img_with_text_bg.width + 2 * border_width, img_with_text_bg.height + 2 * border_width), (255, 0, 0))
            img_with_border.paste(img_with_text_bg, (border_width, border_width))

            images.append(img_with_border)

        if images:
            max_images_per_row = 2
            num_rows = (len(images) + max_images_per_row - 1) // max_images_per_row
            row_images = []
            max_row_width = 0
            total_height = 0

            for row in range(num_rows):
                start_index = row * max_images_per_row
                end_index = min(start_index + max_images_per_row, len(images))
                row_images_subset = images[start_index:end_index]
                total_width = sum(img.width for img in row_images_subset)
                max_height = max(img.height for img in row_images_subset)
                row_image = Image.new('RGB', (total_width, max_height))
                x_offset = 0
                for img in row_images_subset:
                    row_image.paste(img, (x_offset, 0))
                    x_offset += img.width
                row_images.append(row_image)
                max_row_width = max(max_row_width, total_width)
                total_height += max_height

            combined_image = Image.new('RGB', (max_row_width, total_height))
            y_offset = 0
            for row_img in row_images:
                combined_image.paste(row_img, (0, y_offset))
                y_offset += row_img.height

            return [task_name, answer, reward, blocked_status, eval_info, combined_image, ac_tree, difficulty, domain, subdomain, website, evaluator_reference]
        else:
            return [task_name, answer, reward, blocked_status, eval_info, "No images available", ac_tree, difficulty, domain, subdomain, website, evaluator_reference]

    else:
        return ["Invalid trajectory ID", "", "", "", "", "No images available", "Invalid trajectory ID", "N/A", "N/A", "N/A", "N/A", "N/A"]

def get_step_detailed_info(traj_id, step_id):
    """Get raw response and submission judgment for a specific step."""
    traj_id = int(traj_id)
    step_id = int(step_id)

    if 0 <= traj_id < len(data):
        traj = data[traj_id]
        if len(traj) == 0:
            return "Empty trajectory"
        if 0 <= step_id < len(traj):
            step = traj[step_id]

            # Get raw response
            response_obj = safe_get_response_object(step)
            raw_response = ""
            if response_obj:
                raw_response = getattr(response_obj, 'raw_response', '')
                if not raw_response or not raw_response.strip():
                    raw_response = 'No raw response'
            else:
                raw_response = "No Response object"

            # Get submission judgment
            submit_value = safe_get_submit_field(step)
            submission_judgment = safe_get_submission_judgment(step)

            # Build combined output
            output = f"=== RAW MODEL RESPONSE ===\n{raw_response.strip()}\n\n"
            output += f"=== SUBMISSION INFO ===\n"
            output += f"Submit: {submit_value}\n"
            if submission_judgment:
                output += f"\nSubmission Judgment:\n{submission_judgment}"
            else:
                output += f"\nSubmission Judgment: (not available)"

            return output
        return f"Invalid step ID. Trajectory has {len(traj)} steps (0-{len(traj)-1})"
    return "Invalid trajectory ID"

def get_step_prompt(traj_id, step_id):
    """Get raw prompt for a specific step (only if --show-prompt is enabled)."""
    if not args.show_prompt:
        return "Use --show-prompt flag to view prompts"

    traj_id = int(traj_id)
    step_id = int(step_id)

    if 0 <= traj_id < len(data):
        traj = data[traj_id]
        if len(traj) == 0:
            return "Empty trajectory"
        if 0 <= step_id < len(traj):
            step = traj[step_id]
            response_obj = safe_get_response_object(step)
            if response_obj:
                raw_prompt = getattr(response_obj, 'raw_prompt', '')
                if raw_prompt and raw_prompt.strip():
                    # Try to pretty-print if it's JSON
                    try:
                        import json
                        prompt_json = json.loads(raw_prompt)
                        return json.dumps(prompt_json, indent=2, ensure_ascii=False)
                    except:
                        # Not JSON, return as-is
                        return raw_prompt.strip()
                return 'Prompt not saved (old trajectory or error during saving).'
            return "No Response object"
        return f"Invalid step ID. Trajectory has {len(traj)} steps (0-{len(traj)-1})"
    return "Invalid trajectory ID"

with gr.Blocks() as interface:
    gr.Markdown(f"# WebGym Trajectory Viewer - {args.split.upper()} Split")
    gr.Markdown(f"Total trajectories: {len(data)}, Successful: {len(successful_traj_ids)}")
    gr.Markdown(f"Some successful trajectory IDs: {successful_traj_ids}")

    with gr.Row():
        traj_id_input = gr.Number(label="Trajectory ID", value=0)
        step_id_input = gr.Number(label="Step ID", value=0)
        reward_output = gr.Textbox(label="Reward")
        blocked_output = gr.Textbox(label="Website Blocked?")

    with gr.Row():
        task_output = gr.Textbox(label="Task")
    
    # New row for task metadata
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
        images_output = gr.Image(label="Images")

    with gr.Row():
        answer_output = gr.Textbox(label="Answer")
        eval_info_output = gr.Textbox(label="Eval Info")

    gr.Markdown("## Step-Specific Details")
    gr.Markdown("Shows the prompt, response, and submission judgment for the selected step (use --show-prompt flag to enable prompt display)")
    gr.Markdown("**Note:** Prompts are saved with `file://` paths instead of base64 images to reduce file size.")

    with gr.Row():
        raw_prompt_output = gr.Textbox(label="Raw Model Prompt (images as file:// paths)", lines=15, max_lines=30)

    with gr.Row():
        raw_response_output = gr.Textbox(label="Raw Model Response & Submission Judgment", lines=15, max_lines=30)
    
    # Function to handle trajectory ID updates (updates everything)
    def update_display(traj_id, step_id):
        return display_trajectory(traj_id, step_id)
    
    # Function to handle step ID updates (updates step-specific info)
    def update_step_info(traj_id, step_id):
        # Get AC tree
        ac_tree = get_ac_tree_only(traj_id, step_id)

        # Get raw prompt (only if --show-prompt flag is enabled)
        raw_prompt = get_step_prompt(traj_id, step_id)

        # Get raw response
        raw_response = get_step_detailed_info(traj_id, step_id)

        return ac_tree, raw_prompt, raw_response
    
    # Update when trajectory ID changes (refreshes everything)
    traj_id_input.submit(
        fn=lambda traj_id, step_id: (
            *update_display(traj_id, step_id),  # Unpack trajectory display results
            *update_step_info(traj_id, step_id)  # Unpack step info results
        ),
        inputs=[traj_id_input, step_id_input],
        outputs=[
            task_output,
            answer_output,
            reward_output,
            blocked_output,
            eval_info_output,
            images_output,
            ac_tree_output,  # From trajectory display
            difficulty_output,
            benchmark_name_output,
            subdomain_output,
            website_output,
            evaluator_reference_output,
            # Step-specific outputs
            ac_tree_output,  # Updated AC tree
            raw_prompt_output,  # Updated prompt
            raw_response_output  # Updated response
        ]
    )

    # Update when step ID changes (only updates step-specific info)
    step_id_input.submit(
        fn=update_step_info,
        inputs=[traj_id_input, step_id_input],
        outputs=[
            ac_tree_output,
            raw_prompt_output,
            raw_response_output
        ]
    )

interface.launch(share=True)