# webgym/models/qwen/conversation_builder.py
from typing import List, Dict, Any
from ..base.conversation_builder import ConversationBuilder

class QwenConversationBuilder(ConversationBuilder):
    """Multi-turn conversation builder for Qwen3-VL"""

    def __init__(self, interaction_mode: str, variant: str = "instruct", prompt_version: str = "vanilla"):
        """
        Initialize Qwen conversation builder

        Args:
            interaction_mode: 'coordinates' or 'set_of_marks'
            variant: 'instruct' or 'thinking'
            prompt_version: 'vanilla' or 'complete'
        """
        self.interaction_mode = interaction_mode
        self.variant = variant.lower()
        self.prompt_version = prompt_version.lower()

        if self.variant not in ['instruct', 'thinking']:
            raise ValueError(f"Invalid variant: {variant}. Must be 'instruct' or 'thinking'")

        if self.prompt_version not in ['vanilla', 'complete']:
            raise ValueError(f"Invalid prompt_version: {prompt_version}. Must be 'vanilla' or 'complete'")

    def build_conversation(self, task: str, trajectory: List[Dict], current_observation: Dict, **kwargs) -> List[Dict]:
        """
        Build multi-turn conversation history with 4-round sliding window

        Following OSWorld desktop computer-use approach:
        - Keep last 4 rounds as explicit history (with images)
        - Summarize anything older than 4 rounds as text in first user message
        - Add current screenshot as final user message
        - Total: 5 images (4 historical + 1 current)

        Args:
            task: Task description
            trajectory: List of trajectory steps
            current_observation: Current observation dict

        Returns:
            List of message dicts (system, user, assistant alternating)
        """
        messages = []
        HISTORY_WINDOW = 4  # Keep last 4 rounds as explicit history

        # 1. System message with tool definition
        messages.append(self._build_system_message())

        num_steps = len(trajectory)

        # Determine which steps go into rolling history (last 4 rounds + current observation = 5 images)
        # Note: A "round" is (observation, response) pair. We need 4 complete rounds + current observation.
        # This gives us 5 observations total: [obs_n-4, obs_n-3, obs_n-2, obs_n-1, obs_current]
        if num_steps <= HISTORY_WINDOW + 1:
            # All steps fit in window (need HISTORY_WINDOW+1 to account for 4 rounds + current)
            rolling_history_start = 0
            older_steps = []
        else:
            # Split: older steps get summarized, last 5 observations (4 rounds + current) get full treatment
            rolling_history_start = num_steps - HISTORY_WINDOW - 1
            older_steps = trajectory[:rolling_history_start]

        # 2. First user message: task + website + summary of older history (if any)
        website = kwargs.get('website', '')
        if website:
            first_user_text = f"Please generate the next action according to the UI screenshot and task.\n\nTask: {task}\n\nInitial website: {website}\n\n"
        else:
            first_user_text = f"Please generate the next action according to the UI screenshot and task.\n\nTask: {task}\n\n"

        if older_steps:
            # Summarize older steps (focus on state/progress/blockers, not visual details)
            summary = self._summarize_older_history(older_steps)
            first_user_text += f"Previous actions summary:\n{summary}\n\n"

        first_user_text += "Generate the next action to complete the task."

        # Add first user message (no image yet - will be added with rolling history or current)
        if num_steps == 0:
            # No trajectory yet, just task description
            messages.append({
                "role": "user",
                "content": first_user_text
            })
        else:
            # Will add with first rolling history step
            pass

        # 3. Rolling history (last 4 rounds with images)
        for step_idx in range(rolling_history_start, num_steps):
            observation = trajectory[step_idx].get('observation')
            response = trajectory[step_idx].get('response')

            if observation:
                # Build user message with image
                if step_idx == rolling_history_start:
                    # First rolling history step - include task and summary
                    user_msg = self._build_user_message_with_text(
                        observation=observation,
                        text=first_user_text
                    )
                else:
                    # Subsequent steps - just image, no additional text (matches OSWorld)
                    user_msg = self._build_user_message_image_only(observation)
                messages.append(user_msg)

            if response:
                # Assistant message: raw response text
                assistant_msg = self._build_assistant_message(response)
                messages.append(assistant_msg)

        # 4. Current observation (final user message with current screenshot)
        # trajectory[-1] contains the current observation, so it's already included in the loop above

        return messages

    def _build_system_message(self) -> Dict:
        """Build system message with tool definition and response format"""

        if self.interaction_mode == 'coordinates':
            tool_def = self._get_computer_use_tool_def()
        else:
            tool_def = self._get_set_of_marks_tool_def()

        # Add response format based on prompt_version
        response_format = ""
        if self.prompt_version == "complete":
            # Complete version: includes Progress, Intention, Action, and tool call
            response_format = """

# Response format

Response format for every step:
1) Memory: facts you would like to memorize for future actions in json format. Include the current step.
2) Progress: Decompose the task into subtasks and what has been finished so far with json format. Include progress of the current step.
3) Intention: clearly state which subtask you're working on at this step with the json key.
4) Action: a short sentence describing what to do in the UI to accomplish the next subtask.
5) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.

Rules:
- Output exactly in the order: Memory, Progress, Intention, Action, <tool_call>.
- You MUST use json format for the Memory and Progress parts.
- Example Task: "Search and compare the prices and locations of product 1 and product 2 on Amazon."
  - Example of Memory json format: {"Price of product 1": "10.00", "Location of product 1": "10.00", "Price of produce 2": "12.00"}.
  - Example of Progress json format: {"Go to Amazon.com": "finished", "Search for price of product 1": "finished", "Search for location of product 1": "finished", "Search for price of product 2": "finished", "Search for location of product 2": "not finished", "Compare product 1 and product 2": "not finished"}.
  - Example of Intention json key format: "Search for location of product 2".
- You CAN NOT modify previous Memory. Only append to it.
- You CAN modify Progress from previous conversation to further decompose the task and guide your next action.
  - For example, if the previous assistant message specifies Progress: {"Go to Amazon.com": "finished", "Search for product 1": "finished", "Search for product 2": "not finished", "Compare product 1 and 2": "not finished"},
  - You should further decompose "Search for product 1" and "Search for product 2" into "Search for price of product 1" and "Search for location of product 1", and "Search for price of product 2" and "Search for location of product 2".
- Do not output anything else outside those five parts."""
        else:
            # Vanilla version: minimal format without Thoughts or Memory
            response_format = """

# Response format

Response format for every step:
1) Action: a short sentence describing what to do in the UI.
2) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.

Rules:
- Output exactly in the order: Action, <tool_call>.
- Action describes the high-level intention of the tool call within a single sentence.
- Do not output anything else outside those two parts."""

        system_content = f"""You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_def}
</tools>

For each function call, return a JSON object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>{response_format}"""

        return {
            "role": "system",
            "content": system_content
        }

    def _build_user_message(self, task: str, observation: Any, step_idx: int) -> Dict:
        """Build user message with screenshot and query"""

        screenshot_path = observation.image_path

        # First message includes full task
        if task and step_idx == 0:
            text = f"Your task is: {task}\n\nPlease analyze the current screenshot and decide your next action."
        else:
            # Subsequent messages include observation of previous action
            page_title = observation.page_metadata.get('title', 'Page loaded') if hasattr(observation, 'page_metadata') else 'Page loaded'
            text = f"Observation: {page_title}\n\nPlease analyze the current screenshot and decide your next action."

        # Use file:// URL for vLLM (more efficient than base64)
        from webgym.utils import encode_image_to_file_url
        image_url = encode_image_to_file_url(screenshot_path)

        return {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": text}
            ]
        }

    def _build_user_message_text_only(self, task: str, observation: Any, step_idx: int) -> Dict:
        """Build user message with TEXT ONLY (no image) for historical steps"""

        # First message includes full task
        if task and step_idx == 0:
            text = f"Your task is: {task}\n\nPlease analyze the screenshot and decide your next action."
        else:
            # Subsequent messages include observation of previous action
            page_title = observation.page_metadata.get('title', 'Page loaded') if hasattr(observation, 'page_metadata') else 'Page loaded'
            text = f"Observation: {page_title}"

        return {
            "role": "user",
            "content": text  # Text only, no image
        }

    def _build_user_message_with_text(self, observation: Any, text: str) -> Dict:
        """Build user message with both image and custom text"""
        # Handle both dict (current_observation) and object (trajectory observation)
        if isinstance(observation, dict):
            screenshot_path = observation['image_path']
        else:
            screenshot_path = observation.image_path

        # Use file:// URL for vLLM (more efficient than base64)
        from webgym.utils import encode_image_to_file_url
        image_url = encode_image_to_file_url(screenshot_path)

        return {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": text}
            ]
        }

    def _build_user_message_image_only(self, observation: Any) -> Dict:
        """Build user message with only image (no text) - matches OSWorld style"""
        # Handle both dict (current_observation) and object (trajectory observation)
        if isinstance(observation, dict):
            screenshot_path = observation['image_path']
        else:
            screenshot_path = observation.image_path

        # Use file:// URL for vLLM (more efficient than base64)
        from webgym.utils import encode_image_to_file_url
        image_url = encode_image_to_file_url(screenshot_path)

        return {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }

    def _summarize_older_history(self, older_steps: List[Dict]) -> str:
        """
        Create compact text summary of steps older than 4-round window.
        Focus on actions taken, avoid stale visual details or coordinates.
        Matches OSWorld "Previous actions:" format.
        """
        if not older_steps:
            return ""

        summary_lines = []
        for i, step in enumerate(older_steps):
            observation = step.get('observation')
            action = step.get('action')
            response = step.get('response')

            if action and response:
                # Get webpage name from observation
                webpage_name = "Unknown page"
                if observation and hasattr(observation, 'page_metadata'):
                    webpage_name = observation.page_metadata.get('title', 'Unknown page')

                # Get action description from response (the "Action:" field)
                action_desc = response.answering_tokens.get('action', '')

                # Fallback: if no action description, construct from action details
                if not action_desc:
                    action_key = action.action.get('key', 'unknown')
                    action_args = action.action.get('arguments', {})

                    # Create more informative fallback description
                    if action_key == 'click' and 'element_id' in action_args:
                        action_desc = f"Click element {action_args['element_id']}"
                    elif action_key == 'type':
                        if 'element_id' in action_args:
                            text = action_args.get('content', '')[:30]  # First 30 chars
                            action_desc = f"Type '{text}' into element {action_args['element_id']}"
                        elif 'text' in action_args:
                            text = action_args.get('text', '')[:30]
                            action_desc = f"Type '{text}'"
                    elif action_key == 'scroll':
                        direction = action_args.get('direction', 'down')
                        action_desc = f"Scroll {direction}"
                    elif action_key == 'answer':
                        content = action_args.get('content', '')[:50]
                        action_desc = f"Answer: {content}"
                    else:
                        action_desc = f"{action_key.capitalize()} action"

                # Get action effect (observation) - only tells if page changed or not
                observation_text = response.answering_tokens.get('observation', '')

                # Extract whether page changed (observation is generated by comparing screenshots)
                if observation_text:
                    if 'did not change' in observation_text:
                        effect = "page unchanged"
                    elif 'changed' in observation_text:
                        effect = "page changed"
                    else:
                        effect = "executed"
                else:
                    effect = "executed"

                # Create rich summary: Webpage, Action, Effect
                summary_lines.append(f"Step {i+1} on [{webpage_name}]: {action_desc} → {effect}")

        if not summary_lines:
            return "No previous actions."

        return "\n".join(summary_lines)

    def _build_assistant_message(self, response: Any) -> Dict:
        """Build assistant message from response

        For thinking variant, only answer tokens (without thinking) are included
        in conversation history. This prevents the model from seeing its previous
        thinking tokens.
        """

        # Get full response (thinking + answer tokens for thinking variant)
        raw_response = response.raw_response

        # For thinking variant, exclude thinking tokens from conversation history
        # Only include answer tokens so model doesn't see previous thinking
        if self.variant == 'thinking':
            import re
            # Remove <think>...</think> blocks to get answer tokens only
            answer_tokens = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL)
            # Also handle case where thinking doesn't have opening tag (auto-start)
            answer_tokens = re.sub(r'^.*?</think>\s*', '', answer_tokens, flags=re.DOTALL)
            answer_tokens = answer_tokens.strip()
        else:
            # For non-thinking variants, all tokens are answer tokens
            answer_tokens = raw_response

        return {
            "role": "assistant",
            "content": answer_tokens
        }

    def _get_computer_use_tool_def(self) -> str:
        """Tool definition for coordinates mode"""
        return """{
  "name": "computer_use",
  "description": "Use a mouse and keyboard to interact with a computer. The screen's resolution is 1000x1000.\\n* You do not have access to download files or play videos.\\n* Focus on web browsing and navigation tasks only.",
  "parameters": {
    "type": "object",
    "properties": {
      "action": {
        "type": "string",
        "description": "The action to perform:\\n* `left_click`: Click the left mouse button at the specified coordinates.\\n* `type`: Type text at the specified coordinates. The system will automatically click at the coordinates, type the text, and press Enter.\\n* `scroll`: Scroll the page in the specified direction (up or down).\\n* `wait`: Wait for the specified number of seconds for changes to occur.\\n* `go_back`: Go back to the previous page in browser history.\\n* `navigate`: Navigate directly to a specific website URL. The URL must start with https://. CRITICAL: If you see reCAPTCHA or any CAPTCHA challenge on the screen, DO NOT attempt to solve it. Instead, immediately use the navigate action to go to a different relevant website to complete your task. Search engines (Bing, Google, DuckDuckGo) will block you with reCAPTCHA - bypass them by navigating directly to websites like wikipedia.org, news sites, company websites, government sites, etc. NEVER navigate to google.com.\\n* `answer`: Provide the final answer to complete the task.",
        "enum": ["left_click", "type", "scroll", "wait", "go_back", "navigate", "answer"]
      },
      "coordinate": {
        "type": "array",
        "description": "[x, y] coordinates (0-1000 range). Required for left_click and type actions. For type action, specify WHERE to type (e.g., coordinates of input field).",
        "items": {
          "type": "integer",
          "minimum": 0,
          "maximum": 1000
        },
        "minItems": 2,
        "maxItems": 2
      },
      "text": {
        "type": "string",
        "description": "Text to type or answer. Required for type and answer actions. Note: For type action, the system will automatically click at the coordinates, type the text, and press Enter - no need to click separately before typing."
      },
      "direction": {
        "type": "string",
        "enum": ["up", "down"],
        "description": "Scroll direction. Required for scroll action."
      },
      "time": {
        "type": "number",
        "description": "Seconds to wait. Required for wait action."
      },
      "url": {
        "type": "string",
        "description": "URL to navigate to. Required for navigate action. Must start with https://. When you encounter reCAPTCHA, use this to navigate away to a different website instead of trying to solve the CAPTCHA. Navigate to relevant websites that can help complete the task. Avoid google.com (will block you). Examples: wikipedia.org, news sites, company websites, government sites, etc. IMPORTANT: If a website fails to load (you see a 'Navigation failed' message), try the URL with www. added/removed: if the URL is 'https://example.com', try 'https://www.example.com'; if the URL is 'https://www.example.com', try 'https://example.com'."
      }
    },
    "required": ["action"]
  }
}"""

    def _get_set_of_marks_tool_def(self) -> str:
        """Tool definition for set_of_marks mode"""
        return """{
  "name": "web_interaction",
  "description": "Interact with web elements using their numerical labels shown on the screenshot.\\n* You do not have access to download files or play videos.\\n* Focus on web browsing and navigation tasks only.",
  "parameters": {
    "type": "object",
    "properties": {
      "action": {
        "type": "string",
        "description": "The action to perform:\\n* `click`: Click on the element with the specified numerical label.\\n* `type`: Type text into the element with the specified numerical label. The system will automatically click the element, type the text, and press Enter.\\n* `hover`: Hover the mouse over the element with the specified numerical label.\\n* `scroll`: Scroll the page in the specified direction (up or down).\\n* `wait`: Wait for the specified number of seconds for changes to occur.\\n* `go_back`: Go back to the previous page in browser history.\\n* `navigate`: Navigate directly to a specific website URL. The URL must start with https://. CRITICAL: If you see reCAPTCHA or any CAPTCHA challenge on the screen, DO NOT attempt to solve it. Instead, immediately use the navigate action to go to a different relevant website to complete your task. Search engines (Bing, Google, DuckDuckGo) will block you with reCAPTCHA - bypass them by navigating directly to websites like wikipedia.org, news sites, company websites, government sites, etc. NEVER navigate to google.com.\\n* `answer`: Provide the final answer to complete the task.",
        "enum": ["click", "type", "hover", "scroll", "wait", "go_back", "navigate", "answer"]
      },
      "element_id": {
        "type": "string",
        "description": "Numerical label of the element. Required for click, type, hover actions. For type action, specify WHICH element to type into."
      },
      "text": {
        "type": "string",
        "description": "Text to type or answer. Required for type and answer actions. Note: For type action, the system will automatically click the element, type the text, and press Enter - no need to click separately before typing."
      },
      "direction": {
        "type": "string",
        "enum": ["up", "down"],
        "description": "Scroll direction. Required for scroll action."
      },
      "time": {
        "type": "number",
        "description": "Seconds to wait. Required for wait action."
      },
      "url": {
        "type": "string",
        "description": "URL to navigate to. Required for navigate action. Must start with https://. When you encounter reCAPTCHA, use this to navigate away to a different website instead of trying to solve the CAPTCHA. Navigate to relevant websites that can help complete the task. Avoid google.com (will block you). Examples: wikipedia.org, news sites, company websites, government sites, etc. IMPORTANT: If a website fails to load (you see a 'Navigation failed' message), try the URL with www. added/removed: if the URL is 'https://example.com', try 'https://www.example.com'; if the URL is 'https://www.example.com', try 'https://example.com'."
      }
    },
    "required": ["action"]
  }
}"""

    def summarize_trajectory(self, trajectory: List[Dict]) -> str:
        """Summarize trajectory for evaluation - creates a simple text summary of actions and observations"""
        summary_lines = []

        for i, step in enumerate(trajectory):
            action = step.get('action')
            response = step.get('response')

            if action and response:
                action_str = action.action_string if action.action_string else ''
                observation = response.answering_tokens.get('observation', '')
                summary_lines.append(f"Step {i}: {action_str} | Observation: {observation}")

        return "\n".join(summary_lines) if summary_lines else "No trajectory available"

    def get_conversation_type(self) -> str:
        return "multi-turn"
