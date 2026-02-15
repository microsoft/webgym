# webgym/models/evaluator.py
import os
import base64
import re
from typing import List, Dict, Any, Tuple, Optional
from .base.evaluation_prompt import (
    evaluation_system_prompt, evaluation_user_prompt,
    criterion_a_system_prompt, criterion_a_user_prompt,
    criterion_b_system_prompt, criterion_b_user_prompt,
    blocking_detection_system_prompt, blocking_detection_user_prompt
)
from webgym.data.components import Reward


class Evaluator:
    """
    Evaluator for web agent trajectories.
    Uses OpenAI API to evaluate task completion and detect blocking.
    Supports different models/endpoints for each evaluation task.
    """

    # Task types for per-task configuration
    TASK_KEYPOINT_DETECTION = "keypoint_detection"
    TASK_BLOCKING_DETECTION = "blocking_detection"
    TASK_EVALUATION = "evaluation"  # Covers criterion_a, criterion_b, and reference_answer

    def __init__(self, openai_config: Dict, conversation_builder=None, max_retries: int = 1, verbose: bool = True):
        """
        Initialize Evaluator.

        Args:
            openai_config: Configuration dict with default settings and optional per-task overrides.
                Required keys:
                    - model: Default model name
                Optional keys:
                    - openai_api_key_env_var: Env var for API key (default: OPENAI_API_KEY)
                    - base_url: Base URL for API (default: None for OpenAI)
                    - keypoint_detection: Override config for keypoint/image relevance detection
                    - blocking_detection: Override config for blocking detection task
                    - evaluation: Override config for evaluation (criterion_a, criterion_b, reference_answer)
                Each override can specify: model, openai_api_key_env_var, base_url
            conversation_builder: ConversationBuilder instance for trajectory summarization
            max_retries: Maximum number of retries for API calls
            verbose: Whether to print progress messages
        """
        self.openai_config = openai_config
        self.conversation_builder = conversation_builder
        self.max_retries = max_retries
        self.verbose = verbose

        # Cache for clients (keyed by (api_key_env_var, base_url))
        self._clients = {}

        # Initialize clients for all configured tasks
        self._setup_clients()

    def _get_task_config(self, task_type: str) -> Dict:
        """Get merged configuration for a specific task type.

        Task-specific config overrides default config values.
        """
        # Start with default config
        default_config = {
            "model": self.openai_config.get("model"),
            "openai_api_key_env_var": self.openai_config.get("openai_api_key_env_var", "OPENAI_API_KEY"),
            "base_url": self.openai_config.get("base_url"),
        }

        # Get task-specific overrides if they exist
        task_overrides = self.openai_config.get(task_type, {})

        # Merge: task-specific values override defaults
        merged = default_config.copy()
        for key in ["model", "openai_api_key_env_var", "base_url"]:
            if key in task_overrides and task_overrides[key] is not None:
                merged[key] = task_overrides[key]

        return merged

    def _get_client_for_config(self, config: Dict):
        """Get or create a client for the given configuration."""
        from openai import OpenAI

        api_key_env_var = config["openai_api_key_env_var"]
        base_url = config.get("base_url")

        # Create cache key
        cache_key = (api_key_env_var, base_url)

        if cache_key not in self._clients:
            if api_key_env_var not in os.environ:
                raise ValueError(f"Environment variable {api_key_env_var} not found")

            client_kwargs = {"api_key": os.environ[api_key_env_var]}
            if base_url:
                client_kwargs["base_url"] = base_url

            self._clients[cache_key] = OpenAI(**client_kwargs)

        return self._clients[cache_key]

    def _get_client_and_model(self, task_type: str) -> Tuple[Any, str]:
        """Get the client and model name for a specific task type."""
        config = self._get_task_config(task_type)
        client = self._get_client_for_config(config)
        model = config["model"]
        return client, model

    def _setup_clients(self):
        """Setup clients for all task types and validate configuration."""
        # Validate default model exists
        if not self.openai_config.get("model"):
            raise ValueError("model must be specified in openai_config")

        # Pre-initialize clients for all task types to catch config errors early
        task_types = [
            self.TASK_KEYPOINT_DETECTION,
            self.TASK_BLOCKING_DETECTION,
            self.TASK_EVALUATION,
        ]

        if self.verbose:
            print("Initializing evaluator API clients:")

        for task_type in task_types:
            config = self._get_task_config(task_type)
            client = self._get_client_for_config(config)

            if self.verbose:
                base_url = config.get("base_url")
                provider = "Gemini" if base_url and "generativelanguage.googleapis.com" in base_url else "OpenAI"
                print(f"  {task_type}: {provider} ({config['model']})")

        # For backward compatibility, set default client and model
        default_config = self._get_task_config(self.TASK_EVALUATION)
        self.client = self._get_client_for_config(default_config)
        self.reward_model_name = default_config["model"]

    def get_verifiable_reward(self, trajectory: List[Dict]) -> Tuple[int, str, bool]:
        """Get verifiable reward and blocking status using OpenAI

        Evaluation logic:
        - Criterion B (anti-hallucination): Checked ONCE per task - verifies agent's response is supported by screenshots
        - Criterion A (fact verification): Checked for EACH rubric/fact - verifies screenshots show evidence for each fact
        - Final reward = 1 if (Criterion B passes) AND (ALL Criterion A checks pass)

        Returns:
            tuple: (reward, evaluation_text, is_blocked)
                - reward: 1 if successful, 0 otherwise
                - evaluation_text: String or list of evaluation responses
                - is_blocked: True if website blocked the agent, False otherwise
        """
        # STEP 0: Judge which images should be submitted
        print("📋 Step 0: Judging which images contain necessary task information...")
        try:
            self.judge_submission_images(trajectory)
        except Exception as e:
            print(f"⚠️ Image judging crashed (content filter?), marking all images for submission: {e}")
            for step in trajectory:
                if step.get('observation') and hasattr(step['observation'], 'image_path'):
                    if 'reward' not in step or step['reward'] is None:
                        step['reward'] = Reward(reward=0, evaluation="", submit=True, submission_judgment="fallback")
                    else:
                        step['reward'].submit = True

        # STEP 1: Check for blocking ONCE for the entire trajectory (independent of rubrics)
        is_blocked = self.check_if_blocked(trajectory)

        # Get task info
        task = trajectory[-1]['observation'].task
        reference_answer = getattr(task, 'reference_answer', None)
        agent_response = trajectory[-1]['action'].action['arguments']['content']
        trajectory_summary = self.conversation_builder.summarize_trajectory(trajectory)

        # Prepare submitted screenshots
        submitted_screenshots = []
        for i in range(len(trajectory)):
            if trajectory[i].get('reward') and hasattr(trajectory[i]['reward'], 'submit') and trajectory[i]['reward'].submit:
                submitted_screenshots.append(trajectory[i]['observation'].image_path)

        # Cap at 48 to stay under OpenAI's 50-image limit (keep last 48 as they're most relevant)
        if len(submitted_screenshots) > 48:
            print(f"⚠️ Trajectory has {len(submitted_screenshots)} submitted screenshots, capping to last 48 for API limit")
            submitted_screenshots = submitted_screenshots[-48:]

        screenshots_messages = []
        for i in range(len(submitted_screenshots)):
            image_base64 = base64.b64encode(open(submitted_screenshots[i], "rb").read()).decode('utf-8')
            screenshots_messages.append({
                "type": "image_url",
                "image_url": { "url": f"data:image/png;base64,{image_base64}" }
            })

        responses = []

        # =====================================================================
        # STEP 2: Check Criterion B ONCE for the entire task (anti-hallucination)
        # Verifies that the agent's response is supported by the screenshots
        # =====================================================================
        print("📋 Step 2: Checking Criterion B (anti-hallucination) for agent response...")
        criterion_b_passed = False
        try:
            # Get task-specific client and model
            eval_client, eval_model = self._get_client_and_model(self.TASK_EVALUATION)

            # Build Criterion B prompt
            criterion_b_prompt = criterion_b_user_prompt.replace("[task_instruction]", task.task_name)
            criterion_b_prompt = criterion_b_prompt.replace("[response]", agent_response)

            messages = [
                {"role": "system", "content": criterion_b_system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": criterion_b_prompt},
                        *screenshots_messages
                    ]
                }
            ]

            # Call API
            completion = eval_client.chat.completions.create(
                model=eval_model,
                messages=messages,
                max_tokens=800,
                temperature=0.7,
                top_p=0.95,
                stream=False
                #reasoning_effort="minimal"  # add this only for reasoning models
            )
            response = completion.choices[0].message.content

            criterion_b_result, _ = self._extract_verification_response(response)
            criterion_b_passed = (criterion_b_result == 1)
            responses.append(f"[Criterion B - Anti-Hallucination] {response}")

            if not criterion_b_passed:
                print("❌ Criterion B failed: Agent response not verified by screenshots")

        except Exception as e:
            import traceback
            print(f"Error during Criterion B evaluation: {e}")
            traceback.print_exc()
            responses.append(f"[Criterion B - Anti-Hallucination] Error: {e}")
            criterion_b_passed = False

        # If Criterion B fails, we can skip Criterion A checks (optimization)
        if not criterion_b_passed:
            print("⏭️ Skipping Criterion A checks since Criterion B failed")
            return 0, responses, is_blocked

        # =====================================================================
        # STEP 3: Check Criterion A for EACH rubric/fact (fact verification)
        # Verifies that the screenshots contain evidence for each fact
        # =====================================================================
        print("📋 Step 3: Checking Criterion A (fact verification) for each rubric...")
        rubrics = trajectory[-1]['observation'].task.evaluator_reference
        criterion_a_results = []

        # Reuse the same eval client and model (already fetched for Criterion B)
        # eval_client, eval_model are the same since they use TASK_EVALUATION

        # Build base prompt for Criterion A
        criterion_a_base = criterion_a_user_prompt.replace("[task_instruction]", task.task_name)
        criterion_a_base = criterion_a_base.replace("[trajectory]", trajectory_summary)

        for i, rubric in enumerate(rubrics):
            try:
                # Handle both old format (string) and new format (dict with 'description')
                rubric_text = rubric['description'] if isinstance(rubric, dict) else rubric

                # For fact_group, use all rubrics as context
                fact_group = "\n".join([f"- {r['description'] if isinstance(r, dict) else r}" for r in rubrics])

                prompt_for_rubric = criterion_a_base.replace("[fact_group]", fact_group)
                prompt_for_rubric = prompt_for_rubric.replace("[fact_to_check]", rubric_text)

                messages = [
                    {"role": "system", "content": criterion_a_system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_for_rubric},
                            *screenshots_messages
                        ]
                    }
                ]

                # Call API
                completion = eval_client.chat.completions.create(
                    model=eval_model,
                    messages=messages,
                    max_tokens=800,
                    temperature=0.7,
                    top_p=0.95,
                    stream=False
                    #reasoning_effort="minimal"  # add this only for reasoning models
                )
                response = completion.choices[0].message.content

                verification_result, _ = self._extract_verification_response(response)
                criterion_a_results.append(verification_result)
                responses.append(f"[Criterion A - Fact {i+1}] {response}")

                if verification_result == 0:
                    print(f"❌ Criterion A failed for rubric {i+1}: {rubric_text[:50]}...")
                    break  # Early exit on first failure

            except Exception as e:
                import traceback
                print(f"Error during Criterion A verification for rubric {i+1}: {e}")
                traceback.print_exc()
                criterion_a_results.append(0)
                responses.append(f"[Criterion A - Fact {i+1}] Error: {e}")

        # =====================================================================
        # STEP 4: Additional evaluation based on reference answer if it exists
        # =====================================================================
        has_reference_answer = reference_answer and reference_answer.strip()
        if has_reference_answer and len(criterion_a_results) == len(rubrics) and all(r == 1 for r in criterion_a_results):
            print(f"📋 Step 4: Evaluating with reference answer: {reference_answer}")
            try:
                # Reuse eval_client and eval_model (same as Criterion A/B)
                reference_rubric = f"""The agent should arrive at an answer that matches or is similar to the reference answer.

Reference Answer: {reference_answer}

The agent's response should either:
1. Match the reference answer (for factual answers), OR
2. Be similar/equivalent to the reference answer (for example answers)"""

                fact_group = "\n".join([f"- {r['description'] if isinstance(r, dict) else r}" for r in rubrics])
                prompt_for_reference = criterion_a_base.replace("[fact_group]", fact_group)
                prompt_for_reference = prompt_for_reference.replace("[fact_to_check]", reference_rubric)

                messages = [
                    {"role": "system", "content": criterion_a_system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_for_reference},
                            *screenshots_messages
                        ]
                    }
                ]

                completion = eval_client.chat.completions.create(
                    model=eval_model,
                    messages=messages,
                    max_tokens=800,
                    temperature=0.7,
                    top_p=0.95,
                    stream=False
                    #reasoning_effort="minimal"  # add this only for reasoning models
                )
                response = completion.choices[0].message.content

                verification_result, _ = self._extract_verification_response(response)
                criterion_a_results.append(verification_result)
                responses.append(f"[Reference Answer Evaluation] {response}")

            except Exception as e:
                import traceback
                print(f"Error during reference answer evaluation: {e}")
                traceback.print_exc()
                criterion_a_results.append(0)
                responses.append(f"[Reference Answer Evaluation] Error: {e}")

        # =====================================================================
        # FINAL: Calculate reward
        # Success = Criterion B passes AND all Criterion A checks pass
        # =====================================================================
        expected_criterion_a_count = len(rubrics) + (1 if has_reference_answer else 0)
        all_criterion_a_passed = (len(criterion_a_results) == expected_criterion_a_count and
                                   all(result == 1 for result in criterion_a_results))

        if criterion_b_passed and all_criterion_a_passed:
            final_reward = 1
            print("✅ All criteria passed - reward = 1")
        else:
            final_reward = 0
            if not criterion_b_passed:
                print("❌ Final reward = 0 (Criterion B failed)")
            else:
                print(f"❌ Final reward = 0 (Criterion A: {sum(criterion_a_results)}/{expected_criterion_a_count} passed)")

        return final_reward, responses, is_blocked

    def check_if_blocked(self, trajectory: List[Dict]) -> bool:
        """Check if website blocked the agent by analyzing screenshots in trajectory

        This method is called when the agent does NOT answer (runs out of steps).
        It sends up to 20 randomly sampled screenshots to detect blocking.

        Returns:
            bool: True if website blocked the agent, False otherwise
        """
        import random

        if not trajectory or len(trajectory) == 0:
            return False

        try:
            # Get task-specific client and model
            blocking_client, blocking_model = self._get_client_and_model(self.TASK_BLOCKING_DETECTION)

            # Get task description
            task_name = trajectory[0]['observation'].task.task_name if trajectory[0].get('observation') else "Unknown task"

            # Collect indices of steps with screenshots
            screenshot_indices = []
            for i in range(len(trajectory)):
                if trajectory[i].get('observation') and hasattr(trajectory[i]['observation'], 'image_path'):
                    screenshot_indices.append(i)

            if not screenshot_indices:
                print("⚠️ No screenshots available for blocking detection")
                return False

            # Sample up to 20 screenshots if there are more than 20
            if len(screenshot_indices) > 20:
                sampled_indices = sorted(random.sample(screenshot_indices, 20))
                print(f"📊 Sampling 20 screenshots from {len(screenshot_indices)} total screenshots for blocking detection")
            else:
                sampled_indices = screenshot_indices

            # Prepare screenshots from sampled indices
            screenshots_messages = []
            for i in sampled_indices:
                image_path = trajectory[i]['observation'].image_path
                try:
                    image_base64 = base64.b64encode(open(image_path, "rb").read()).decode('utf-8')
                    screenshots_messages.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                    })
                except Exception as e:
                    print(f"⚠️ Failed to load screenshot {image_path}: {e}")
                    continue

            if not screenshots_messages:
                print("⚠️ No screenshots could be loaded for blocking detection")
                return False

            # Build prompt
            trajectory_summary = self.conversation_builder.summarize_trajectory(trajectory)
            prompt = blocking_detection_user_prompt.replace("[task]", task_name)
            prompt = prompt.replace("[trajectory]", trajectory_summary)

            messages = [
                {"role": "system", "content": blocking_detection_system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *screenshots_messages
                    ]
                }
            ]

            # Call API
            completion = blocking_client.chat.completions.create(
                model=blocking_model,
                messages=messages,
                max_tokens=500,
                temperature=0.7,
                top_p=0.95,
                stream=False
                #reasoning_effort="minimal"  # add this only for reasoning models
            )
            response = completion.choices[0].message.content

            # Parse response for blocking status
            is_blocked = self._extract_blocking_status(response)
            return is_blocked

        except Exception as e:
            import traceback
            print(f"⚠️ Error during blocking detection: {e}")
            traceback.print_exc()
            return False

    def _extract_blocking_status(self, response: str) -> bool:
        """Extract blocking status from blocking detection response

        Returns:
            bool: True if blocked, False otherwise
        """
        if "Blocked:" in response:
            blocked_part = response.split("Blocked:")[1].split("\n")[0] if "Blocked:" in response else ""
            return "YES" in blocked_part.upper()
        return False

    def judge_submission_images(self, trajectory: List[Dict]) -> None:
        """Batch-process trajectory images in parallel to determine which should be submitted

        Uses OpenAI to evaluate each image in parallel and sets submit=True/False based on
        whether the image contains necessary information for task completion.

        Args:
            trajectory: List of trajectory steps, modified in-place
        """
        import concurrent.futures

        if not trajectory or len(trajectory) == 0:
            return

        # Get task-specific client and model for keypoint detection
        keypoint_client, keypoint_model = self._get_client_and_model(self.TASK_KEYPOINT_DETECTION)

        # Get task information from first observation
        task = trajectory[0]['observation'].task
        task_name = task.task_name
        # Handle both old format (string) and new format (dict with 'description')
        key_points = "\n".join([f"- {rubric['description'] if isinstance(rubric, dict) else rubric}" for rubric in task.evaluator_reference])

        system_msg = """You are an expert evaluator determining whether an image contains relevant information for completing a task.

**Instructions**:
- Answer "YES" if the image shows ANY task-related content: actions taken, progress, search results, tool usage, error messages, or blocking screens.
- Answer "NO" only if completely irrelevant: generic homepage, unrelated webpage, or blank screens with no context.
- When in doubt, answer "YES".

**Response format**:
1. **Reasoning**: [One sentence explanation]
2. **Decision**: [YES or NO]"""

        prompt_template = """**Task**: {task}

**Key Points for Task Completion**:
{key_points}

The snapshot of the web page is shown in the image. Does this image contain relevant information for the task? (Answer YES unless it's completely irrelevant)"""

        # Collect steps with images
        steps_with_images = []
        for i, step in enumerate(trajectory):
            if step.get('observation') and hasattr(step['observation'], 'image_path'):
                steps_with_images.append((i, step))

        if not steps_with_images:
            return

        # Separate last step from others (last step is always submitted, no need to judge)
        last_image_index = steps_with_images[-1][0]
        steps_to_judge = steps_with_images[:-1]  # All except last

        print(f"🔍 Judging {len(steps_to_judge)} images for submission in parallel (skipping last step which is always included)...")

        # Define function to judge a single image with retry logic
        def judge_single_image(args):
            i, step = args
            image_path = step['observation'].image_path

            # Retry loop
            for attempt in range(self.max_retries + 1):
                try:
                    # Encode image
                    image_base64 = base64.b64encode(open(image_path, "rb").read()).decode('utf-8')

                    # Build prompt
                    text = prompt_template.format(task=task_name, key_points=key_points)

                    messages = [
                        {"role": "system", "content": system_msg},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{image_base64}", "detail": "high"}
                                }
                            ]
                        }
                    ]

                    # Call API using task-specific client and model
                    completion = keypoint_client.chat.completions.create(
                        model=keypoint_model,
                        messages=messages,
                        max_tokens=600,
                        temperature=0.7,
                        top_p=0.95,
                        stream=False
                        #reasoning_effort="minimal"  # add this only for reasoning models
                    )
                    response = completion.choices[0].message.content

                    # Parse decision
                    should_submit = self._extract_submission_decision(response)

                    return (i, should_submit, response)

                except Exception as e:
                    if attempt == self.max_retries:
                        # Final attempt failed, default to not submitting
                        print(f"⚠️ Error judging image at step {i+1} after {self.max_retries + 1} attempts: {e}")
                        return (i, False, None)
                    else:
                        # Retry on next iteration
                        import time
                        time.sleep(0.5)  # Brief pause before retry
                        continue

        # Execute judgments in parallel (only for non-last steps)
        results = []
        if steps_to_judge:
            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                results = list(executor.map(judge_single_image, steps_to_judge))

        # Update submit flags and judgments in rewards
        for i, should_submit, response in results:
            # Create or update reward object with submission info
            if 'reward' not in trajectory[i] or trajectory[i]['reward'] is None:
                trajectory[i]['reward'] = Reward(
                    reward=0,  # Default, will be set later if this is final step
                    evaluation="",  # Will be set later if this is final step
                    submit=should_submit,
                    submission_judgment=response
                )
            else:
                trajectory[i]['reward'].submit = should_submit
                trajectory[i]['reward'].submission_judgment = response

        # ALWAYS ensure the last screenshot is submitted (most important for evaluation)
        # Skip API call since we always include it anyway
        if 'reward' not in trajectory[last_image_index] or trajectory[last_image_index]['reward'] is None:
            trajectory[last_image_index]['reward'] = Reward(
                reward=0,
                evaluation="",
                submit=True,
                submission_judgment="Last step is always included for evaluation"
            )
        else:
            trajectory[last_image_index]['reward'].submit = True
            trajectory[last_image_index]['reward'].submission_judgment = "Last step is always included for evaluation"

        # Summary - count from actual trajectory submit flags (not results)
        num_submitted = sum(1 for step in trajectory if step.get('reward') and hasattr(step['reward'], 'submit') and step['reward'].submit)
        print(f"✅ Submission judgment complete: {num_submitted}/{len(steps_with_images)} images marked for submission")

    def _extract_submission_decision(self, response: str) -> bool:
        """Extract yes/no decision from submission judgment response

        Returns:
            bool: True if should submit, False otherwise
        """
        # Try to find Decision: followed by YES or NO (handles markdown and whitespace)
        # Matches patterns like: "Decision: YES", "**Decision**: YES", "Decision : NO", etc.
        decision_match = re.search(r'\*\*Decision\*\*\s*:\s*(YES|NO)', response, re.IGNORECASE)
        if decision_match:
            return decision_match.group(1).upper() == "YES"

        # Fallback: try without markdown
        decision_match = re.search(r'Decision\s*:\s*(YES|NO)', response, re.IGNORECASE)
        if decision_match:
            return decision_match.group(1).upper() == "YES"

        # Last resort: check for YES/NO after "Decision" keyword
        if "Decision" in response or "decision" in response:
            decision_line = response.lower().split("decision")[-1]
            # If the line contains YES before NO, or only YES, return True
            if "yes" in decision_line and "no" not in decision_line:
                return True
            if "yes" in decision_line and "no" in decision_line:
                # Return True if YES appears before NO
                return decision_line.index("yes") < decision_line.index("no")

        return False

    def check_single_screenshot_for_blocking(self, screenshot_path: str, task_name: str, step_number: int) -> bool:
        """Check if a single screenshot shows website blocking

        This is called after each step to detect blocking immediately.
        Sends only the current screenshot to the API.

        Args:
            screenshot_path: Path to the screenshot to check
            task_name: Name of the task being performed
            step_number: Current step number in the trajectory

        Returns:
            bool: True if website is blocking the agent, False otherwise
        """
        try:
            # Get task-specific client and model
            blocking_client, blocking_model = self._get_client_and_model(self.TASK_BLOCKING_DETECTION)

            # Encode screenshot
            try:
                image_base64 = base64.b64encode(open(screenshot_path, "rb").read()).decode('utf-8')
            except Exception as e:
                print(f"⚠️ Failed to load screenshot {screenshot_path} for blocking detection: {e}")
                return False

            # Simplified prompt for single-step blocking detection
            prompt = f"""Task: {task_name}

Current Step: {step_number}

Based on the screenshot, is the website EXPLICITLY BLOCKING the automated agent with anti-bot measures?

Look for:
- CAPTCHA challenges
- "Access Denied" or "Blocked" messages
- Cloudflare/bot detection pages
- Security verification pages

Response format:
1. Analysis: [brief description]
2. Blocked: [YES or NO]"""

            messages = [
                {"role": "system", "content": blocking_detection_system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                        }
                    ]
                }
            ]

            # Call API
            completion = blocking_client.chat.completions.create(
                model=blocking_model,
                messages=messages,
                max_tokens=300,
                temperature=0.3,  # Lower temperature for more consistent detection
                top_p=0.95,
                stream=False
                #reasoning_effort="minimal"  # add this only for reasoning models
            )
            response = completion.choices[0].message.content

            # Parse response for blocking status
            is_blocked = self._extract_blocking_status(response)

            if is_blocked:
                print(f"🚫 Step {step_number}: Blocking detected - {response.split('Analysis:')[1].split('Blocked:')[0].strip() if 'Analysis:' in response else 'blocking page detected'}")

            return is_blocked

        except Exception as e:
            print(f"⚠️ Error during step-level blocking detection: {e}")
            return False

    def _extract_verification_response(self, response: str) -> Tuple[int, None]:
        """Extract binary reward from verification response

        Returns:
            tuple: (reward, None) where reward is 0 or 1, second value is None for compatibility
        """
        reward = 0

        # Extract verdict
        if "Verdict:" in response:
            verdict_part = response.split("Verdict:")[1].split("\n")[0] if "Verdict:" in response else response
            reward = 0 if "NOT SUCCESS" in verdict_part else 1
        else:
            reward = 0 if "NOT SUCCESS" in response else 1

        # Return None for blocking status (handled separately now)
        return reward, None
