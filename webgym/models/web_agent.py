# webgym/models/web_agent.py
import os
import threading
import time
from typing import List, Dict, Any, Tuple, Optional
from webgym.data.components import Action, Response
from webgym.context import ContextManager
from webgym.data.response_decomposer import decompose_raw_response, get_action_string
from webgym.utils import BlocklistManager
import httpx


class WebAgent:
    """
    Web automation agent that uses context management for action generation.
    Uses vLLM for model inference.
    """

    def __init__(self, policy_config, context_config, model_config, save_path, vllm_server_url: str,
                 openai_config: Optional[Dict] = None, operation_timeout: int = 120,
                 vllm_timeout: int = 120, max_retries: int = 1, max_vllm_sessions: int = 32,
                 verbose: bool = True):
        """
        Initialize WebAgent.

        Args:
            policy_config: Policy configuration
            context_config: Context configuration
            model_config: Model configuration (e.g., {"model_type": "qwen3vl", "variant": "instruct"})
            save_path: Path to save models
            vllm_server_url: URL for vLLM server (required)
            openai_config: OpenAI configuration for evaluation (optional, creates internal Evaluator)
            operation_timeout: General operation timeout in seconds
            vllm_timeout: Timeout specifically for vLLM requests in seconds
            max_vllm_sessions: Maximum concurrent vLLM sessions (default: 32)
        """
        self.policy_config = policy_config
        self.context_config = context_config
        self.model_config = model_config
        self.save_path = save_path
        self.vllm_server_url = vllm_server_url
        self.openai_config = openai_config
        self.operation_timeout = operation_timeout
        self.vllm_timeout = vllm_timeout
        self.max_retries = max_retries
        self.max_vllm_sessions = max_vllm_sessions

        # Configure vLLM HTTP client with connection pooling
        self.vllm_client = httpx.Client(
            limits=httpx.Limits(
                max_connections=2400,
                max_keepalive_connections=1200,
                keepalive_expiry=300.0
            ),
            timeout=httpx.Timeout(vllm_timeout, connect=30.0),
            http2=False
        )

        # Simple semaphore to limit concurrent requests
        self.vllm_semaphore = threading.Semaphore(max_vllm_sessions)

        # Suppress verbose httpx logging
        import logging
        logging.getLogger("httpx").setLevel(logging.WARNING)

        if verbose:
            print(f"✅ vLLM httpx client configured:")
            print(f"   - Max connections: 2400")
            print(f"   - Keepalive connections: 1200")
            print(f"   - Keepalive expiry: 300s")
            print(f"   - Connect timeout: 30s")
            print(f"   - Request timeout: {vllm_timeout}s")
            print(f"   - Concurrent requests: {max_vllm_sessions}")

        # Pre-warm connection pool for large session counts
        if max_vllm_sessions > 10:
            self._prewarm_connection_pool(verbose=verbose)

        # These will be set by concrete implementations
        self.model = None
        self.train_model = None

        # Use checkpoint_path from policy_config if available
        checkpoint_name = getattr(policy_config, 'checkpoint_path', 'model.pt')
        if checkpoint_name and checkpoint_name.strip():
            checkpoint_full_path = os.path.join(save_path, checkpoint_name)
            if os.path.exists(checkpoint_full_path):
                self.updated_model_path = checkpoint_full_path
            else:
                self.updated_model_path = None
        else:
            self.updated_model_path = None

        # Initialize ContextManager with model_config
        self.context_manager = ContextManager(self.context_config, self.model_config, verbose)
        self.parser = self.context_manager.get_parser()

        # For backward compatibility - expose conversation builder
        self.prompt_constructor = self.context_manager.get_model_interface().conversation_builder

        # Initialize blocklist manager
        self.blocklist_manager = BlocklistManager(save_path)

        # Setup Evaluator if OpenAI config provided (for backward compatibility)
        self.evaluator = None
        if openai_config:
            from .evaluator import Evaluator
            self.evaluator = Evaluator(
                openai_config=openai_config,
                conversation_builder=self.prompt_constructor,
                max_retries=max_retries,
                verbose=verbose
            )

    def _get_combined_action_and_observation(self, trajectory: List[Dict], screenshot_path: str,
                                                page_metadata: Dict, ac_tree: str) -> Tuple[Action, str, Response]:
        """Get action, observation summary, and full response object"""

        # Now trajectory should always have at least one element with current observation
        task_name = trajectory[-1]['observation'].task.task_name
        task_website = trajectory[-1]['observation'].task.website

        # Build current observation dict for the conversation builder
        # Read submit from reward if available, otherwise default to False
        submit_flag = False
        if trajectory and trajectory[-1].get('reward') and hasattr(trajectory[-1]['reward'], 'submit'):
            submit_flag = trajectory[-1]['reward'].submit

        current_observation = {
            'image_path': screenshot_path,
            'ac_tree': ac_tree,
            'page_metadata': page_metadata,
            'task': task_name,
            'submit': submit_flag
        }

        # Build conversation using new ContextManager API
        messages = self.context_manager.build_conversation(
            task=task_name,
            trajectory=trajectory,
            current_observation=current_observation,
            website=task_website
        )

        # Ensure messages is a list
        if isinstance(messages, dict):
            messages = [messages]

        # Get response from vLLM
        raw_response = self._make_vllm_request_direct(messages)

        # Parse response using model-specific parser
        parsed_response = self.context_manager.parse_response(raw_response)

        # Defensive check: ensure parsed_response is a dict
        if not isinstance(parsed_response, dict):
            raise Exception(f"parse_response returned {type(parsed_response)} instead of dict. Raw response: {str(raw_response)[:200]}")

        # Extract action using model-specific extractor
        action_info = self.context_manager.extract_action(parsed_response)

        # Get action string for backward compatibility
        action_string = parsed_response.get('action', '') or str(action_info)

        # Create Action object
        action = Action(
            action=action_info,
            action_string=action_string
        )

        # Create Response object with model interface for proper parsing
        # Convert base64 images in prompt to file paths before storing
        from webgym.utils import convert_messages_to_path_format
        raw_prompt_with_paths = convert_messages_to_path_format(messages, trajectory, current_observation)

        response_obj = decompose_raw_response(
            raw_response,
            model_interface=self.context_manager.get_model_interface(),
            raw_prompt=raw_prompt_with_paths
        )

        return action, response_obj

    def get_action_and_observation_sync(self, trajectory: List[Dict], screenshot_path: str,
                                            page_metadata: Dict, step_data: Dict = None) -> Tuple[Action, str, Response]:
        """
        Get action, observation summary, and response object using combined approach
        """

        # Extract AC tree from step_data if provided
        ac_tree = step_data.get('ac_tree', '') if step_data else ''

        # Get action, observation summary, and response object
        action, response_obj = self._get_combined_action_and_observation(
            trajectory, screenshot_path, page_metadata, ac_tree
        )

        # Make sure response_obj is not None
        if response_obj is None:
            print("WARNING: Response object is None in get_action_and_observation_sync")

        return action, response_obj

    def parse_action_to_browser_command(self, action: Action) -> Dict:
        """Convert action to browser command using context manager"""
        return self.context_manager.parse_action_to_browser_command(action)

    # ================================================================
    # CONNECTION MANAGEMENT
    # ================================================================

    def __enter__(self):
        """No-op context manager entry for compatibility"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up client on exit"""
        if hasattr(self, 'vllm_client'):
            self.vllm_client.close()

    def _prewarm_connection_pool(self, num_connections: int = None, verbose: bool = True):
        """Pre-create HTTP connections to avoid cold start

        Args:
            num_connections: Number of connections to pre-warm (default: min(max_vllm_sessions, 100))
            verbose: Whether to print progress messages
        """
        import concurrent.futures

        num_connections = num_connections or min(self.max_vllm_sessions, 100)

        if verbose:
            print(f"🔥 Pre-warming {num_connections} vLLM connections...")

        def warmup_connection():
            """Simple health check to establish connection"""
            try:
                response = self.vllm_client.get(
                    f"{self.vllm_server_url}/health",
                    timeout=5.0
                )
                return response.status_code == 200
            except:
                return False

        # Run warmup in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_connections) as executor:
            results = list(executor.map(lambda _: warmup_connection(), range(num_connections)))

        successful = sum(results)
        if verbose:
            print(f"✅ Pre-warmed {successful}/{num_connections} connections")

    # ================================================================
    # VLLM REQUEST HANDLING
    # ================================================================

    def _make_vllm_request_direct(self, messages: List[Dict]):
        """Make direct vLLM request using httpx - let vLLM do the batching"""
        model_name = getattr(self.policy_config, 'base_model', 'unknown')

        payload = {
            "model": self.updated_model_path if self.updated_model_path else model_name,
            "messages": messages,
            "temperature": getattr(self.policy_config, 'temperature', 0.7),
            "top_p": getattr(self.policy_config, 'top_p', 1e-5),
            "top_k": getattr(self.policy_config, 'top_k', 20),
            "max_tokens": getattr(self.policy_config, 'max_new_tokens', 512),
            "logprobs": 1,
            "echo": False,
        }

        # Use semaphore to avoid overwhelming vLLM
        with self.vllm_semaphore:
            for attempt in range(self.max_retries + 1):
                try:
                    response = self.vllm_client.post(
                        f"{self.vllm_server_url}/v1/chat/completions",
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )

                    if response.status_code == 200:
                        data = response.json()
                        choice = data["choices"][0]
                        message = choice["message"]
                        content = message["content"]

                        # Check for reasoning_content (from vLLM reasoning parser)
                        if "reasoning_content" in message and message["reasoning_content"]:
                            reasoning = message["reasoning_content"]
                            content = f"<think>\n{reasoning}\n</think>\n\n{content}"

                        return content
                    else:
                        error_msg = f"vLLM server returned status {response.status_code}: {response.text[:200]}"
                        if attempt == self.max_retries:
                            raise Exception(error_msg)
                        else:
                            print(f"   🔄 vLLM request attempt {attempt + 1} failed: {error_msg}")
                            time.sleep(1 + attempt)

                except Exception as e:
                    if attempt == self.max_retries:
                        import traceback
                        print(f"vLLM request failed after {self.max_retries + 1} attempts: {e}")
                        traceback.print_exc()
                        raise Exception(f"vLLM server request failed: {e}")
                    else:
                        error_msg = str(e)
                        print(f"   🔄 vLLM request attempt {attempt + 1} failed: {error_msg[:100]}")
                        time.sleep(1 + attempt)

    def _build_http_messages(self, system_prompt: str, user_prompt: str,
                           screenshot_path: str, trajectory: list = None):
        """
        Build messages for HTTP requests with current image only.
        """
        from webgym.utils import encode_image_to_base64

        # Only add system message if system_prompt is not empty
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})

        user_content = [{"type": "text", "text": user_prompt}]

        # Add current screenshot only
        current_image_url = encode_image_to_base64(screenshot_path)
        user_content.append({
            "type": "image_url",
            "image_url": {"url": current_image_url}
        })

        messages.append({"role": "user", "content": user_content})
        return messages
