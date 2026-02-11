# Create this file: webgym/environment/pickleable_http_functions.py

"""
Standalone functions for HTTP operations that can be pickled for multiprocessing.
These functions replace the lambda functions that were causing pickle errors.
"""

import os
import time
from PIL import Image
from webgym.environment import client
from webgym.navigation_error_logger import log_navigation_error


def _truncate_error_msg(error_msg: str, max_length: int = 150) -> str:
    """Truncate error messages to keep logs clean and readable"""
    return error_msg[:max_length] + "..." if len(error_msg) > max_length else error_msg


class MockResult:
    """Mock result object for operations that don't need HTTP requests (e.g., local sleep)"""
    def __init__(self, status_code=200, text=""):
        self.status_code = status_code
        self.text = text


def allocate_instance(host, port, api_key):
    """Allocate browser instance from server"""
    master_client = client.MasterClient(host=host, port=port, api_key=api_key)
    return master_client.get_instance(45)  # 45 minutes


def reset_instance(host, port, api_key, instance):
    """Reset browser instance on server"""
    master_client = client.MasterClient(host=host, port=port, api_key=api_key)
    return master_client.reset_instance(instance)


def execute_command(host, port, api_key, instance, command, max_retries=2):
    """Execute command on browser instance with retry logic that ignores errors after max_retries attempts"""

    # OPTIMIZATION: Handle sleep commands locally without HTTP request
    if command and isinstance(command, dict) and 'sleep' in command:
        duration = command['sleep'].get('duration', 2.0)
        time.sleep(duration)

        # Return mock success result
        return MockResult(status_code=200, text=f"Local sleep completed ({duration}s)")

    # Check if this is a navigate action (visit_page command)
    is_navigate_action = command and isinstance(command, dict) and 'visit_page' in command
    navigate_url = command.get('visit_page', {}).get('url', '') if is_navigate_action else ''

    master_client = client.MasterClient(host=host, port=port, api_key=api_key)

    for attempt in range(max_retries + 1):
        try:
            result = master_client.execute(instance, command)
            if result.status_code == 200:
                return result
            else:
                error_msg = f"Command failed with status {result.status_code}: {result.text}"
                if attempt == max_retries:
                    # For navigate actions, print debug message and continue gracefully
                    if is_navigate_action:
                        print(f"   🚫 Navigation failed: URL '{navigate_url}' is not accessible or does not exist")
                        print(f"   🐞 DEBUG: {_truncate_error_msg(error_msg, 200)}")
                        # Return success so trajectory continues
                        return MockResult(status_code=200, text=f"Navigation failed: {navigate_url}")
                    else:
                        # After retries, silently ignore and continue
                        return result  # Return the failed result instead of raising exception
                else:
                    print(f"   🔄 Execute command attempt {attempt + 1} failed: {_truncate_error_msg(error_msg)}")
                    time.sleep(1 + attempt)  # Progressive backoff: 1s, 2s, 3s
        except Exception as e:
            error_msg = str(e)
            if attempt == max_retries:
                # For navigate actions, print debug message and continue gracefully
                if is_navigate_action:
                    print(f"   🚫 Navigation failed: URL '{navigate_url}' is not accessible or does not exist")
                    print(f"   🐞 DEBUG: {_truncate_error_msg(error_msg, 200)}")
                    # Return success so trajectory continues
                    return MockResult(status_code=200, text=f"Navigation failed: {navigate_url}")
                else:
                    # After retries, silently ignore and continue
                    return MockResult(status_code=200, text="Error ignored after retries")
            else:
                print(f"   🔄 Execute command attempt {attempt + 1} failed: {_truncate_error_msg(error_msg)}")
                time.sleep(1 + attempt)  # Progressive backoff: 1s, 2s, 3s


def navigate_with_retries_and_correction(host, port, api_key, instance, task_id, url, max_retries, correction_manager=None):
    """Navigate to URL with retry logic (correction_manager parameter ignored for compatibility)"""
    return navigate_with_retries(host, port, api_key, instance, task_id, url, max_retries)


def navigate_with_retries(host, port, api_key, instance, task_id, url, max_retries=2):
    """Navigate to URL with retry logic and exponential backoff

    Returns success even if navigation fails, allowing trajectory to continue.
    Agent can then use navigate action to try alternative websites.
    """
    master_client = client.MasterClient(host=host, port=port, api_key=api_key)

    # Simple retry logic with exponential backoff
    for attempt in range(max_retries + 1):
        try:
            result = master_client.execute(instance, {"visit_page": {"url": url}})
            if result.status_code == 200:
                return result
            else:
                error_text = _truncate_error_msg(result.text, 80)
                raise Exception(f"Navigation failed with status {result.status_code}: {error_text}")
        except Exception as e:
            if attempt == max_retries:
                # Final failure - log the navigation error
                log_navigation_error(url, task_id)
                error_msg = _truncate_error_msg(str(e), 200)
                print(f"   🚫 {task_id}: Initial task URL '{url}' failed to load after {max_retries} retries")
                print(f"   🐞 DEBUG: {error_msg}")
                print(f"   💡 {task_id}: Trajectory will continue - agent can navigate to alternative websites")
                # Return success so trajectory doesn't crash - agent can use navigate action
                return MockResult(status_code=200, text=f"Initial navigation failed: {url}")
            else:
                # Retry with backoff
                error_msg = _truncate_error_msg(str(e), 120)
                print(f"   🔄 {task_id}: Navigation attempt {attempt + 1} failed: {error_msg}")
                time.sleep(2 ** attempt)  # Exponential backoff


def capture_screenshot(host, port, api_key, instance, output_dir, step, interaction_mode, max_retries=2, prev_screenshot_path=None, prev_action=None):
    """Capture screenshot with longer waits at scale - graceful degradation for resource exhaustion

    Args:
        prev_screenshot_path: Path to previous step's screenshot (for context in error messages)
        prev_action: Previous step's action dict (for context in error messages)

    Returns:
        tuple: (screenshot_path, is_fallback) where is_fallback indicates if we returned the previous screenshot
               due to blank screenshot after navigation failure
    """
    from webgym.misc import is_white_image
    import shutil

    master_client = client.MasterClient(host=host, port=port, api_key=api_key)

    os.makedirs(output_dir, exist_ok=True)
    screenshot_path = os.path.join(output_dir, f"step_{step:03d}.png")

    # Use max_retries parameter from config for white screenshot checks
    max_white_retries = max_retries
    # Wait times for white screenshot check: 5s, 5s, 5s, ... (one per retry)
    wait_times = [5] * max_retries if max_retries > 0 else [5]

    # Check if previous action was a navigate action (agent-proposed navigation)
    is_prev_action_navigate = False
    navigate_url = None
    if prev_action and isinstance(prev_action, dict):
        action_key = prev_action.get('key', '')
        if action_key == 'navigate':
            is_prev_action_navigate = True
            navigate_url = prev_action.get('arguments', {}).get('url', '')

    # Build context string for error messages
    context_str = ""
    if prev_screenshot_path:
        context_str += f" | Previous screenshot: {prev_screenshot_path}"
    if prev_action:
        # Print full action dict as string (no truncation)
        if isinstance(prev_action, dict):
            context_str += f" | Previous action: {prev_action}"
        else:
            context_str += f" | Previous action: {prev_action}"

    for attempt in range(max_white_retries):
        try:
            # Use streaming to avoid buffer exhaustion
            screenshot = master_client.screenshot(instance, interaction_mode, stream=True)
            if screenshot:
                # Check if screenshot is entirely white before saving
                if is_white_image(screenshot):
                    # Always print context when blank screenshot is detected
                    print(f"   ⚪ Step {step}: Screenshot blank (attempt {attempt + 1}){context_str}")
                    if attempt < max_white_retries - 1:
                        wait_time = wait_times[attempt]
                        print(f"      Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # After retries exhausted - check if this was due to agent's navigate action
                        if is_prev_action_navigate and prev_screenshot_path and os.path.exists(prev_screenshot_path):
                            # Gracefully handle: copy previous screenshot and return with fallback flag
                            print(f"   🚫 Step {step}: Navigation to '{navigate_url}' resulted in blank page - using previous screenshot")
                            print(f"   💡 Agent will be notified that the navigation failed")
                            shutil.copy(prev_screenshot_path, screenshot_path)
                            return (screenshot_path, True)  # Return with fallback flag
                        else:
                            # Not a navigate action or no previous screenshot - raise error as before
                            raise Exception(f"Step {step}: Screenshot blank after {max_white_retries} attempts{context_str}")

                # Screenshot is not white, save and return it
                screenshot.save(screenshot_path)
                # Verify file was created and has reasonable size
                if os.path.exists(screenshot_path) and os.path.getsize(screenshot_path) > 1000:
                    return (screenshot_path, False)  # Normal screenshot, no fallback
                else:
                    error_msg = f"Screenshot file invalid: {screenshot_path}"
                    raise Exception(f"Screenshot file invalid after save: {_truncate_error_msg(error_msg)}")
            else:
                error_msg = "Failed to capture screenshot from server"
                if attempt < max_white_retries - 1:
                    wait_time = wait_times[attempt]
                    print(f"   🔄 Screenshot attempt {attempt + 1} failed: {error_msg} - waiting {wait_time}s (extended wait for high-scale)")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Screenshot failed after {max_white_retries} attempts: {error_msg}")
        except Exception as e:
            error_msg = str(e)
            # Check if this is a white image error - propagate immediately
            if "blank" in error_msg:
                raise

            # Check if this is a retryable error (screenshot errors, SSL errors, connection errors)
            is_retryable = any(keyword in error_msg for keyword in [
                "Screenshot file invalid",
                "Failed to capture screenshot",
                "cannot identify image",  # PIL UnidentifiedImageError - server sent corrupted/invalid image data
                "UnidentifiedImageError",  # PIL error when image format is unrecognized
                "SSLError",
                "SSLEOFError",
                "SSL: UNEXPECTED_EOF",
                "Max retries exceeded",
                "Connection",
                "HTTPSConnectionPool"
            ])

            # Retry on retryable errors with extended waits for high-scale systems
            if is_retryable:
                if attempt < max_white_retries - 1:
                    wait_time = wait_times[attempt]
                    print(f"   🔄 Screenshot attempt {attempt + 1} failed: {_truncate_error_msg(error_msg)} - waiting {wait_time}s")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Screenshot failed after {max_white_retries} attempts: {_truncate_error_msg(error_msg)}")
            else:
                # For other exceptions (non-retryable errors), raise immediately
                raise Exception(f"Screenshot error: {_truncate_error_msg(error_msg)}")

    # Should not reach here, but just in case
    raise Exception(f"Screenshot failed after {max_white_retries} attempts")


def get_ac_tree(host, port, api_key, instance, max_retries=2):
    """Get accessibility tree from browser with retry logic for SSL errors"""
    master_client = client.MasterClient(host=host, port=port, api_key=api_key)

    for attempt in range(max_retries + 1):
        try:
            result = master_client.execute(instance, {"get_screenshot_info": {}})

            if result.status_code != 200:
                return ""

            response_data = result.json()
            visible_element_ids = response_data.get('visible_rects', [])

            if not visible_element_ids:
                return ""

            id_mapping = response_data.get('id_mapping', {})
            interactive_regions = response_data.get('interactive_regions', {})

            mapped_ids = [id_mapping.get(str(id), str(id)) for id in visible_element_ids]
            text_content = [interactive_regions.get(str(id), {}).get('aria_name', '') for id in mapped_ids]
            tag_name = [interactive_regions.get(str(id), {}).get('tag_name', '') for id in mapped_ids]

            id_to_text = {id: (text, tag) for id, text, tag in zip(visible_element_ids, text_content, tag_name)}
            return "\n".join([f"id: {id}, text: {text}, tag: {tag}" for id, (text, tag) in id_to_text.items()])

        except Exception as e:
            if attempt < max_retries:
                wait_time = 1 + attempt
                print(f"   🔄 get_ac_tree attempt {attempt + 1} failed: {_truncate_error_msg(str(e))} - waiting {wait_time}s")
                time.sleep(wait_time)
            else:
                return ""


def get_metadata(host, port, api_key, instance, max_retries=2):
    """Get instance viewport metadata (width, height) with retry logic for SSL errors

    Note: This gets viewport dimensions, not page metadata (title/URL).
    Uses the /metadata HTTP endpoint (not the execute command).
    """
    master_client = client.MasterClient(host=host, port=port, api_key=api_key)

    for attempt in range(max_retries + 1):
        try:
            # Use the dedicated /metadata endpoint (not execute command)
            result = master_client.metadata(instance)
            return result
        except Exception as e:
            if attempt < max_retries:
                wait_time = 1 + attempt
                print(f"   🔄 get_metadata attempt {attempt + 1} failed: {_truncate_error_msg(str(e))} - waiting {wait_time}s")
                time.sleep(wait_time)
            else:
                return {}


def get_page_metadata(host, port, api_key, instance, max_retries=2):
    """Get page metadata with retry logic for SSL errors"""
    master_client = client.MasterClient(host=host, port=port, api_key=api_key)

    for attempt in range(max_retries + 1):
        try:
            result = master_client.execute(instance, {"get_page_metadata": {}})
            return result.json() if result.status_code == 200 else {}
        except Exception as e:
            error_msg = str(e)
            # Check if this is a retryable SSL/connection error
            is_retryable = any(keyword in error_msg for keyword in [
                "SSLError",
                "SSLEOFError",
                "SSL: UNEXPECTED_EOF",
                "Max retries exceeded",
                "Connection",
                "HTTPSConnectionPool"
            ])

            if is_retryable and attempt < max_retries:
                wait_time = 1 + attempt  # Progressive backoff: 1s, 2s
                print(f"   🔄 get_page_metadata attempt {attempt + 1} failed: {_truncate_error_msg(error_msg)} - waiting {wait_time}s")
                time.sleep(wait_time)
            else:
                # After retries or non-retryable error, return empty dict
                print(f"   ⚠️ get_page_metadata failed: {_truncate_error_msg(error_msg)}")
                return {}


def wait_for_content(host, port, api_key, instance):
    """Wait for page to have actual content before taking screenshot"""
    master_client = client.MasterClient(host=host, port=port, api_key=api_key)

    # Execute JS to check for content
    for attempt in range(3):
        result = master_client.execute(instance, {
            "get_page_metadata": {}
        })

        if result.status_code == 200:
            metadata = result.json()
            # Check if page has loaded (not blank)
            if metadata.get('title') and metadata['title'] not in ['Unknown', 'Unknown Page', '']:
                return True

        time.sleep(2)  # Wait between checks

    return False  # Failed to verify content
