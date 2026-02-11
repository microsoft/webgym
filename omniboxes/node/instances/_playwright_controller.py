import asyncio
import base64
import os
import random
import time
import functools
from typing import Any, Callable, Dict, Optional, Tuple, Union, TypeVar, Awaitable, cast
from pathlib import Path

from playwright._impl._errors import Error as PlaywrightError
from playwright._impl._errors import TimeoutError, TargetClosedError
from playwright.async_api import Download, Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from omniboxes.node.instances._types import (
    InteractiveRegion,
    VisualViewport,
    interactiveregion_from_dict,
    visualviewport_from_dict,
)

# Some of the Code for clicking coordinates and keypresses adapted from https://github.com/openai/openai-cua-sample-app/blob/main/computers/base_playwright.py
# Copyright 2025 OpenAI - MIT License
CUA_KEY_TO_PLAYWRIGHT_KEY = {
    "/": "Divide",
    "\\": "Backslash",
    "alt": "Alt",
    "arrowdown": "ArrowDown",
    "arrowleft": "ArrowLeft",
    "arrowright": "ArrowRight",
    "arrowup": "ArrowUp",
    "backspace": "Backspace",
    "capslock": "CapsLock",
    "cmd": "Meta",
    "ctrl": "Control",
    "delete": "Delete",
    "end": "End",
    "enter": "Enter",
    "esc": "Escape",
    "home": "Home",
    "insert": "Insert",
    "option": "Alt",
    "pagedown": "PageDown",
    "pageup": "PageUp",
    "shift": "Shift",
    "space": " ",
    "super": "Meta",
    "tab": "Tab",
    "win": "Meta",
}

F = TypeVar('F', bound=Callable[..., Awaitable[Any]])

def handle_target_closed(max_retries: int = 2, timeout_secs: int = 30):
    """
    Decorator to handle TargetClosedError by attempting to recover the page.
    
    Args:
        max_retries: Maximum number of retry attempts
        timeout_secs: Timeout for page operations during recovery
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract the page object - assume it's the first argument after self
            page = None
            if len(args) >= 2 and hasattr(args[1], 'url'):  # Check if second arg looks like a Page
                page = args[1]
            
            retries = 0
            last_error = None
            
            while retries <= max_retries:
                try:
                    return await func(*args, **kwargs)
                except TargetClosedError as e:
                    last_error = e
                    retries += 1
                    
                    if retries > max_retries:
                        raise e
                    
                    if page is None:
                        # Can't recover without page reference
                        raise e
                    
                    print(f"TargetClosedError in {func.__name__}, attempting recovery (retry {retries}/{max_retries})")
                    
                    try:
                        # Attempt to recover the page
                        await _recover_page(page, timeout_secs)
                        # Small delay before retry
                        await asyncio.sleep(0.5)
                    except Exception as recovery_error:
                        print(f"Page recovery failed: {recovery_error}")
                        # If recovery fails, raise the original error
                        raise e from recovery_error
            
            # This shouldn't be reached, but just in case
            raise last_error
        
        return wrapper
    return decorator

async def _recover_page(page: Page, timeout_secs: int = 30) -> None:
    """
    Attempt to recover a closed page by reloading it.
    
    Args:
        page: The Playwright page object to recover
        timeout_secs: Timeout for recovery operations
    """
    try:
        # First, try to check if the page is still responsive
        await page.evaluate("1", timeout=1000)
        # If we get here, the page is actually fine
        return
    except Exception:
        # Page is indeed problematic, attempt recovery
        pass
    
    try:
        # Stop any ongoing navigation
        await page.evaluate("window.stop()", timeout=2000)
    except Exception:
        # Ignore errors from window.stop()
        pass
    
    try:
        # Try to reload the page
        await page.reload(timeout=timeout_secs * 1000)
        await page.wait_for_load_state("load", timeout=timeout_secs * 1000)
        print("playwright_controller._recover_page(): Page recovery successful")
    except Exception as e:
        print(f"playwright_controller._recover_page(): Page reload failed: {e}")
        
        # Try alternative recovery: navigate to current URL
        try:
            current_url = page.url
            if current_url and current_url != "about:blank":
                await page.goto(current_url, timeout=timeout_secs * 1000)
                await page.wait_for_load_state("load", timeout=timeout_secs * 1000)
                print("playwright_controller._recover_page(): Page recovery via goto successful")
            else:
                raise Exception("playwright_controller._recover_page(): No valid URL to navigate to")
        except Exception as goto_error:
            raise Exception(f"playwright_controller._recover_page(): All recovery methods failed. Reload error: {e}, Goto error: {goto_error}")

class PlaywrightController:
    def __init__(
        self,
        animate_actions: bool = False,
        downloads_folder: Optional[str] = None,
        viewport_width: int = 1440,
        viewport_height: int = 900,
        _download_handler: Optional[Callable[[Download], None]] = None,
        to_resize_viewport: bool = True,
        single_tab_mode: bool = False,
        sleep_after_action: int = 10,
        timeout_load: int = 1,
        timeout_action: int = 10,
        timeout_download: int = 60,
    ) -> None:
        """
        A controller for Playwright to interact with web pages.
        """
        self.animate_actions = animate_actions
        self.downloads_folder = downloads_folder
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self._download_handler = _download_handler
        self.to_resize_viewport = to_resize_viewport
        self.single_tab_mode = single_tab_mode
        self._sleep_after_action = sleep_after_action
        self._timeout_load = timeout_load

        self._page_script: str = ""
        self.last_cursor_position: Tuple[float, float] = (0.0, 0.0)

        # Load page script
        script_path = Path(__file__).parent / "_page_script.js"
        with open(script_path, "rt") as fh:
            self._page_script = fh.read()

    async def sleep(self, page: Page, duration: Union[int, float]) -> None:
        await page.wait_for_timeout(duration * 1000)

    @handle_target_closed()
    async def get_interactive_rects(self, page: Page) -> Dict[str, InteractiveRegion]:
        await self._ensure_page_ready(page)
        # Read the regions from the DOM
        try:
            await page.evaluate(self._page_script)
        except Exception:
            pass
        result = cast(Dict[str, Dict[str, Any]], await page.evaluate("MultimodalWebSurfer.getInteractiveRects();"))

        # Convert the results into appropriate types
        assert isinstance(result, dict)
        typed_results: Dict[str, InteractiveRegion] = {}
        for k in result:
            assert isinstance(k, str)
            typed_results[k] = interactiveregion_from_dict(result[k])

        return typed_results

    @handle_target_closed()
    async def get_visual_viewport(self, page: Page) -> VisualViewport:
        await self._ensure_page_ready(page)
        try:
            await page.evaluate(self._page_script)
        except Exception:
            pass
        return visualviewport_from_dict(await page.evaluate("MultimodalWebSurfer.getVisualViewport();"))

    @handle_target_closed()
    async def get_focused_rect_id(self, page: Page) -> str:
        await self._ensure_page_ready(page)
        try:
            await page.evaluate(self._page_script)
        except Exception:
            pass
        result = await page.evaluate("MultimodalWebSurfer.getFocusedElementId();")
        return str(result)

    @handle_target_closed()
    async def get_page_metadata(self, page: Page) -> Dict[str, Any]:
        assert page is not None
        
        # Initialize result with guaranteed fields
        result = {
            "title": "Unknown Page",
            "url": "about:blank"
        }
        
        try:
            # Get basic page information - these should always work
            try:
                title = await page.title()
                if title and title.strip():
                    result["title"] = title.strip()
            except Exception:
                pass
                
            try:
                url = page.url
                if url and url.strip():
                    result["url"] = url.strip()
            except Exception:
                pass
            
            # Try to get additional structured metadata (optional)
            attempts = 3
            while attempts > 0:
                try:
                    await self._ensure_page_ready(page)
                    await page.evaluate(self._page_script)
                    
                    # Get structured metadata from the page script
                    structured_data = await page.evaluate("MultimodalWebSurfer.getPageMetadata();")
                    if isinstance(structured_data, dict):
                        # Merge structured data with basic metadata, keeping title and url as primary
                        for key, value in structured_data.items():
                            if key not in result:  # Don't override title and url
                                result[key] = value
                    break
                except Exception as e:
                    print(f"Error getting structured metadata: {str(e)}, attempting again...")
                    attempts -= 1
                    if attempts > 0:
                        time.sleep(0.5)
                    
        except Exception as e:
            print(f"Error in get_page_metadata: {str(e)}")
            # result already has default values, so we can continue
            
        return result

    @handle_target_closed()
    async def on_new_page(self, page: Page) -> None:
        assert page is not None
        if self._download_handler:
            page.on("download", self._download_handler) # type: ignore
        if self.to_resize_viewport and self.viewport_width and self.viewport_height:
            await page.set_viewport_size({"width": self.viewport_width, "height": self.viewport_height})
        await self.sleep(page, 0.2)
        script_path = Path(__file__).parent / "_page_script.js"
        await page.add_init_script(path=str(script_path))
        try:
            await page.wait_for_load_state(timeout=30000)
        except PlaywrightTimeoutError:
            print("WARNING: Page load timeout, page might not be loaded")
            # stop page loading
            await page.evaluate("window.stop()")

    @handle_target_closed()
    async def _ensure_page_ready(self, page: Page) -> None:
        assert page is not None
        await self.on_new_page(page)

    @handle_target_closed()
    async def get_screenshot(self, page: Page, path: str | None = None) -> bytes:
        """
        Capture a screenshot of the current page.

        Args:
            page (Page): The Playwright page object.
            path (str, optional): The file path to save the screenshot. If None, the screenshot will be returned as bytes. Default: None
        """
        await self._ensure_page_ready(page)
        try:
            screenshot = await page.screenshot(path=path, timeout=15000)
            return screenshot
        except Exception:
            await page.evaluate("window.stop()")
            # try again
            screenshot = await page.screenshot(path=path, timeout=15000)
            return screenshot

    @handle_target_closed()
    async def back(self, page: Page) -> None:
        await self._ensure_page_ready(page)
        await page.go_back()

    @handle_target_closed()
    async def visit_page(self, page: Page, url: str) -> Tuple[bool, bool]:
        await self._ensure_page_ready(page)
        reset_prior_metadata_hash = False
        reset_last_download = False
        try:
            # Regular webpage
            await page.goto(url)
            await page.wait_for_load_state()
            reset_prior_metadata_hash = True
        except Exception as e_outer:
            # Downloaded file
            if self.downloads_folder and "net::ERR_ABORTED" in str(e_outer):
                async with page.expect_download() as download_info:
                    try:
                        await page.goto(url)
                    except Exception as e_inner:
                        if "net::ERR_ABORTED" in str(e_inner):
                            pass
                        else:
                            raise e_inner
                    download = await download_info.value
                    fname = os.path.join(self.downloads_folder, download.suggested_filename)
                    await download.save_as(fname)
                    message = f"<body style=\"margin: 20px;\"><h1>Successfully downloaded '{download.suggested_filename}' to local path:<br><br>{fname}</h1></body>"
                    await page.goto(
                        "data:text/html;base64," + base64.b64encode(message.encode("utf-8")).decode("utf-8")
                    )
                    reset_last_download = True
            else:
                raise e_outer
        return reset_prior_metadata_hash, reset_last_download

    @handle_target_closed()
    async def page_down(self, page: Page, amount: int = 400, full_page: bool = False) -> None:
        await self._ensure_page_ready(page)
        # Move mouse to top-left to avoid scrollable elements
        await page.mouse.move(10, 10)
        if full_page:
            await page.mouse.wheel(0, self.viewport_height - 50)
        else:
            await page.mouse.wheel(0, amount)

    @handle_target_closed()
    async def page_up(self, page: Page, amount: int = 400, full_page: bool = False) -> None:
        await self._ensure_page_ready(page)
        # Move mouse to top-left to avoid scrollable elements
        await page.mouse.move(10, 10)
        if full_page:
            await page.mouse.wheel(0, -self.viewport_height + 50)
        else:
            await page.mouse.wheel(0, -amount)

    async def gradual_cursor_animation(
        self, page: Page, start_x: float, start_y: float, end_x: float, end_y: float
    ) -> None:
        # animation helper
        steps = 20
        for step in range(steps):
            x = start_x + (end_x - start_x) * (step / steps)
            y = start_y + (end_y - start_y) * (step / steps)
            await page.evaluate(f"""
                (function() {{
                    let cursor = document.getElementById('red-cursor');
                    if (cursor) {{
                        cursor.style.left = '{x}px';
                        cursor.style.top = '{y}px';
                    }}
                }})();
            """)
            await asyncio.sleep(0.05)

        self.last_cursor_position = (end_x, end_y)

    async def add_cursor_box(self, page: Page, identifier: str) -> None:
        # animation helper
        await page.evaluate(f"""
            (function() {{
                let elm = document.querySelector("[__elementId='{identifier}']");
                if (elm) {{
                    elm.style.transition = 'border 0.3s ease-in-out';
                    elm.style.border = '2px solid red';
                }}
            }})();
        """)
        await asyncio.sleep(0.3)

        # Create a red cursor
        await page.evaluate("""
            (function() {
                let cursor = document.createElement('div');
                cursor.id = 'red-cursor';
                cursor.style.width = '10px';
                cursor.style.height = '10px';
                cursor.style.backgroundColor = 'red';
                cursor.style.position = 'absolute';
                cursor.style.borderRadius = '50%';
                cursor.style.zIndex = '10000';
                document.body.appendChild(cursor);
            })();
        """)

    async def remove_cursor_box(self, page: Page, identifier: str) -> None:
        # Remove the highlight and cursor
        await page.evaluate(f"""
            (function() {{
                let elm = document.querySelector("[__elementId='{identifier}']");
                if (elm) {{
                    elm.style.border = '';
                }}
                let cursor = document.getElementById('red-cursor');
                if (cursor) {{
                    cursor.remove();
                }}
            }})();
        """)

    @handle_target_closed()
    async def click_coords(self, page: Page, x: float, y: float) -> Page | None:
        new_page: Page | None = None
        await self._ensure_page_ready(page)

        # In single tab mode, remove target attributes to avoid opening new tabs
        if self.single_tab_mode:
            await page.evaluate(f"""
                (x, y) => {{
                    const element = document.elementFromPoint({x}, {y});
                    if (element) {{
                        // Remove target attribute from clicked element and all ancestors
                        let el = element;
                        while (el) {{
                            if (el.removeAttribute) {{
                                el.removeAttribute('target');
                            }}
                            el = el.parentElement;
                        }}
                    }}
                    // Remove target from all _blank links/forms
                    document.querySelectorAll('a[target=_blank], form[target=_blank]')
                        .forEach(e => e.removeAttribute('target'));
                }}
            """)

        if self.animate_actions:
            # Move cursor to the box slowly
            start_x, start_y = self.last_cursor_position
            await self.gradual_cursor_animation(page, start_x, start_y, x, y)
            await asyncio.sleep(0.1)

            if self.single_tab_mode:
                await page.mouse.click(x, y, delay=10)
            else:
                try:
                    # Give it a chance to open a new page
                    async with page.expect_event("popup", timeout=1000) as page_info:  # type: ignore
                        await page.mouse.click(x, y, delay=10)
                        new_page = await page_info.value  # type: ignore
                        assert isinstance(new_page, Page)
                        await self.on_new_page(new_page)
                except TimeoutError:
                    pass
        else:
            if self.single_tab_mode:
                await page.mouse.click(x, y, delay=10)
            else:
                try:
                    # Give it a chance to open a new page
                    async with page.expect_event("popup", timeout=1000) as page_info:  # type: ignore
                        await page.mouse.click(x, y, delay=10)
                        new_page = await page_info.value  # type: ignore
                        assert isinstance(new_page, Page)
                        await self.on_new_page(new_page)
                except TimeoutError:
                    pass
        return new_page

    @handle_target_closed()
    async def click_id(self, page: Page, identifier: str) -> Page | None:
        """
        Returns new page if a new page is opened, otherwise None.
        """
        new_page: Page | None = None
        await self._ensure_page_ready(page)
        selector = f"[__elementId='{identifier}']"
        try:
            # Wait for the element to be visible and scroll it into view
            await page.wait_for_selector(
                selector, state="visible", timeout=self._timeout_load * 1000
            )
            target = page.locator(selector)
            await target.scroll_into_view_if_needed()
        except TimeoutError:
            raise ValueError(
                f"Element with identifier {identifier} not found or not visible"
            )

        # Retrieve bounding box to determine the center for clicking
        box = await target.bounding_box()
        if not box:
            raise ValueError(
                f"Element with identifier {identifier} is not visible on the page."
            )
        center_x = box["x"] + box["width"] / 2
        center_y = box["y"] + box["height"] / 2

        # In single tab mode, override target attributes to avoid opening a new tab
        if self.single_tab_mode:
            await target.evaluate("""
                el => {
                    // Remove target attribute from clicked element and all _blank links/forms
                    el.removeAttribute('target');
                    document.querySelectorAll('a[target=_blank], form[target=_blank]')
                        .forEach(e => e.removeAttribute('target'));
                }
            """)

        download = None
        download_future: asyncio.Task[Download] | None = None

        # Start listening for a download event if downloads are enabled
        if self.downloads_folder:
            try:
                download_future = asyncio.create_task(
                    page.wait_for_event(  # type: ignore
                        "download", timeout=500
                    )
                )
            except Exception as e:
                print(f"Failed to set up download listener: {e}")
                download_future = None

        async def perform_click() -> Optional[Page]:
            nonlocal download
            try:
                if self.single_tab_mode:
                    await page.mouse.move(center_x, center_y, steps=1)
                    await page.mouse.click(center_x, center_y)
                    return None
                else:
                    # Create a task to wait for a new page event
                    context = page.context
                    new_page_promise: asyncio.Task[Page] = asyncio.create_task(
                        context.wait_for_event(  # type: ignore
                            "page", timeout=self._timeout_load * 1000
                        )
                    )

                    # Perform the click
                    await page.mouse.move(center_x, center_y, steps=1)
                    await page.mouse.click(center_x, center_y, delay=10)

                    try:
                        # Wait for the new page to open
                        new_page = await new_page_promise
                        await self.on_new_page(new_page)
                        return new_page
                    except TimeoutError:
                        # No new page opened within timeout
                        return None
            except Exception as e:
                raise e

        # Optionally animate the click
        if self.animate_actions:
            await self.add_cursor_box(page, identifier)
            start_x, start_y = self.last_cursor_position
            await self.gradual_cursor_animation(
                page, start_x, start_y, center_x, center_y
            )

        new_page = await perform_click()

        # Handle any download that occurred
        if download_future:
            try:
                if not download:
                    # Use asyncio.wait_for with a reasonable timeout
                    try:
                        download = await asyncio.wait_for(
                            download_future, timeout=self._timeout_load * 1000
                        )
                    except asyncio.TimeoutError:
                        # No download occurred within the timeout period
                        pass

                if download:
                    print(
                        f"Downloading {download.suggested_filename} to {self.downloads_folder}"
                    )
                    assert self.downloads_folder is not None
                    fname = os.path.join(
                        self.downloads_folder, download.suggested_filename
                    )
                    await download.save_as(fname)
            except Exception as e:
                pass
            finally:
                if not download_future.done():
                    download_future.cancel()

        if self.animate_actions:
            await self.remove_cursor_box(page, identifier)

        if new_page:
            await new_page.wait_for_load_state()
            if self._sleep_after_action > 0:
                await new_page.wait_for_timeout(self._sleep_after_action * 1000)
        else:
            await page.wait_for_load_state()
            if self._sleep_after_action > 0:
                await page.wait_for_timeout(self._sleep_after_action * 1000)

        return new_page

    @handle_target_closed()
    async def select_option(
        self, page: Page, identifier: str
    ) -> Optional[Page]:
        """
        Select an option element with the given identifier.
        """
        await self._ensure_page_ready(page)
        new_page: Optional[Page] = None
        try:
            # Wait for element to be present
            await page.wait_for_selector(
                f"[__elementId='{identifier}']", state="attached"
            )

            try:
                # First try normal click if element is visible
                target = page.locator(f"[__elementId='{identifier}']").first
                # Get the bounding box to check element size
                box = await target.bounding_box()

                if box and box["width"] > 0 and box["height"] > 0:
                    # Element has visible size - use normal click
                    return await self.click_id(page, identifier)

            except PlaywrightError as e:
                if "strict mode violation" in str(e):
                    # If multiple elements found, try clicking the first visible one
                    elements = await page.locator(
                        f"[__elementId='{identifier}']"
                    ).all()
                    for element in elements:
                        try:
                            if await element.is_visible():
                                await element.click()
                                return new_page
                        except PlaywrightError:
                            continue

            # If click didn't work, try programmatic selection
            # First check if it's a standard <option> element
            option_element = await page.evaluate(
            """
                (identifier) => {
                    const elements = document.querySelectorAll(`[__elementId='${identifier}']`);
                    for (const el of elements) {
                        if (el.tagName.toLowerCase() === 'option') {
                            return true;
                        }
                    }
                    return false;
                }
                """,
                identifier,
            )

            if option_element:
                # Handle standard <select> dropdown
                await page.evaluate(
                    """
                    (identifier) => {
                        const option = Array.from(document.querySelectorAll(`[__elementId='${identifier}']`))
                            .find(el => el.tagName.toLowerCase() === 'option');
                        if (!option) throw new Error('Option not found');
                        const select = option.closest('select');
                        if (select) {
                            option.selected = true;
                            select.dispatchEvent(new Event('change', { bubbles: true }));
                            select.blur();
                        }
                    }
                    """,
                    identifier,
                )
            else:
                # Handle custom dropdown/combobox options
                await page.evaluate(
                    """
                    (identifier) => {
                        const element = document.querySelector(`[__elementId='${identifier}']`);
                        if (!element) throw new Error('Element not found');

                        // Dispatch multiple events to ensure the selection is registered
                        const events = ['mousedown', 'mouseup', 'click', 'change'];
                        events.forEach(eventType => {
                            element.dispatchEvent(new Event(eventType, { bubbles: true }));
                        });

                        // If element has aria-selected, set it
                        if (element.hasAttribute('aria-selected')) {
                            element.setAttribute('aria-selected', 'true');
                        }

                        // If element has a data-value, try to set it on the parent
                        const value = element.getAttribute('data-value');
                        if (value) {
                            const parent = element.closest('[role="listbox"], [role="combobox"]');
                            if (parent) {
                                parent.setAttribute('data-value', value);
                            }
                        }
                    }
                    """,
                    identifier,
                )

            # Optional sleep/pause after the action
            if self._sleep_after_action > 0:
                await page.wait_for_timeout(self._sleep_after_action * 1000)

        except PlaywrightTimeoutError:
            raise ValueError(
                f"No option found with identifier '{identifier}' within "
                f"{self._timeout_load} seconds."
            ) from None
        return new_page

    @handle_target_closed()
    async def hover_id(self, page: Page, identifier: str) -> None:
        """
        Hovers the mouse over the target with the given id.
        """
        await self._ensure_page_ready(page)
        target = page.locator(f"[__elementId='{identifier}']")

        # See if it exists
        try:
            await target.wait_for(timeout=5000)
        except TimeoutError:
            raise RuntimeError(f"Tool use response is invalid: no such element to hover: {identifier}")

        # Hover over it
        await target.scroll_into_view_if_needed()
        await asyncio.sleep(0.3)

        box = cast(Dict[str, Union[int, float]], await target.bounding_box())

        if self.animate_actions:
            await self.add_cursor_box(page, identifier)
            # Move cursor to the box slowly
            start_x, start_y = self.last_cursor_position
            end_x, end_y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
            await self.gradual_cursor_animation(page, start_x, start_y, end_x, end_y)
            await asyncio.sleep(0.1)
            await page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)

            await self.remove_cursor_box(page, identifier)
        else:
            await page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)

    @handle_target_closed()
    async def hover_coords(self, page: Page, x: float, y: float) -> None:
        """
        Hovers the mouse at the specified coordinates without clicking.

        Args:
            page: The page to interact with
            x: X coordinate
            y: Y coordinate
        """
        await self._ensure_page_ready(page)

        if self.animate_actions:
            # Move cursor to the coordinates slowly
            start_x, start_y = self.last_cursor_position
            await self.gradual_cursor_animation(page, start_x, start_y, x, y)
            await asyncio.sleep(0.1)

        await page.mouse.move(x, y)
        self.last_cursor_position = (x, y)

    @handle_target_closed()
    async def hover_and_scroll_coords(self, page: Page, x: float, y: float, direction: str = "down") -> None:
        """
        Hovers the mouse at the specified coordinates and then scrolls the element at that position.

        This method implements best practices for coordinate-based scrolling:
        1. Finds the closest scrollable element (e.g., dropdown menu) at the coordinates
        2. Verifies the element is actually scrollable (checks overflow CSS and pointer-events)
        3. Scrolls by 80% of element height for smooth incremental scrolling with content overlap
        4. Handles nested scrollable containers by walking up the DOM tree

        Based on research from Playwright, Selenium, and Puppeteer best practices for
        dropdown menus and virtual scrolling containers.

        Args:
            page: The page to interact with
            x: X coordinate to hover over
            y: Y coordinate to hover over
            direction: Scroll direction ('up' or 'down')
        """
        await self._ensure_page_ready(page)

        # First, hover at the coordinates
        if self.animate_actions:
            start_x, start_y = self.last_cursor_position
            await self.gradual_cursor_animation(page, start_x, start_y, x, y)
            await asyncio.sleep(0.1)

        await page.mouse.move(x, y)
        self.last_cursor_position = (x, y)

        # Find the closest scrollable element and scroll it directly using JavaScript
        scroll_delta = -100 if direction.lower() == "up" else 100  # Small scroll amount

        scroll_result = await page.evaluate(f"""
            (function() {{
                // Identify element at coordinates
                let element = document.elementFromPoint({x}, {y});
                if (!element) {{
                    return {{ success: false }};
                }}

                // Walk up DOM tree to find scrollable parent
                let scrollable = element;
                while (scrollable && scrollable !== document.documentElement) {{
                    const style = getComputedStyle(scrollable);
                    const overflowY = style.overflowY;
                    const hasOverflow = scrollable.scrollHeight > scrollable.clientHeight;
                    const canScroll = overflowY === 'scroll' || overflowY === 'auto';
                    const canInteract = style.pointerEvents !== 'none';

                    if (hasOverflow && canScroll && canInteract) {{
                        break; // Found scrollable parent
                    }}

                    scrollable = scrollable.parentElement;
                }}

                // If we found a scrollable element (not the document), scroll it directly
                if (scrollable && scrollable !== document.documentElement) {{
                    // Calculate scroll amount: 80% of visible height for proper scrollable containers
                    const clientHeight = scrollable.clientHeight;
                    const scrollAmount = Math.floor(clientHeight * 0.8);
                    const oldScrollTop = scrollable.scrollTop;
                    scrollable.scrollTop += {scroll_delta} > 0 ? scrollAmount : -scrollAmount;
                    return {{ success: true, scrolled: scrollable.scrollTop !== oldScrollTop }};
                }} else {{
                    // No scrollable container found, scroll the element itself by a small amount
                    // This prevents the entire page from scrolling
                    if (element && element !== document.documentElement && element !== document.body) {{
                        const oldScrollTop = element.scrollTop;
                        element.scrollTop += {scroll_delta};
                        return {{ success: true, scrolled: element.scrollTop !== oldScrollTop }};
                    }}
                }}
                return {{ success: false }};
            }})();
        """)

        # If JavaScript scrolling didn't work, try dispatching wheel event to the element
        # This handles cases like custom dropdowns that respond to wheel events
        if not scroll_result.get('scrolled', False):
            delta_y = -100 if direction.lower() == "up" else 100
            # Dispatch wheel event at the specific coordinates
            await page.mouse.wheel(0, delta_y)

        # Brief wait for content stabilization (especially important for virtual scrolling)
        await asyncio.sleep(0.2)

    @handle_target_closed()
    async def fill_coords(
            self, page: Page, x: float, y: float, value: str, press_enter: bool = True, delete_existing_text: bool = False
        ) -> Page | None:
        await self._ensure_page_ready(page)
        new_page: Page | None = None

        if self.animate_actions:
            # Move cursor to the box slowly
            start_x, start_y = self.last_cursor_position
            await self.gradual_cursor_animation(page, start_x, start_y, x, y)
            await asyncio.sleep(0.1)

        await page.mouse.click(x, y)

        if delete_existing_text:
            await page.keyboard.press("ControlOrMeta+A")
            await page.keyboard.press("Backspace")

        # fill char by char to mimic human speed for short text and type fast for long text
        if len(value) < 100:
            delay_typing_speed = 50 + 100 * random.random()
        else:
            delay_typing_speed = 10

        if self.animate_actions:
            # Give it a chance to open a new page
            try:
                async with page.expect_event("popup", timeout=1000) as page_info:  # type: ignore
                    await page.keyboard.type(value, delay=delay_typing_speed)
                    if press_enter:
                        await page.keyboard.press("Enter")
                    new_page = await page_info.value  # type: ignore
                    assert isinstance(new_page, Page)
                    await self.on_new_page(new_page)
            except TimeoutError:
                pass
        else:
            try:
                # Give it a chance to open a new page
                async with page.expect_event("popup", timeout=1000) as page_info:  # type: ignore
                    try:
                        await page.keyboard.type(value)
                    except PlaywrightError:
                        await page.keyboard.type(value, delay=delay_typing_speed)
                    if press_enter:
                        await page.keyboard.press("Enter")
                    new_page = await page_info.value  # type: ignore
                    assert isinstance(new_page, Page)
                    await self.on_new_page(new_page)
            except TimeoutError:
                pass

        return new_page

    @handle_target_closed()
    async def fill_id(
        self, page: Page, identifier: str, value: str, press_enter: bool = True, delete_existing_text: bool = False
    ) -> Page | None:
        """
        Fill the element with the given identifier with the specified value.
        """
        await self._ensure_page_ready(page)
        await page.wait_for_selector(f"[__elementId='{identifier}']", state="visible")
        target = page.locator(f"[__elementId='{identifier}']")
        await target.scroll_into_view_if_needed()

        # See if it exists
        try:
            await target.wait_for(timeout=5000)
        except TimeoutError:
            raise RuntimeError(f"Tool use response is invalid: No such element to fill input_text into: {identifier}") from None

        # Fill it
        box = cast(Dict[str, Union[int, float]], await target.bounding_box())

        if self.single_tab_mode:
            # Remove target attributes to prevent new tabs
            await target.evaluate("""
                el => el.removeAttribute('target')
                // Remove 'target' on all <a> tags
                for (const a of document.querySelectorAll('a[target=_blank]')) {
                    a.removeAttribute('target');
                }
                // Remove 'target' on all <form> tags
                for (const frm of document.querySelectorAll('form[target=_blank]')) {
                    frm.removeAttribute('target');
                }
            """)

        page = await self.fill_coords(
            page,
            float(box["x"] + box["width"] / 2),
            float(box["y"] + box["height"] / 2),
            value,
            press_enter,
            delete_existing_text
        )
        return page

    @handle_target_closed()
    async def scroll_id(self, page: Page, identifier: str, direction: str) -> None:
        await self._ensure_page_ready(page)
        await page.evaluate(
            f"""
        (function() {{
            let elm = document.querySelector("[__elementId='{identifier}']");
            if (elm) {{
                if ("{direction}" == "up") {{
                    elm.scrollTop = Math.max(0, elm.scrollTop - elm.clientHeight);
                }} else if ("{direction}" == "down") {{
                    elm.scrollTop = Math.min(elm.scrollHeight - elm.clientHeight, elm.scrollTop + elm.clientHeight);
                }} else if ("{direction}" == "left") {{
                    elm.scrollLeft = Math.max(0, elm.scrollLeft - elm.clientWidth);
                }} else if ("{direction}" == "right") {{
                    elm.scrollLeft = Math.min(elm.scrollWidth - elm.clientWidth, elm.scrollLeft + elm.clientWidth);
                }}
            }}
        }})();
    """
        )

    async def keypress(self, page: Page, keys: list[str]) -> None:
        """
        Press specified keys in sequence.
        """
        await self._ensure_page_ready(page)
        mapped_keys = [CUA_KEY_TO_PLAYWRIGHT_KEY.get(key.lower(), key) for key in keys]
        try:
            if self.animate_actions:
                for key in mapped_keys:
                    await page.keyboard.down(key)
                    await asyncio.sleep(0.05)  # Small delay between key presses
                for key in reversed(mapped_keys):
                    await page.keyboard.up(key)
                    await asyncio.sleep(0.05)  # Small delay between key releases
            else:
                for key in mapped_keys:
                    await page.keyboard.down(key)
                for key in reversed(mapped_keys):
                    await page.keyboard.up(key)
        except Exception as e:
            raise RuntimeError(f"I tried to keypress(keys={keys}), but I got an error: {e}") from None

    @handle_target_closed()
    async def get_webpage_text(self, page: Page, n_lines: int = 100) -> str:
        """
        page: playwright page object
        n_lines: number of lines to return from the page innertext
        return: text in the first n_lines of the page
        """
        await self._ensure_page_ready(page)
        try:
            text_in_viewport = await page.evaluate("""() => {
                return document.body.innerText;
            }""")
            text_in_viewport = "\n".join(text_in_viewport.split("\n")[:n_lines])
            # remove empty lines
            text_in_viewport = "\n".join([line for line in text_in_viewport.split("\n") if line.strip()])
            assert isinstance(text_in_viewport, str)
            return text_in_viewport
        except Exception:
            return ""

    async def get_page_markdown(self, page: Page) -> str:
        # TODO: replace with mdconvert
        await self._ensure_page_ready(page)
        return await self.get_webpage_text(page, n_lines=1000)

    # type TAB then hit ENTER
    async def tab_and_enter(self, page: Page) -> None:
        await self._ensure_page_ready(page)
        await page.keyboard.press("Tab")
        await page.keyboard.press("Enter")