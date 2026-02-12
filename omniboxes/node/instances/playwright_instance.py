from typing import Any, Dict
from io import BytesIO

from playwright.async_api import async_playwright
from fastapi import HTTPException

from omniboxes.node.instances.base import InstanceBase, Status
from omniboxes.node.instances._set_of_marks import add_set_of_mark
from omniboxes.node.instances._playwright_controller import PlaywrightController


class PlaywrightInstance(InstanceBase):
    def __init__(self, instance_num=0, logger=None):
        super().__init__(instance_num=instance_num, logger=logger)
        self.page = None
        self.context = None
        self.browser = None
        self.p = None
        self._last_screenshot_info = None
        self.controller = PlaywrightController(
            viewport_width=1280,
            viewport_height=768,
            single_tab_mode=True,
            animate_actions=False
        )

    async def _create(self) -> None:
        self.p = await async_playwright().start()
        self.browser = await self.p.chromium.launch()
        self.context = await self.browser.new_context(
            viewport={'width': 1280, 'height': 768})
        self.page = await self.context.new_page()
        await self.controller.on_new_page(self.page)

    async def _delete(self) -> None:
        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.p:
            await self.p.stop()

    def _translate_displayed_id_to_original(self, displayed_id: str) -> str:
        """
        Translate a displayed ID (what user sees on screenshot) to the original element ID.
        """
        if self._last_screenshot_info and 'id_mapping' in self._last_screenshot_info:
            id_mapping = self._last_screenshot_info['id_mapping']
            return id_mapping.get(displayed_id, displayed_id)
        return displayed_id

    async def _screenshot(self, interaction_mode: str = "set_of_marks") -> BytesIO:
        """Take a screenshot with optional set-of-marks annotations based on interaction mode."""
        # Determine whether to apply annotations based on interaction_mode
        apply_annotations = interaction_mode == "set_of_marks"
        use_sequential_ids = True  # Default behavior
        
        # Get the base screenshot
        screenshot_bytes = await self.controller.get_screenshot(self.page)
        
        # If annotations are disabled (coordinates mode), return plain screenshot
        if not apply_annotations:
            return BytesIO(screenshot_bytes)
        
        # Get interactive regions using the controller
        interactive_regions = await self.controller.get_interactive_rects(self.page)
        
        if not interactive_regions:
            # If no interactive regions found, return plain screenshot
            return BytesIO(screenshot_bytes)
        
        try:
            # Apply set-of-marks annotation and capture the ID mapping
            annotated_image, visible_rects, rects_above, rects_below, id_mapping = add_set_of_mark(
                screenshot_bytes,
                interactive_regions,
                use_sequential_ids=use_sequential_ids
            )
            
            # Convert PIL Image back to bytes
            output_buffer = BytesIO()
            annotated_image.save(output_buffer, format='PNG')
            output_buffer.seek(0)
            
            # Store the mapping information for potential use
            self._last_screenshot_info = {
                'visible_rects': visible_rects,
                'rects_above': rects_above,
                'rects_below': rects_below,
                'interactive_regions': interactive_regions,
                'id_mapping': id_mapping  # Store the ID mapping for command translation
            }
            
            return output_buffer
            
        except Exception as e:
            # If annotation fails, return plain screenshot
            self.logger.error(f"Error applying set-of-marks: {str(e)}")
            return BytesIO(screenshot_bytes)

    async def _probe(self) -> bool:
        return self.status >= Status.STARTED

    async def _metadata(self) -> Dict[str, Any]:
        return {
            "width": 1280,
            "height": 768
        }

    async def _execute(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Get the first (and should be only) command from the dict
            if len(command_data) != 1:
                raise ValueError("Only one command per request is supported")
            
            command_type, args = next(iter(command_data.items()))
            
            if command_type == "visit_page":
                # Format: {"url": "https://example.com"}
                await self.controller.visit_page(self.page, args["url"])
                
            elif command_type == "back":
                await self.controller.back(self.page)
                
            elif command_type == "click_coords":
                # Format: {"x": x, "y": y}
                x = float(args["x"])
                y = float(args["y"])
                await self.controller.click_coords(self.page, x, y)
                
            elif command_type == "click_id":
                # Format: {"id": "123"}
                # Translate displayed ID to original element ID
                original_id = self._translate_displayed_id_to_original(args["id"])
                await self.controller.click_id(self.page, original_id)
                
            elif command_type == "fill_coords":
                # Format: {"x": x, "y": y, "value": "text", "press_enter": true, "delete_existing": false}
                x = float(args["x"])
                y = float(args["y"])
                value = args["value"]
                press_enter = args.get("press_enter", True)
                delete_existing = args.get("delete_existing", False)
                
                await self.controller.fill_coords(self.page, x, y, value, press_enter, delete_existing)
                    
            elif command_type == "fill_id":
                # Format: {"id": "123", "value": "text", "press_enter": true, "delete_existing": false}
                displayed_id = args["id"]
                value = args["value"]
                press_enter = args.get("press_enter", True)
                delete_existing = args.get("delete_existing", False)
                
                # Translate displayed ID to original element ID
                original_id = self._translate_displayed_id_to_original(displayed_id)
                await self.controller.fill_id(self.page, original_id, value, press_enter, delete_existing)
                    
            elif command_type == "select_option":
                # Format: {"id": "123"}
                # Translate displayed ID to original element ID
                original_id = self._translate_displayed_id_to_original(args["id"])
                await self.controller.select_option(self.page, original_id)
                    
            elif command_type == "hover_id":
                # Format: {"id": "123"}
                original_id = self._translate_displayed_id_to_original(args["id"])
                await self.controller.hover_id(self.page, original_id)
                
            elif command_type == "keypress":
                # Format: {"keys": ["ctrl", "a"]} or {"keys": ["Enter"]}
                keys = args["keys"]
                await self.controller.keypress(self.page, keys)
                    
            elif command_type == "page_down":
                # Format: {"amount": 200, "full_page": false}
                amount = args.get("amount", 200)
                full_page = args.get("full_page", False)
                await self.controller.page_down(self.page, amount, full_page)
                    
            elif command_type == "page_up":
                # Format: {"amount": 200, "full_page": false}
                amount = args.get("amount", 200)
                full_page = args.get("full_page", False)
                await self.controller.page_up(self.page, amount, full_page)
                    
            elif command_type == "scroll_id":
                # Format: {"id": "123", "direction": "down"}
                displayed_id = args["id"]
                direction = args["direction"].lower()

                # Translate displayed ID to original element ID
                original_id = self._translate_displayed_id_to_original(displayed_id)
                await self.controller.scroll_id(self.page, original_id, direction)

            elif command_type == "hover_coords":
                # Format: {"x": 100, "y": 200}
                x = float(args["x"])
                y = float(args["y"])
                await self.controller.hover_coords(self.page, x, y)

            elif command_type == "hover_and_scroll_coords":
                # Format: {"x": 100, "y": 200, "direction": "down"}
                x = float(args["x"])
                y = float(args["y"])
                direction = args.get("direction", "down").lower()
                await self.controller.hover_and_scroll_coords(self.page, x, y, direction)

            elif command_type == "sleep":
                # Format: {"duration": 2.0}
                duration = float(args["duration"])
                await self.controller.sleep(self.page, duration)

            elif command_type == "tab_and_enter":
                await self.controller.tab_and_enter(self.page)
                
            elif command_type == "get_webpage_text":
                # Format: {"n_lines": 100}
                n_lines = args.get("n_lines", 100)
                text = await self.controller.get_webpage_text(self.page, n_lines)
                return {"text": text}
                
            elif command_type == "get_page_metadata":
                # Get page metadata with guaranteed title and url fields
                metadata = await self.controller.get_page_metadata(self.page)
                return metadata
                
            elif command_type == "get_interactive_rects":
                # Get current interactive regions using the controller
                rects = await self.controller.get_interactive_rects(self.page)
                return rects
                
            elif command_type == "get_screenshot_info":
                # Return information about the last screenshot's interactive elements
                if self._last_screenshot_info:
                    return self._last_screenshot_info
                else:
                    return {"error": "No screenshot info available"}
                    
            elif command_type == "screenshot":
                # Format: {"interaction_mode": "set_of_marks"}
                interaction_mode = args.get("interaction_mode", "set_of_marks")
                screenshot_buffer = await self._screenshot(interaction_mode=interaction_mode)
                return {"status": "screenshot_taken"}
                
            else:
                raise ValueError(f"Unsupported command: {command_type}")
                
            return {"status": "success"}
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error executing command: {str(e)}")