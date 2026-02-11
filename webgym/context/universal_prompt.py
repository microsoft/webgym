# Universal prompt format for web agents
# Adapts to both set_of_marks and coordinates modes

class UniversalPrompt:
    """Universal prompt structure for all models, adapted for interaction mode"""

    @staticmethod
    def build_prompt(task: str, web_url: str, ac_tree: str, memory: str, history: list, interaction_mode: str) -> str:
        """
        Build prompt adapted for interaction mode.

        Args:
            task: Task description
            web_url: Current webpage URL
            ac_tree: Accessibility tree (only used in set_of_marks mode)
            memory: Current memory state (raw string from model output, not parsed)
            history: List of "Thought, Action, Observation" strings (last 15 steps)
            interaction_mode: "set_of_marks" or "coordinates"
        """

        # Format history (last 15 steps only) - history already includes step IDs in the strings
        history_text = ""
        if history:
            # Calculate starting step ID based on total number of history items
            # Since we keep last 15 steps, if we have fewer than 15, start from 0
            total_history = len(history[-15:])
            start_idx = len(history) - total_history  # This gives us the actual step ID

            for idx, h in enumerate(history[-15:]):
                step_id = start_idx + idx
                history_text += f"{step_id}. {h}\n"

        # Memory is already a string (raw model output), use as-is
        memory_str = memory if memory else ""

        # Web elements section (only for set_of_marks mode)
        web_elements_section = ""
        if interaction_mode == 'set_of_marks':
            web_elements = ac_tree if ac_tree else "No web elements available"
            web_elements_section = f"""E. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.
{web_elements}

"""

        # Action format based on interaction mode
        if interaction_mode == 'set_of_marks':
            action_examples = """B. Correspondingly, Action should STRICTLY follow the format:
    - Click [Numerical_Label]. For example the Action "Click [1]" means clicking on the web element with a Numerical_Label of "1".
    - Hover [Numerical_Label]. For example the Action "Hover [10]" means hovering over the web element with a Numerical_Label of "10".
    - Type [Numerical_Label] [content]. For example the Action "Type [2] [5$]" means typing "5$" in the web element with a Numerical_Label of "2".
    - Scroll [Numerical_Label or WINDOW] [up or down]. For example the Action "Scroll [6] [up]" means scrolling up in the web element with a Numerical_Label of "6".
    - Wait. For example the Action "Wait" means waiting for 5 seconds.
    - GoBack. For example the Action "GoBack" means going back to the previous webpage.
    - TabAndEnter. For example the Action "TabAndEnter" means pressing Tab then Enter.
    - ANSWER [content]. For example the Action "ANSWER [Guatemala]" means answering the task with "Guatemala"."""

            element_description = "This screenshot will feature Numerical Labels placed in the upper right corner of each Web Element (or lower right corner if the upper right is outside the screen)."
            element_guideline = "6) You can only interact with web elements in the screenshot that have a numerical label. Before giving the action, double-check that the numerical label appears on the screen."
            browsing_guideline = "3) Focus on the numerical labels in the upper right corner of each rectangle (element), or lower right if outside screen. Ensure you don't mix them up with other numbers (e.g. Calendar) on the page."
        else:  # coordinates
            action_examples = """B. Correspondingly, Action should STRICTLY follow the format:
    - Click [x, y]. For example the Action "Click [257, 144]" means clicking at coordinates (257, 144).
    - Hover [x, y]. For example the Action "Hover [143, 688]" means hovering at coordinates (143, 688) without clicking.
    - Type [x, y] [content]. For example the Action "Type [129, 44] [Boston]" means typing "Boston" at coordinates (129, 44).
    - Scroll [x, y or WINDOW] [up or down]. For example the Action "Scroll [228, 51] [up]" means scrolling up at coordinates (228, 51). Use WINDOW for whole page scroll.
    - HoverAndScroll [x, y] [up or down]. For example the Action "HoverAndScroll [228, 51] [down]" means hovering at coordinates (228, 51) and then scrolling down the element at that location. This is useful for scrolling dropdown menus or scrollable containers.
    - Wait. For example the Action "Wait" means waiting for 5 seconds.
    - GoBack. For example the Action "GoBack" means going back to the previous webpage.
    - TabAndEnter. For example the Action "TabAndEnter" means pressing Tab then Enter.
    - ANSWER [content]. For example the Action "ANSWER [Guatemala]" means answering the task with "Guatemala"."""

            element_description = "Carefully analyze the visual information to identify the precise pixel coordinates for interaction."
            element_guideline = "6) Coordinates should be precise pixel locations based on careful analysis of the screenshot."
            browsing_guideline = "3) Analyze the screenshot carefully to determine exact pixel coordinates for each action."

        prompt = f"""Imagine you are an Agent operating a computer, much like how humans do, capable of moving the mouse,
clicking the mouse buttons, and typing text with the keyboard.
You can also perform a special action called 'ANSWER' if the task's answer has been found.
You are tasked with completing a final mission: "{task}", Please interact with {web_url} and get the answer. Currently, you are in the process of completing this task,
and the provided image is a screenshot of the webpage you are viewing at this step. {element_description}
Carefully analyze the visual information to identify the Web Element that requires interaction, then follow the guidelines and choose one of the following actions:
1. Click a Web Element.
2. Delete existing content in a textbox and then type content.
3. Scroll up or down. Multiple scrolls are allowed to browse the webpage. Pay attention!! The default scroll is the whole window. If the scroll widget is located in a certain area of the webpage (like a dropdown menu or scrollable container), use HoverAndScroll at those coordinates to scroll that specific element.
4. Wait. Typically used to wait for unfinished webpage processes, with a duration of 5 seconds.
5. Go back, returning to the previous webpage.
6. Answer. This action should only be chosen when all questions in the task have been solved.

Here's some additional information:
A. Your final task is {task}.
{action_examples}
C. You have **already performed the following actions** (format: Step_ID. Thought: ..., Action: ..., Action Effect: ...):
{history_text}D. The "Memory" only stores the information obtained from the web page that is relevant to the task,  and the "Memory" is strictly in JSON format. For example: {{"user_email_address": "test@163.com", "user_email_password": "123456", "jack_email_address": "jack@163.com"}}.
The "Memory" does not include future plans, descriptions of current actions, or other reflective content; it only records visual information that is relevant to the task obtained from the screenshot. The "Memory" in the current step as follow:
Memory:{memory_str}
{web_elements_section}Note: The screenshot you are currently viewing is the LATEST screenshot at this step. When you provide your response, you should analyze THIS current screenshot to decide your next action.
Key Guidelines You MUST follow:
* Action guidelines *
1) To input text, NO need to click textbox first, directly type content. After typing, the system automatically hits `ENTER` key. Sometimes you should click the search button to apply search filters. Try to use simple language when searching.
2) You must Distinguish between textbox and search button, don't type content into the button! If no textbox is found, you may need to click the search button first before the textbox is displayed.
3) Execute only one action per iteration.
4) STRICTLY Avoid repeating the same action if the webpage remains unchanged. You may have selected the wrong web element or numerical label. Continuous use of the Wait is also NOT allowed.
5) When a complex Task involves multiple questions or steps, select "ANSWER" only at the very end, after addressing all of these questions (steps). Flexibly combine your own abilities with the information in the web page. Double check the formatting requirements in the task when ANSWER.
{element_guideline}
7) If any web elements in the screenshot have not finished loading, you need to wait for them to load completely.

* Web Browsing Guidelines *
1) Don't interact with useless web elements like Login, Sign-in, donation that appear in Webpages. Pay attention to Key Web Elements like search textbox and menu.
2) Visit video websites like YouTube is allowed BUT you can't play videos. You are NOT allowed to download any files (PDFs, documents, etc.).
{browsing_guideline}
4) Focus on the date in task, you must look for results that match the date. It may be necessary to find the correct year, month and day at calendar.
5) During the process of browsing web page content, to ensure the complete acquisition of the target content, it may be necessary to scroll down the page until confirming the appearance of the end marker for the target content. For example, when new webpage information appears, or the webpage scroll bar has reached the bottom, etc.
6) Try your best to find the answer that best fits the task, if any situations that do not meet the task requirements occur during the task, correct the mistakes promptly.

Your reply should strictly follow the format:
Thought: {{Your brief thoughts (briefly summarize the info that will help ANSWER)}}
Action: {{One Action format you choose}}
Memory_Updated: {{The latest version of memory generated by modifying or supplementing the original memory content based on the visual information in the current screenshot.}}

Then the User will provide:
Observation: {{A labeled screenshot Given by User}}"""

        return prompt
