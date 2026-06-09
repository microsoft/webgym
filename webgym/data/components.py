domain_subdomain_map = {
    "Lifestyle & Leisure": [
        "Shopping", "Food & Cooking", "Sports & Fitness",
        "Health & Medicine", "Pets & Animal Welfare",
        "Fashion & Beauty", "Hobbies & DIY"
    ],
    "Entertainment": [
        "Films & TV Shows", "Gaming & Virtual Worlds",
        "Live Shows & Performances", "Music", "Books & Reading"
    ],
    "Misc.": [
        "General Info.", "News", "Legal & Government Services",
        "Real Estate", "Finance & Investment"
    ],
    "Science & Research": ["Research & Academia", "Technology & Science"],
    "Career & Education": ["Education & Learning", "Jobs & Career"],
    "Travel & Transportation": [
        "Travel & Accommodation", "Outdoor & Recreation",
        "Ticketed Activities"
    ],
}

class Action():
    def __init__(self, action, action_string):
        # Remove reasoning parameter since it's now in Response
        self.action = action
        self.action_string = action_string
    
    def __str__(self):
        return f"Action(action={self.action})"

    def __getitem__(self, idx):
        raise NotImplementedError("Action is not iterable")

class Observation():
    def __init__(self, task, image_path, ac_tree, submit=None, page_metadata=None):
        self.task = task
        self.image_path = image_path
        self.ac_tree = ac_tree
        # Keep submit for backward compatibility but it's now stored in Reward
        if submit is not None:
            import warnings
            warnings.warn("submit parameter in Observation is deprecated, use Reward.submit instead", DeprecationWarning)
        self.page_metadata = page_metadata or {}

    def __str__(self):
        return f"Observation(task={self.task}, image_path={self.image_path}, ac_tree={self.ac_tree}, page_metadata={self.page_metadata})"

    def __getitem__(self, idx):
        raise NotImplementedError("Observation is not iterable")

class Task():
    def __init__(self, task_name, domain, subdomain, website, difficulty, evaluator_reference, reference_answer, attempt_level=0, task_id=None, max_steps=None, trajectory_index=None):
        self.task_name = task_name
        self.domain = domain
        self.subdomain = subdomain
        self.website = website
        self.difficulty = difficulty
        self.evaluator_reference = evaluator_reference
        self.reference_answer = reference_answer
        self.attempt_level = attempt_level
        self.task_id = task_id  # Sequential task ID from task file
        self.max_steps = max_steps  # Per-task max steps based on difficulty
        self.trajectory_index = trajectory_index  # Unique index per task slot (distinguishes repeats)

    def __str__(self):
        return f"Task(task_id={self.task_id}, trajectory_index={self.trajectory_index}, task_name={self.task_name}, domain={self.domain}, subdomain={self.subdomain}, website={self.website}, difficulty={self.difficulty}, max_steps={self.max_steps}, evaluator_reference={self.evaluator_reference}, reference_answer={self.reference_answer}, attempt_level={self.attempt_level})"

    def __getitem__(self, idx):
        raise NotImplementedError("Task is not iterable")

class Reward():
    def __init__(self, reward, evaluation, is_blocked=False, submit=False, submission_judgment=None):
        self.reward = reward
        self.evaluation = evaluation
        self.is_blocked = is_blocked  # True if website blocked the agent
        self.submit = submit  # True if this step's screenshot should be submitted for evaluation
        self.submission_judgment = submission_judgment  # AI judge's reasoning for submit decision

    def __str__(self):
        blocked_str = ", blocked=True" if self.is_blocked else ""
        submit_str = f", submit={self.submit}" if hasattr(self, 'submit') else ""
        eval_preview = self.evaluation[:50] if isinstance(self.evaluation, str) else str(self.evaluation)[:50]
        return f"Reward(reward={self.reward}{blocked_str}{submit_str}, evaluation={eval_preview}...)"

    def __getitem__(self, idx):
        raise NotImplementedError("Reward is not iterable")

class Response():
    def __init__(self, raw_response, answering_tokens, raw_prompt=""):
        self.raw_response = raw_response  # Full original response (for debugging/logging)
        self.answering_tokens = answering_tokens  # Parsed fields used in prompts
        self.raw_prompt = raw_prompt  # Full original prompt sent to model (for debugging/logging)

    def __str__(self):
        return f"Response(answering_tokens.keys()={list(self.answering_tokens.keys())})"

    def __getitem__(self, idx):
        raise NotImplementedError("Response is not iterable")
