# webgym/utils/blocklist_manager.py
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Set
from urllib.parse import urlparse
import fcntl


class BlocklistManager:
    """Manages blocked websites and timestamps"""

    def __init__(self, save_path: str):
        """
        Initialize BlocklistManager

        Args:
            save_path: Directory where blocklist.jsonl will be stored
        """
        self.save_path = save_path
        self.blocklist_file = os.path.join(save_path, "blocklist.jsonl")
        self.block_duration_hours = 12

        # Ensure save_path exists
        os.makedirs(save_path, exist_ok=True)

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            # Remove 'www.' prefix if present
            domain = parsed.netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except Exception:
            return url.lower()

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        return datetime.now().isoformat()

    def _is_timestamp_expired(self, timestamp_str: str) -> bool:
        """Check if timestamp is older than block_duration_hours"""
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            cutoff_time = datetime.now() - timedelta(hours=self.block_duration_hours)
            return timestamp < cutoff_time
        except Exception:
            # If we can't parse the timestamp, consider it expired
            return True

    def _read_blocklist(self) -> List[Dict]:
        """Read all entries from blocklist.jsonl"""
        entries = []
        if not os.path.exists(self.blocklist_file):
            return entries

        try:
            with open(self.blocklist_file, 'r') as f:
                # Use file locking to prevent concurrent access issues
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                entry = json.loads(line)
                                entries.append(entry)
                            except json.JSONDecodeError:
                                continue
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            print(f"Warning: Error reading blocklist: {e}")

        return entries

    def _write_blocklist(self, entries: List[Dict]) -> None:
        """Write entries to blocklist.jsonl, overwriting the file"""
        try:
            with open(self.blocklist_file, 'w') as f:
                # Use file locking to prevent concurrent access issues
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    for entry in entries:
                        f.write(json.dumps(entry) + '\n')
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            print(f"Warning: Error writing blocklist: {e}")

    def _append_to_blocklist(self, entry: Dict) -> None:
        """Append a single entry to blocklist.jsonl"""
        try:
            with open(self.blocklist_file, 'a') as f:
                # Use file locking to prevent concurrent access issues
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(entry) + '\n')
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            print(f"Warning: Error appending to blocklist: {e}")

    def record_blocked_website(self, url: str, task_id: str = None, screenshot_path: str = None) -> None:
        """
        Record a website as blocked with current timestamp

        Args:
            url: The URL of the website that blocked the agent
            task_id: Unique identifier for the task (optional)
            screenshot_path: Path to screenshot when blocking occurred (optional)
        """
        domain = self._extract_domain(url)
        timestamp = self._get_current_timestamp()

        entry = {
            "domain": domain,
            "url": url,
            "blocked_at": timestamp
        }

        # Add optional fields if provided
        if task_id:
            entry["task_id"] = task_id
        if screenshot_path:
            entry["screenshot_path"] = screenshot_path

        print(f"Recording blocked website: {domain} at {timestamp}" +
              (f" (task: {task_id}, screenshot: {screenshot_path})" if task_id or screenshot_path else ""))
        self._append_to_blocklist(entry)

    def clean_expired_entries(self) -> None:
        """Remove entries older than block_duration_hours from blocklist"""
        entries = self._read_blocklist()
        valid_entries = []

        for entry in entries:
            timestamp = entry.get('blocked_at', '')
            if not self._is_timestamp_expired(timestamp):
                valid_entries.append(entry)

        # Only rewrite if we actually removed some entries
        if len(valid_entries) != len(entries):
            print(f"Cleaning blocklist: removed {len(entries) - len(valid_entries)} expired entries")
            self._write_blocklist(valid_entries)

    def get_blocked_domains(self) -> Set[str]:
        """
        Get set of currently blocked domains (not expired)

        Returns:
            Set of blocked domain names
        """
        # First clean expired entries
        self.clean_expired_entries()

        # Then get the current blocked domains
        entries = self._read_blocklist()
        blocked_domains = set()

        for entry in entries:
            domain = entry.get('domain', '')
            timestamp = entry.get('blocked_at', '')

            if domain and not self._is_timestamp_expired(timestamp):
                blocked_domains.add(domain)

        return blocked_domains

    def is_website_blocked(self, url: str) -> bool:
        """
        Check if a website is currently blocked

        Args:
            url: URL to check

        Returns:
            True if the website is blocked, False otherwise
        """
        domain = self._extract_domain(url)
        blocked_domains = self.get_blocked_domains()
        return domain in blocked_domains

    def get_blocklist_stats(self) -> Dict:
        """Get statistics about the blocklist"""
        entries = self._read_blocklist()
        current_blocked = self.get_blocked_domains()

        return {
            "total_entries": len(entries),
            "currently_blocked": len(current_blocked),
            "blocked_domains": list(current_blocked)
        }

