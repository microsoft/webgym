"""
Navigation Error Logger

Thread-safe utility to log websites that encounter navigation errors.
Maintains a persistent list across runs and avoids duplicates.
"""

import os
import fcntl
from urllib.parse import urlparse


class NavigationErrorLogger:
    """
    Logs websites that fail navigation to a persistent file.
    Thread-safe and duplicate-free.
    """

    def __init__(self, log_file_path="/data/v-baihao/logs/navigation_error_websites.txt"):
        """
        Initialize the navigation error logger.

        Args:
            log_file_path: Path to the log file (default: /data/v-baihao/logs/navigation_error_websites.txt)
        """
        self.log_file_path = log_file_path
        self._ensure_log_file_exists()

    def _ensure_log_file_exists(self):
        """Create the log directory and file if they don't exist"""
        log_dir = os.path.dirname(self.log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # Create the file if it doesn't exist
        if not os.path.exists(self.log_file_path):
            with open(self.log_file_path, 'w') as f:
                pass  # Create empty file

    def _normalize_website(self, url):
        """
        Normalize a URL to its domain for consistent logging.

        Args:
            url: The URL to normalize

        Returns:
            The normalized domain (e.g., "example.com" from "https://example.com/path")
        """
        try:
            parsed = urlparse(url)
            # Return domain with scheme for better identification
            domain = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme else parsed.netloc
            return domain if domain else url
        except Exception:
            # If parsing fails, return the original URL
            return url

    def _read_existing_websites(self, file_handle):
        """
        Read existing websites from the log file.

        Args:
            file_handle: Open file handle

        Returns:
            Set of existing website domains
        """
        file_handle.seek(0)
        existing_websites = set()
        for line in file_handle:
            line = line.strip()
            if line:  # Skip empty lines
                existing_websites.add(line)
        return existing_websites

    def log_navigation_error(self, url, task_id=None):
        """
        Log a website that encountered a navigation error.
        Thread-safe with file locking. Avoids duplicates.

        Args:
            url: The URL that failed to navigate
            task_id: Optional task ID for debugging (not stored in file)
        """
        website = self._normalize_website(url)

        # Use file locking for thread safety
        with open(self.log_file_path, 'a+') as f:
            try:
                # Acquire exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)

                # Read existing websites
                existing_websites = self._read_existing_websites(f)

                # Check if website is already logged
                if website not in existing_websites:
                    # Append new website
                    f.write(f"{website}\n")
                    f.flush()
                    if task_id:
                        print(f"📝 Logged navigation error for {website} (task: {task_id})")
                    else:
                        print(f"📝 Logged navigation error for {website}")
                else:
                    # Website already exists, skip
                    if task_id:
                        print(f"ℹ️  Navigation error for {website} already logged (task: {task_id})")
            finally:
                # Release lock
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def get_error_websites(self):
        """
        Get all websites that have encountered navigation errors.

        Returns:
            Set of website domains that have navigation errors
        """
        with open(self.log_file_path, 'r') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Shared lock for reading
            try:
                return self._read_existing_websites(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def is_website_logged(self, url):
        """
        Check if a website is already logged as having navigation errors.

        Args:
            url: The URL to check

        Returns:
            True if the website is already logged, False otherwise
        """
        website = self._normalize_website(url)
        error_websites = self.get_error_websites()
        return website in error_websites


# Global singleton instance
_global_logger = None


def set_navigation_error_logger(log_file_path):
    """
    Set the global navigation error logger with a custom log file path.
    Should be called once at initialization before any navigation errors occur.

    Args:
        log_file_path: Path to the log file
    """
    global _global_logger
    _global_logger = NavigationErrorLogger(log_file_path)


def get_navigation_error_logger():
    """
    Get the global navigation error logger instance.

    Returns:
        NavigationErrorLogger instance
    """
    global _global_logger
    if _global_logger is None:
        # Check for environment variable set by parent process
        log_path = os.environ.get('WEBGYM_NAV_ERROR_LOG_PATH')
        if log_path:
            _global_logger = NavigationErrorLogger(log_path)
        else:
            # Fallback to a relative path if not configured
            _global_logger = NavigationErrorLogger("./logs/navigation_error_websites.txt")
    return _global_logger


def log_navigation_error(url, task_id=None):
    """
    Convenience function to log a navigation error.

    Args:
        url: The URL that failed to navigate
        task_id: Optional task ID for debugging
    """
    logger = get_navigation_error_logger()
    logger.log_navigation_error(url, task_id)
