"""
Redis-based Service Registry for OmniBoxes Multi-Node Deployment

Provides node discovery and registration using Redis as a distributed registry.
This replaces Azure Key Vault for manual multi-machine deployments.

The registry stores node information in Redis with the following schema:
- Key: 'omnibox:nodes:{hostname}' -> Value: 'http://{ip}:{port}'
- Expiration: 120 seconds (nodes must refresh)
- Master polls every 10 seconds to discover new/dead nodes
"""

import redis
import socket
import time
import logging
from typing import Dict, Optional, List


class RedisRegistry:
    """Service registry using Redis for node discovery"""

    def __init__(self,
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 registry_db: int = 1,  # Use DB 1 for registry (DB 0 for node operations)
                 ttl: int = 120,  # Node registration TTL in seconds
                 logger: Optional[logging.Logger] = None):
        """
        Initialize Redis registry client

        Args:
            redis_host: Redis server host (can be remote master's IP)
            redis_port: Redis server port
            registry_db: Redis database number for registry data
            ttl: Time-to-live for node registrations (seconds)
            logger: Optional logger instance
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.registry_db = registry_db
        self.ttl = ttl
        self.logger = logger or self._default_logger()

        try:
            self.client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=registry_db,
                socket_connect_timeout=5,
                socket_timeout=5,
                decode_responses=True  # Return strings instead of bytes
            )
            # Test connection
            self.client.ping()
            self.logger.info(f"✅ Connected to Redis registry at {redis_host}:{redis_port}")
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Redis registry: {e}")
            raise

    def _default_logger(self) -> logging.Logger:
        """Create default logger"""
        logger = logging.getLogger("redis-registry")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def register_node(self, node_url: str, refresh: bool = False) -> bool:
        """
        Register a node in the registry

        Args:
            node_url: Node URL (e.g., 'http://10.0.0.5:8080')
            refresh: If True, this is a refresh (don't log)

        Returns:
            True if registration succeeded
        """
        hostname = socket.gethostname()
        key = f"omnibox:nodes:{hostname}"

        try:
            self.client.setex(key, self.ttl, node_url)
            if not refresh:
                self.logger.info(f"✅ Registered node: {hostname} -> {node_url} (TTL: {self.ttl}s)")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to register node {hostname}: {e}")
            return False

    def get_nodes(self) -> Dict[str, str]:
        """
        Get all registered nodes

        Returns:
            Dict mapping hostname -> node_url
        """
        try:
            keys = self.client.keys("omnibox:nodes:*")
            nodes = {}

            for key in keys:
                hostname = key.replace("omnibox:nodes:", "")
                node_url = self.client.get(key)
                if node_url:
                    nodes[hostname] = node_url

            return nodes
        except Exception as e:
            self.logger.error(f"❌ Failed to get nodes from registry: {e}")
            return {}

    def unregister_node(self, hostname: Optional[str] = None) -> bool:
        """
        Unregister a node from the registry

        Args:
            hostname: Hostname to unregister (defaults to current machine)

        Returns:
            True if unregistration succeeded
        """
        if hostname is None:
            hostname = socket.gethostname()

        key = f"omnibox:nodes:{hostname}"

        try:
            self.client.delete(key)
            self.logger.info(f"✅ Unregistered node: {hostname}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to unregister node {hostname}: {e}")
            return False

    def get_local_ip(self) -> str:
        """
        Get the local IP address for registration

        Returns:
            Local IP address (prefers non-localhost)
        """
        hostname = socket.gethostname()

        try:
            # Try to get IP by connecting to external address (doesn't actually connect)
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception:
            # Fallback to hostname resolution
            try:
                return socket.gethostbyname(hostname)
            except Exception:
                self.logger.warning("⚠️  Could not determine local IP, using localhost")
                return "127.0.0.1"

    def start_heartbeat(self, node_url: str, interval: int = 30):
        """
        Start a background heartbeat to keep node registration alive

        Args:
            node_url: Node URL to register
            interval: Heartbeat interval in seconds (should be < TTL)

        Note: This is a blocking call that runs forever. Run in a thread.
        """
        self.logger.info(f"🫀 Starting heartbeat for {node_url} (every {interval}s)")

        while True:
            self.register_node(node_url, refresh=True)
            time.sleep(interval)
