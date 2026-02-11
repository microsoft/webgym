import asyncio
import httpx
import logging
from fastapi import FastAPI, HTTPException, status, Response
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import omniboxes.master.logging_utils


class NodeRegistration(BaseModel):
    url: str


class NodeStatus(BaseModel):
    url: str
    hash: str
    healthy: bool
    capacity: int
    available: int
    instances: List[str]

    @staticmethod
    def failed(url):
        return NodeStatus(url=url, hash=url_hash(url), healthy=False, capacity=0, available=0, instances=[])


def url_hash(url: str) -> str:
    return url


def _default_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("omnibox-master")


class NodeManager:
    def __init__(self, api_key, logger = None, update_timeout: int = 10, node_update_timeout: int = 10, **kwargs):
        self._nodes = {}
        self._static_nodes = {}  # Nodes added via --nodes (never removed by discovery)
        self.api_key = api_key
        self._node_info = {}
        self.logger = logger or _default_logger()
        self.update_timeout = update_timeout
        self.node_update_timeout = node_update_timeout
        # Extract optional Redis registry for service discovery
        self.redis_registry = kwargs.get('redis_registry', None)
        self._lock = asyncio.Lock()

    async def _get_status(self, node_url):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                self.logger.info(f"Node {node_url} health check")
                response = await client.get(f"{node_url}/info", headers = {'x-api-key': self.api_key})
                if response.status_code != 200:
                    return NodeStatus.failed(node_url)
                data = response.json()
                return NodeStatus(
                    url = node_url,
                    hash = url_hash(node_url),
                    healthy = True,
                    capacity = data.get("capacity", 0),
                    available = data.get("available", 0),
                    instances = data.get("in_use", [])
                )
        except Exception as e:
            self.logger.warning(f"Node {node_url} health check failed: {str(e)}")
            return NodeStatus.failed(node_url)

    async def update_statuses(self):
        await self._lock.acquire()
        try:
            # Remove stale entries not in _nodes
            for node_hash in list(self._node_info.keys()):
                if node_hash not in self._nodes:
                    del self._node_info[node_hash]
            for node_hash in self._nodes.keys():
                self._node_info[node_hash] = await self._get_status(node_hash)
        finally:
            self._lock.release()

    async def update_nodes(self):
        """
        Periodically update node list from service discovery (Redis)
        This enables automatic node discovery without manual --nodes configuration
        """
        while True:
            # Redis registry discovery (for multi-machine deployments)
            if self.redis_registry:
                node_info = self.redis_registry.get_nodes()
                if node_info:
                    self.logger.info(f"Discovered {len(node_info)} nodes from Redis: {list(node_info.values())}")
                    await self._lock.acquire()
                    try:
                        # Build discovered nodes from registry (hostname -> url)
                        discovered = {}
                        for hostname, node_url in node_info.items():
                            discovered[url_hash(node_url)] = NodeRegistration(url=node_url)

                        # Merge: static nodes + discovered nodes
                        merged = dict(self._static_nodes)
                        merged.update(discovered)

                        # Check for new nodes
                        for node_hash, node_reg in merged.items():
                            if node_hash not in self._nodes:
                                self.logger.info(f"✅ New node discovered: {node_reg.url}")

                        # Check for removed nodes (only non-static ones)
                        for node_hash in list(self._nodes.keys()):
                            if node_hash not in merged and node_hash not in self._static_nodes:
                                self.logger.info(f"⚠️  Node removed (heartbeat expired): {self._nodes[node_hash].url}")

                        self._nodes = merged
                    finally:
                        self._lock.release()
                else:
                    if not self._nodes:
                        self.logger.warning("⚠️  No nodes registered in Redis yet")

            await asyncio.sleep(self.node_update_timeout)

    async def lease_instance(self, url_hash, _):
        await self._lock.acquire()
        try:
            self._node_info[url_hash].available -= 1
        finally:
            self._lock.release()

    async def update_statuses_worker(self):
        while True:
            await self.update_statuses()
            await asyncio.sleep(self.update_timeout)

    async def register_node(self, node: NodeRegistration):
        await self._lock.acquire()
        try:
            node_h = url_hash(node.url)
            self._nodes[node_h] = node
            self._static_nodes[node_h] = node  # Track as static (won't be removed by discovery)
            self._node_info[node_h] = await self._get_status(node.url)
        finally:
            self._lock.release()          

    async def unregister_node(self, node_url: str):
        """Unregister a worker node from the master"""
        node_hash = url_hash(node_url)
        if node_hash not in self._nodes:
            self.logger.warning(f"Node {node_url} not found for unregistration")
            return False
        
        await self._lock.acquire()
        try:
            self._nodes.pop(node_hash, None)
            self._node_info.pop(node_hash, None)
        finally:
            self._lock.release()
        return True
        
    def get_best_node(self) -> Optional[str]:
        result = None
        available = 0
        for node in self._node_info.values():
            if node.healthy and node.available > available:
                result = node
                available = node.available
        return result
    
    def get_node(self, hash: str) -> Optional[NodeStatus]:
        return self._node_info.get(hash, None)

    def node_info(self):
        return self._node_info
    
