import asyncio
import uvicorn
import logging
from fastapi import FastAPI, HTTPException, status, Response, Depends, Header
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from omniboxes.master.node_manager import NodeManager, NodeRegistration
import requests
import argparse


async def get_api_key(x_api_key: str = Header(None, alias="x-api-key")):
    """Dependency to verify the API key in the request header."""
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key is missing",
        )
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid API Key.",
        )
    return x_api_key


parser = argparse.ArgumentParser(description="OmniBox Host")
parser.add_argument("--port", type=int, default=7000, help="Port to run the server on")
parser.add_argument("--nodes", type=str, nargs='+', default=[], help="List of node URLs to register")
parser.add_argument("--workers", type=int, default=1, help="Number of worker processes to run")
parser.add_argument("--redis-registry", action="store_true", help="Enable Redis-based node discovery")
parser.add_argument("--redis-host", type=str, default="localhost", help="Redis host for node registry")
parser.add_argument("--redis-port", type=int, default=6379, help="Redis port for node registry")
args = parser.parse_args()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("omnibox-master")

# Setup Redis registry if enabled
redis_registry = None
if args.redis_registry:
    from omniboxes.common.redis_registry import RedisRegistry
    logger.info(f"Enabling Redis-based node discovery: {args.redis_host}:{args.redis_port}")
    try:
        redis_registry = RedisRegistry(
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            registry_db=1,  # Use DB 1 for registry
            logger=logger
        )
        logger.info("✅ Redis registry initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Redis registry: {e}")
        redis_registry = None

API_KEY = 'default_key'

NODE_API_KEY = API_KEY # todo: make separate

node_manager = NodeManager(api_key = NODE_API_KEY, logger = logger, redis_registry=redis_registry)

@asynccontextmanager
async def lifespan(app: FastAPI):
    for node_url in args.nodes:
        await node_manager.register_node(NodeRegistration(url = node_url))
    tasks = [
        asyncio.create_task(node_manager.update_statuses_worker()),
        asyncio.create_task(node_manager.update_nodes())
    ]
    yield
    # Shutdown code goes here

app = FastAPI(
    title="OmniBox Master Node",
    description="Manages redirection of instance operations to worker nodes",
    lifespan = lifespan)

@app.post("/get")
async def create_instance(lifetime_mins: int = 60, api_key: str = Depends(get_api_key)):
    """Get an available new instance from less occupied node"""
    node = node_manager.get_best_node()
    if node is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No available nodes with capacity to create new instance"
        )
    
    data = requests.post(f'{node.url}/get', params = {'lifetime_mins': lifetime_mins}, headers = {'x-api-key': NODE_API_KEY}).json()
    if 'instance_id' in data:
        await node_manager.lease_instance(node.hash, data['instance_id'])
        return {
            'instance_id': data['instance_id'],
            'node': node.hash,
        }
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="No available nodes with capacity to create new instance"
    )

@app.post("/reset")
async def reset(instance_id: str, node: str, api_key: str = Depends(get_api_key)):
    """Reset an existing instance by delegating to the worker node that hosts it"""
    node_info = node_manager.get_node(node)
    if node_info is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Node {node} is not found"
        )
    response = requests.post(f'{node_info.url}/reset', params={"instance_id": instance_id}, headers = {'x-api-key': NODE_API_KEY})
    return JSONResponse(content=response.json(), status_code=response.status_code)


@app.get("/probe")
async def probe(instance_id: str, node: str, api_key: str = Depends(get_api_key)):
    """Probe the instance by delegating to the worker node that hosts it"""
    node_info = node_manager.get_node(node)
    if node_info is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Node {node} is not found"
        )
    response = requests.get(f'{node_info.url}/probe', params={"instance_id": instance_id}, headers = {'x-api-key': NODE_API_KEY})
    return JSONResponse(content=response.json(), status_code=response.status_code)

@app.get("/screenshot")
async def screenshot(instance_id: str, node: str, interaction_mode: str = "set_of_marks", api_key: str = Depends(get_api_key)):
    """Make a screenshot of an existing instance by delegating to the worker node that hosts it"""
    node_info = node_manager.get_node(node)
    if node_info is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Node {node} is not found"
        )
    response = requests.get(f'{node_info.url}/screenshot', params={"instance_id": instance_id, "interaction_mode": interaction_mode}, headers = {'x-api-key': NODE_API_KEY})   
    if response.status_code == 200:
        return Response(content=response.content, media_type="image/png")

    return JSONResponse(
        content={"status": "error", "message": f"Failed to get screenshot: {response.text}"},
        status_code=response.status_code
    )

@app.post("/execute")
async def execute(command_data: Dict[str, Any], api_key: str = Depends(get_api_key)):
    """Forward execute command to the Flask server in the specified instance"""
    node = command_data.pop('node')
    instance_id = command_data.pop('instance_id')
    node_info = node_manager.get_node(node)
    if node_info is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Node {node} is not found"
        )
    response = requests.post(f'{node_info.url}/execute',
                            params={"instance_id": instance_id},
                            json = command_data, headers = {'x-api-key': NODE_API_KEY})   
    if response.status_code == 200:
        return JSONResponse(content=response.json(), status_code=response.status_code)
    
    return JSONResponse(
        content={"status": "error", "message": f"Failed to execute command: {response.text}"},
        status_code=response.status_code
    )

@app.get("/metadata")
async def metadata(instance_id: str, node: str, api_key: str = Depends(get_api_key)):
    node_info = node_manager.get_node(node)
    if node_info is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Node {node} is not found"
        )
    response = requests.get(f'{node_info.url}/metadata', params={"instance_id": instance_id}, headers = {'x-api-key': NODE_API_KEY})   
    if response.status_code == 200:
        return Response(content=response.content, media_type="image/png")

    return JSONResponse(
        content={"status": "error", "message": f"Failed to get screenshot: {response.text}"},
        status_code=response.status_code
    )

@app.get("/info")
def get_info(api_key: str = Depends(get_api_key)):
    node_info = node_manager.node_info()
    return JSONResponse(
        content={
            "nodes": [
                {
                    "url": node.url,
                    "hash": node.hash,
                    "healthy": node.healthy,
                    "capacity": node.capacity,
                    "available": node.available,
                    "instances": node.instances
                }
                for node in node_info.values()
            ]
        },
        status_code=status.HTTP_200_OK
    )


if __name__ == "__main__":
    uvicorn.run(
        "omniboxes.master.server:app",
        host="0.0.0.0",
        port=args.port,
        workers=args.workers)
