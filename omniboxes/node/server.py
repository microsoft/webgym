from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse, Response
import uvicorn
from typing import Dict, Any
from contextlib import asynccontextmanager
import argparse
from omniboxes.node.instances.playwright_instance import PlaywrightInstance
from omniboxes.node.logging_utils import default_logger
from omniboxes.node.utils import GetApiKey
from fastapi.responses import JSONResponse
from pathlib import Path
import redis
import httpx

parser = argparse.ArgumentParser(description="OmniBox Host")
parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
parser.add_argument('--lifetime_max', type=int, default=60, help="Maximum lifetime of an instance in minutes")
parser.add_argument('--redis_port', type=int, default=6379, help="Port for Redis connection")
parser.add_argument('--redis_host', type=str, default='localhost', help="Host for Redis connection")
parser.add_argument('--workers', type=int, default=1, help="Number of worker processes to run")
args = parser.parse_args()

API_KEY = 'default_key'

logger = default_logger()

INSTANCE_API_KEY = API_KEY    # todo: make separate

r = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ...
    yield
    ...

app = FastAPI(lifespan=lifespan)

@app.post("/get")
async def get_instance(lifetime_mins: int = 60, api_key: str = Depends(GetApiKey(API_KEY))):
    """Get an available instance from the pool"""
    if lifetime_mins < 0 or lifetime_mins > args.lifetime_max:
        raise HTTPException(status_code=400, detail=f"Invalid lifetime_mins: {lifetime_mins}. Must be between 0 and {args.lifetime_max} minutes.")
    port = r.spop('available')
    if not port:
        raise HTTPException(status_code=503, detail="No instances available")
    async with httpx.AsyncClient() as client:
        try:
            port = port.decode('utf-8')
            response = await client.post(f"http://127.0.0.1:{port}/get", timeout=None, headers={"x-api-key": INSTANCE_API_KEY})
            instance_id = f'{response.json()["instance_id"]}:{port}'
            
            # Track the in-use instance
            r.sadd('in_use', instance_id)
            
            return JSONResponse(content = {'instance_id': instance_id}, status_code=response.status_code)
        except httpx.RequestError as e:
            # If there was an error, put the port back to available
            r.sadd('available', port)
            raise HTTPException(status_code=500, detail=f"Error leasing the instance")


@app.post("/reset")
async def reset_instance(instance_id: str, api_key: str = Depends(GetApiKey(API_KEY))):
    """Reset an instance to its initial state and make it available again"""
    # Check if this instance is actually in use before resetting
    # This prevents race conditions where an instance has already been reallocated
    full_instance_id = instance_id
    in_use_instances = [m.decode('utf-8') for m in r.smembers('in_use')]
    if full_instance_id not in in_use_instances:
        logger.info(f"Instance {full_instance_id} not in use (already released or reallocated), skipping reset")
        raise HTTPException(status_code=400, detail="Instance not in use or already released")

    instance_id, port = instance_id.split(':')
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"http://127.0.0.1:{port}/reset", params = {'instance_id': instance_id}, timeout=None, headers={"x-api-key": INSTANCE_API_KEY})

            # Only remove from in_use if reset succeeded
            if response.status_code == 200:
                full_instance_id = f'{instance_id}:{port}'
                r.srem('in_use', full_instance_id)
                logger.info(f"Successfully reset and released instance {full_instance_id}")
            else:
                logger.warning(f"Instance reset failed with status {response.status_code}, keeping in 'in_use': {response.text}")

            return JSONResponse(content = response.json(), status_code=response.status_code)
        except httpx.RequestError as e:
            logger.error(f"Network error resetting instance {instance_id}:{port}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error resetting the instance")    

@app.get("/probe")
async def probe_instance(instance_id: str, api_key: str = Depends(GetApiKey(API_KEY))):
    """Forward probe request to the Flask server in the specified instance"""
    instance_id, port = instance_id.split(':')
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"http://127.0.0.1:{port}/probe", params = {'instance_id': instance_id}, timeout=None, headers={"x-api-key": INSTANCE_API_KEY})
            return JSONResponse(content = response.json(), status_code=response.status_code)
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Error probing the instance")          

@app.get("/screenshot")
async def get_instance_screenshot(instance_id: str, interaction_mode: str = "set_of_marks", api_key: str = Depends(GetApiKey(API_KEY))):
    """Forward screenshot request to the Flask server in the specified instance"""
    instance_id, port = instance_id.split(':')
    async with httpx.AsyncClient() as client:
        try:
            # Fetch the full content to avoid streaming errors
            proxied = await client.get(
                f"http://127.0.0.1:{port}/screenshot",
                params={'instance_id': instance_id, 'interaction_mode': interaction_mode},
                timeout=None,
                headers={"x-api-key": INSTANCE_API_KEY}
            )
            return Response(content=proxied.content, media_type=proxied.headers.get("content-type"))
        except httpx.RequestError:
            raise HTTPException(status_code=500, detail="Error getting screenshot from instance")
    

@app.post("/execute")
async def execute_instance_command(instance_id: str, command_data: Dict[str, Any], api_key: str = Depends(GetApiKey(API_KEY))):
    """Forward execute command to the Flask server in the specified instance"""
    instance_id, port = instance_id.split(':')    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"http://127.0.0.1:{port}/execute", params={'instance_id': instance_id}, json=command_data, timeout=None, headers={"x-api-key": INSTANCE_API_KEY})
            return JSONResponse(content = response.json(), status_code=response.status_code)
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Error executing command on instance")

@app.get("/info")
async def get_info(api_key: str = Depends(GetApiKey(API_KEY))):
    """Return information about instances"""
    available = r.scard('available')
    in_use_set = r.smembers('in_use')
    in_use = [member.decode('utf-8') for member in in_use_set]
    
    # Calculate total capacity as available + in_use
    capacity = available + len(in_use)
    
    return {
        "available": available,
        "capacity": capacity,
        "in_use": in_use
    }

@app.get("/metadata")
async def metadata(instance_id: str, api_key: str = Depends(GetApiKey(API_KEY))):
    instance_id, port = instance_id.split(':')
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"http://127.0.0.1:{port}/metadata", params = {'instance_id': instance_id}, timeout=None, headers={"x-api-key": INSTANCE_API_KEY})
            return JSONResponse(content = response.json(), status_code=response.status_code)
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Error getting the instance metadata")     


if __name__ == "__main__":
    uvicorn.run('omniboxes.node.server:app', host="0.0.0.0", port=args.port, log_level="info", loop="asyncio", workers=args.workers)
