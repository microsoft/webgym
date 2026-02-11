from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse, Response
import uvicorn
from typing import Dict, Any
from contextlib import asynccontextmanager
import argparse
from omniboxes.node.instances.playwright_instance import PlaywrightInstance
from omniboxes.node.instances.base import InstanceBase, Status
from pathlib import Path
import redis
from omniboxes.node.logging_utils import default_logger
import asyncio
from omniboxes.node.utils import GetApiKey

parser = argparse.ArgumentParser(description="OmniBox Host")
parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
parser.add_argument('--path', type=str, default='../run', help="Path to the instance directory")
parser.add_argument('--instance', default='playwright', help="Instance type (only playwright supported)")
parser.add_argument('--redis_port', type=int, default=6379, help="Port for Redis connection")
parser.add_argument('--redis_host', type=str, default='localhost', help="Host for Redis connection")
args = parser.parse_args()


logger = default_logger()
API_KEY = 'default_key'


class State:
    def __init__(self):
        self.in_lease = False
        # Scheduled resets disabled - lifecycle is 100% manual

state = State()

instance = PlaywrightInstance(instance_num=0, logger=logger)

r = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)

async def redis_watchdog():
    """Periodically verify this instance is registered in Redis.

    Handles the case where Redis restarts and loses the available/in_use sets.
    If this port is missing from both sets, reset any orphaned lease and
    re-register as available.
    """
    port_str = str(args.port)
    while True:
        await asyncio.sleep(30)
        try:
            if r.sismember('available', port_str):
                continue

            in_use_members = r.smembers('in_use')
            in_in_use = any(
                m.decode('utf-8').endswith(f':{port_str}')
                for m in in_use_members
            )
            if in_in_use:
                continue

            # Port missing from both sets — Redis lost our registration
            logger.warning(f"Port {port_str} missing from Redis. Re-registering...")

            if state.in_lease:
                logger.warning(f"Resetting orphaned lease on port {port_str}")
                instance.id = None
                await instance.reset()
                state.in_lease = False

            r.sadd('available', port_str)
            logger.info(f"Re-registered port {port_str} as available")

        except redis.ConnectionError:
            logger.warning("Redis connection error during watchdog check")
        except Exception as e:
            logger.warning(f"Redis watchdog error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await instance.create()
    r.sadd('available', args.port)
    watchdog_task = asyncio.create_task(redis_watchdog())
    yield
    watchdog_task.cancel()
    r.srem('available', args.port)
    await instance.delete()

app = FastAPI(lifespan=lifespan)

# Scheduled resets disabled - lifecycle is 100% manual
# Instances only reset when explicitly called via /reset endpoint

@app.post("/get")
async def get_instance(lifetime_mins: int = 60, api_key: str = Depends(GetApiKey(API_KEY))):
    """
    Get an available instance from the pool.

    NOTE: lifetime_mins parameter is accepted for API compatibility but is ignored.
    Instances will NOT be automatically reset - you must call /reset explicitly.
    """
    if state.in_lease:
        raise HTTPException(status_code=400, detail="Instance is already in lease")
    if instance.status < Status.STARTED:
        raise HTTPException(status_code=503, detail="Instance is not ready yet")

    state.in_lease = True
    r.srem('available', args.port)

    logger.info(f"Allocated instance {instance.id} - manual reset required, no automatic timeout")

    return {"instance_id": instance.id}

@app.post("/reset")
async def reset_instance(instance_id: str, api_key: str = Depends(GetApiKey(API_KEY))):
    """Reset an instance to its initial state and make it available again"""
    if instance_id != instance.id:
        logger.warning(f"Reset called with mismatched instance_id: {instance_id} != {instance.id}")
        raise HTTPException(status_code=400, detail=f"Invalid instance UUID: {instance_id}")

    # No scheduled resets to cancel

    instance.id = None
    await instance.reset()
    state.in_lease = False
    r.sadd('available', args.port)
    logger.info(f"Instance {instance.id} is ready")
    return {"status": "success"}

@app.get("/probe")
async def probe_instance(instance_id: str, api_key: str = Depends(GetApiKey(API_KEY))):
    """Forward probe request to the Flask server in the specified instance"""
    """Reset an instance to its initial state and make it available again"""
    if instance_id != instance.id:
        raise HTTPException(status_code=400, detail=f"Invalid instance UUID: {instance_id}")    
    return await instance.probe()

@app.get("/screenshot")
async def get_instance_screenshot(instance_id: str, interaction_mode: str = "set_of_marks", api_key: str = Depends(GetApiKey(API_KEY))):
    """Forward screenshot request to the Flask server in the specified instance"""
    screenshot = await instance.screenshot(interaction_mode=interaction_mode)
    # Return full image to avoid streaming closed errors
    try:
        content = screenshot.getvalue()
    except AttributeError:
        # If it's raw bytes iterable
        content = b"".join(screenshot)
    return Response(content=content, media_type="image/png")

@app.post("/execute")
async def execute_instance_command(instance_id: str, command_data: Dict[str, Any], api_key: str = Depends(GetApiKey(API_KEY))):
    """Forward execute command to the Flask server in the specified instance"""
    return await instance.execute(command_data)

@app.get("/info")
async def get_info(api_key: str = Depends(GetApiKey(API_KEY))):
    """Get an available instance from the pool"""
    return {
        'in_lease': state.in_lease,
        'ready': instance.status >= Status.STARTED}

@app.get("/metadata")
async def metadata(instance_id: str, api_key: str = Depends(GetApiKey(API_KEY))):
   return await instance.metadata()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info", loop="asyncio", workers=1)

