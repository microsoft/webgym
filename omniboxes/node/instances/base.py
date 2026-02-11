from abc import ABC, abstractmethod
import time
from typing import Dict, Any
from omniboxes.node.logging_utils import default_logger
from fastapi.responses import JSONResponse
import asyncio
from enum import IntEnum
from io import BytesIO
import uuid


class Status(IntEnum):
    INVALID = 1
    STOPPING = 2
    STOPPED = 3,
    STARTING = 4,
    STARTED = 5,
    READY = 6,


class InstanceBase(ABC):
    def __init__(self, instance_num = 0, logger = None):
        self.instance_num = instance_num
        self.logger = logger or default_logger()
        self.status = Status.STOPPED
        self.reset_timeout = 5
        self.id = None

    def desc(self) -> str:
        return f'{self.__class__.__name__}:{self.instance_num}'

    async def create(self) -> None:
        self.logger.info(f"Creating instance {self.desc()}")
        self.status = Status.STARTING
        try:
            await self._create()
            self.status = Status.STARTED
            self.logger.info(f"Instance {self.desc()} created successfully")
            self.id = str(uuid.uuid4())
        except Exception as e:
            self.logger.error(f"Error creating instance {self.desc()}: {str(e)}")
            self.status = Status.INVALID
            raise e

    @abstractmethod
    async def _create(self) -> None:
        raise NotImplementedError("Subclasses should implement this!")

    async def delete(self) -> None:
        self.id = None
        self.logger.info(f"Deleting instance {self.desc()}")
        self.status = Status.STOPPING
        try:
            await self._delete()
            self.status = Status.STOPPED
            self.logger.info(f"Instance {self.desc()} deleted successfully")
        except Exception as e:
            self.logger.error(f"Error deleting instance {self.desc()}: {str(e)}")
            self.status = Status.INVALID
            raise e

    @abstractmethod
    async def _delete(self) -> None:
        raise NotImplementedError("Subclasses should implement this!")

    async def reset(self) -> None:
        await self.delete()
        await self.create()

    async def execute(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info(f"Executing command on instance {self.desc()}")
        try:
            result = await self._execute(command_data)
            self.logger.info(f"Command executed successfully on instance {self.desc()}")
            return result
        except Exception as e:
            self.logger.error(f"Error executing command on instance {self.desc()}: {str(e)}")
            raise e

    @abstractmethod
    async def _execute(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses should implement this!")

    async def screenshot(self, interaction_mode: str = "set_of_marks") -> BytesIO:
        self.logger.info(f"Taking screenshot of instance {self.desc()} with mode {interaction_mode}")
        try:
            result = await self._screenshot(interaction_mode=interaction_mode)
            self.logger.info(f"Screenshot taken successfully for instance {self.desc()}")
            return result
        except Exception as e:
            self.logger.error(f"Error taking screenshot of instance {self.desc()}: {str(e)}")
            raise e
        
    @abstractmethod   
    async def _screenshot(self, interaction_mode: str = "set_of_marks") -> BytesIO:
        raise NotImplementedError("Subclasses should implement this!")

    async def probe(self) -> bool:
        self.logger.info(f"Probing instance {self.desc()}")
        try:
            result = await self._probe()
            self.logger.info(f"Probe is done for instance {self.desc()}: {result}")
            return JSONResponse(content = {"status": "success", "message": "Probe successful"}, status_code=200) if result else JSONResponse(content = {"status": "error", "message": "Probe failed"}, status_code=500)
        except Exception as e:
            self.logger.error(f"Error probing instance {self.desc()}: {str(e)}")
            return JSONResponse(content = {"detail": str(e)}, status_code=500)
        
    @abstractmethod    
    async def _probe(self) -> bool:
        raise NotImplementedError("Subclasses should implement this!")
    
    async def reset_with_callback(self, callback):
        try:
            await self.reset()
            while not await self.probe():
                await asyncio.sleep(self.reset_timeout)
            callback(self)
        except Exception as e:
            self.logger.error(f"Error in reset worker: {str(e)}")

    async def metadata(self) -> Dict[str, Any]:
        self.logger.info(f"Getting metadata of instance {self.desc()}")
        try:
            result = await self._metadata()
            self.logger.info(f"Metadata for instance {self.desc()} is received")
            return result
        except Exception as e:
            self.logger.error(f"Error getting metadata of instance {self.desc()}: {str(e)}")
            raise e
        
    @abstractmethod   
    async def _metadata(self) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses should implement this!")

