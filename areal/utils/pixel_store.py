"""Ray actor for cross-node pixel value sharing.

Stores pixel tensors **in actor memory**.  Training workers call
``get_slice(keys)`` to fetch only the images they need.  The broker
returns tensors directly in the RPC response — Ray serializes and
transfers them to the caller.  Multiple workers call ``get_slice``
concurrently (``max_concurrency``), so cross-node transfers happen
in parallel.

Usage::

    # At startup (controller process):
    from areal.utils.pixel_store import get_or_create_pixel_store
    pixel_store = get_or_create_pixel_store()

    # Rollout worker:
    ref = ray.put(pixel_values)
    ray.get(pixel_store.put.remote(key, ref))
    # Ray auto-resolves ref → actor receives tensor → stored in memory

    # Training worker (each worker requests only its needed keys):
    fetched = ray.get(pixel_store.get_slice.remote(my_keys))
    # fetched is {key: tensor} — only the images this worker needs

    # Cleanup:
    ray.get(pixel_store.free.remote(keys))
"""

from typing import Any

import ray

from areal.utils import logging

logger = logging.getLogger("PixelStore")

PIXEL_STORE_NAME = "pixel_ref_store"
PIXEL_STORE_NAMESPACE = "areal"


@ray.remote(num_cpus=0, max_concurrency=16, max_restarts=-1, max_task_retries=3)
class PixelRefStore:
    """Named Ray actor that stores pixel tensors in actor memory.

    ``max_restarts=-1`` makes Ray auto-revive the actor on crash. The
    in-memory dict is lost on restart so trajectories that already put
    pixels into the dead instance cannot be recovered, but every new
    rollout can resume normally instead of permanently dead-locking
    the train rollout pipeline (this happened on 2026-04-21 02:36 when
    a transient worker-overlay disk-full spike killed the actor and
    every subsequent put_batch saw "Socket closed").

    Rollout workers pass ObjectRefs as arguments — Ray auto-resolves
    them to tensors before the actor receives them.  Tensors are kept
    in a plain Python dict (no ``ray.put()`` per image).

    Training workers call ``get_slice(keys)`` which returns a dict of
    only the requested tensors.  Ray serializes the return value and
    sends it directly to the caller.  ``max_concurrency=16`` allows
    multiple workers to fetch their slices in parallel.
    """

    def __init__(self):
        self._data: dict[str, Any] = {}  # key → raw tensor (in actor memory)
        self._count = 0

    def put(self, key: str, data: Any) -> None:
        """Store tensor in actor memory under *key*.

        Ray auto-resolves ObjectRef args to tensors before delivery.
        """
        if isinstance(data, ray.ObjectRef):
            data = ray.get(data)
        self._data[key] = data
        self._count += 1

    def put_batch(self, keys: list[str], data_list: list[Any]) -> None:
        """Store multiple items at once (fewer RPCs)."""
        for k, d in zip(keys, data_list):
            if isinstance(d, ray.ObjectRef):
                d = ray.get(d)
            self._data[k] = d
        self._count += len(keys)

    def count_missing(self, keys: list[str]) -> int:
        """Return how many of the given keys are NOT currently stored.

        Cheap pre-check used by the trainer before ppo_update: if the
        actor was restarted (e.g. after an OutOfDiskError crash) the
        in-memory dict is empty and any subsequent get_slice would
        return empty values, leading to KeyError inside FSDP forward
        and an NCCL collective desync. The trainer can use this to
        drop the entire batch deterministically on the controller
        before dispatching FSDP work.
        """
        return sum(1 for k in keys if k not in self._data)

    def get_slice(self, keys: list[str]) -> dict[str, Any]:
        """Return tensors for only the requested keys.

        Each worker calls this with the keys it needs.  Multiple workers
        call concurrently — Ray serializes each response independently
        and transfers only the needed data to each caller.

        Same-node workers: fast local serialization.
        Cross-node workers: only their slice is transferred (not all images).
        """
        return {k: self._data[k] for k in keys if k in self._data}

    def free(self, keys: list[str]) -> int:
        """Remove keys and return count of freed refs."""
        freed = 0
        for k in keys:
            if self._data.pop(k, None) is not None:
                freed += 1
        return freed

    def stats(self) -> dict:
        """Return store statistics."""
        return {"stored": len(self._data), "total_puts": self._count}


def get_or_create_pixel_store() -> ray.actor.ActorHandle:
    """Get the named PixelRefStore actor, creating it if needed."""
    try:
        return ray.get_actor(PIXEL_STORE_NAME, namespace=PIXEL_STORE_NAMESPACE)
    except ValueError:
        return PixelRefStore.options(
            name=PIXEL_STORE_NAME,
            namespace=PIXEL_STORE_NAMESPACE,
            lifetime="detached",
        ).remote()
