"""Equal state payloads on separate runs retain their control boundaries."""

import asyncio

from nodes.impl.event_queue import LatestEventQueue


def test_equal_payload_before_barrier_is_not_removed_by_later_coalescing():
    async def run():
        queue = LatestEventQueue()
        first = {"event": "node-start", "data": {"nodeId": "same"}}
        barrier = {"event": "chain-start", "data": {"nodes": ["same"]}}
        second = {"event": "node-start", "data": {"nodeId": "same"}}
        third = {"event": "node-start", "data": {"nodeId": "same"}}
        for item in (first, barrier, second, third):
            queue.put_nowait(item)
        assert queue.qsize() == 3
        for expected in (first, barrier, third):
            assert await queue.get() is expected
            queue.task_done()
        await asyncio.wait_for(queue.join(), 1)

    asyncio.run(run())
