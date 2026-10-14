"""A node's reported time is its own work: time its call spends waiting for other work
(an item ahead holding a shared session, a look-ahead at a later item) is left out."""

import threading

import util


class Clock:
    """perf_counter readings, in the order the code under test asks for them."""

    def __init__(self, *readings: float) -> None:
        self.readings = list(readings)

    def perf_counter(self) -> float:
        return self.readings.pop(0)


def test_time_spent_waiting_for_other_work_is_left_out(monkeypatch):
    # The call starts at 0 and ends at 8; it waits from 1 to 6.
    monkeypatch.setattr(util, "time", Clock(0.0, 1.0, 6.0, 8.0))

    def node() -> str:
        with util.waiting():
            pass
        return "output"

    assert util.timed_supplier(node)() == ("output", 3.0)


def test_a_wait_on_another_thread_is_not_this_calls(monkeypatch):
    # The other thread's wait (2 to 7) is its own; this call keeps its 0 to 9.
    monkeypatch.setattr(util, "time", Clock(0.0, 2.0, 7.0, 9.0))

    def other() -> None:
        with util.waiting():
            pass

    def node() -> None:
        thread = threading.Thread(target=other)
        thread.start()
        thread.join()

    assert util.timed_supplier(node)() == (None, 9.0)
