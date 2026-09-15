from goti.batching.dynamic import DynamicScheduler, Request


def test_dynamic_admit_respects_max_batch():
    sched = DynamicScheduler(max_batch=2)
    for i in range(5):
        sched.enqueue(Request(str(i), [1, 2, 3], max_new_tokens=8))
    admitted = sched.admit()
    assert len(admitted) == 2
    assert len(sched.active) == 2
    assert len(sched.waiting) == 3
