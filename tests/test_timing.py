from goti.timing import benchmark_callable


def test_benchmark_callable_runs():
    n = {"i": 0}

    def fn():
        n["i"] += 1

    stats = benchmark_callable(fn, warmup=2, iters=3)
    assert stats.measured == 3
    assert len(stats.samples_ms) == 3
    assert n["i"] == 5
