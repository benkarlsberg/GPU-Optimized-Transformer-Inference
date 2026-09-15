from goti.env import collect_environment


def test_collect_environment_keys():
    info = collect_environment()
    assert "platform" in info
    assert "python" in info
    assert "torch" in info
