import sys

from switchyard.harness.e4b import capture_run


def test_the_live_runner_returns_the_child_stdout():
    # test_fake_hides_platform: the fakes returned stdout, the live default (bare subprocess.run) returned None.
    completed = capture_run([sys.executable, "-c", "print('pin-report')"])
    assert completed.stdout.strip() == "pin-report"
