import os

from vla_sim_poc.manifest import redact_home


def test_redact_home_replaces_only_the_home_prefix():
    home = os.path.expanduser("~")
    assert redact_home(os.path.join(home, ".cache", "x")) == os.path.join("~", ".cache", "x")
    assert redact_home(home) == "~"
    assert redact_home("/opt/data") == "/opt/data"
    assert redact_home(home + "extra/x") == home + "extra/x"  # sibling dir sharing the prefix
