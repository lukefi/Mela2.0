import os


def pytest_addoption(parser):
    parser.addoption(
        "--update-snapshots",
        action="store_true",
        default=False,
        help="Update snapshot files instead of comparing them.",
    )


def pytest_configure(config):
    if config.getoption("--update-snapshots"):
        os.environ["UPDATE_MELA_SNAPSHOTS"] = "1"
    else:
        os.environ["UPDATE_MELA_SNAPSHOTS"] = "0"
