"""skellyforge: the standard human, and the closed-form math that hydrates it."""

__package_name__ = "skellyforge"
__version__ = "v2024.12.1009"

__author__ = """Aaron Cherian"""
__email__ = "info@freemocap.org"
__repo_owner_github_user_name__ = "freemocap"
__repo_url__ = (
    f"https://github.com/{__repo_owner_github_user_name__}/{__package_name__}/"
)
__repo_issues_url__ = f"{__repo_url__}issues"

from beartype.claw import beartype_this_package

beartype_this_package()


# Dump a Python traceback on native crashes (segfault / Windows access violation, e.g. 0xC0000005)
# instead of dying silently. Spawned workers re-import this package, so they inherit this too.
# Near-zero steady-state overhead — handlers stay dormant until a fatal signal actually fires.
import faulthandler

faulthandler.enable()

# NOTE: temporary stand-in for the skellylogs websocket-backed logger, whose
# multiprocessing queue cannot be opened under the current sandbox. Restore the
# `skellylogs.configure_logging` call when the richer handlers are needed again.
import logging

logging.basicConfig(level=logging.INFO)
