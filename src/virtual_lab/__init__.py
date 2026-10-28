"""Virtual Lab package."""

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True), override=False)

from virtual_lab.__about__ import __version__
from virtual_lab.agent import Agent
from virtual_lab.run_meeting import run_meeting


__all__ = [
    "__version__",
    "Agent",
    "run_meeting",
]
