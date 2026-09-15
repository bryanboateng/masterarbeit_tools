"""Reads what a dataset artifact's name says about it."""

import re

_NAME_PATTERN = re.compile(
    r"^dataset-(?P<environment>[A-Za-z0-9]+)"
    r"-(?P<episode_count>\d+)-of-(?P<source_episode_count>\d+)$"
)


def read_episode_count(*, artifact_name: str) -> int | None:
    """The number of demonstrations the dataset holds, for ordering a table."""
    match = _NAME_PATTERN.match(artifact_name.split(":", maxsplit=1)[0])
    return None if match is None else int(match.group("episode_count"))
