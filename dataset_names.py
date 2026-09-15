"""Turns a dataset percentage into the artifact name the run scripts expect."""

import re

ENVIRONMENT = "PushT"

SOURCE_EPISODE_COUNT = 206

# The subsamples were drawn by percentage, then named by the episode count
# that came out of it. 25% of 206 episodes is 51, not 52.
EPISODE_COUNTS: dict[int, int] = {5: 10, 10: 20, 25: 51, 50: 103, 100: 206}

_PERCENTAGES = {count: percentage for percentage, count in EPISODE_COUNTS.items()}

_ARTIFACT_NAME_PATTERN = re.compile(
    rf"^dataset-{ENVIRONMENT}-(?P<episode_count>\d+)-of-{SOURCE_EPISODE_COUNT}$"
)


def build_artifact_name(*, dataset_percentage: int) -> str:
    if dataset_percentage not in EPISODE_COUNTS:
        raise ValueError(
            f"No dataset exists for {dataset_percentage}%. "
            f"Known percentages: {sorted(EPISODE_COUNTS)}."
        )

    return (
        f"dataset-{ENVIRONMENT}-{EPISODE_COUNTS[dataset_percentage]}"
        f"-of-{SOURCE_EPISODE_COUNT}"
    )


def read_dataset_percentage(*, artifact_name: str) -> int | None:
    """Reads the percentage back off a run's artifact name."""
    match = _ARTIFACT_NAME_PATTERN.match(artifact_name.split(":", maxsplit=1)[0])
    if match is None:
        return None

    return _PERCENTAGES.get(int(match.group("episode_count")))
