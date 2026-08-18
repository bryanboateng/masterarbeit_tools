# /// script
# requires-python = ">=3.13"
# dependencies = ["wandb"]
# ///
"""Builds one HTML page that shows every evaluation result as a linked table.

This script stands on its own. It reads nothing but the run URLs beside it and
the Weights & Biases API, so it does not belong to the experiment code:

    uv run create_results_page.py
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path

import wandb
from wandb.apis.public import Run

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(name=__name__)

RUNS_FILE_PATH = Path(__file__).parent / "runs.txt"
OUTPUT_FILE_PATH = Path(__file__).parent / "results.html"

EVALUATION_STAGE_NAME = "4_evaluation"


@dataclass(frozen=True)
class Method:
    key: str
    # The two heading rows: the augmentation groups the columns, the policy
    # names the single column. A method that is neither, such as GPI, leaves
    # the policy empty and takes both rows.
    augmentation: str
    policy: str


# Also fixes the column order of every table.
METHODS = (
    Method(key="bc", augmentation="no augmentation", policy="BC"),
    Method(key="dp", augmentation="no augmentation", policy="DP"),
    Method(key="ccil", augmentation="CCIL", policy="BC"),
    Method(key="gpi", augmentation="GPI", policy=""),
    Method(key="tacil-bc", augmentation="TaCIL", policy="BC"),
    Method(key="tacil-dp", augmentation="TaCIL", policy="DP"),
)

DATASET_PERCENTAGES = (10, 25, 50, 100)

CONDITION_LABELS = tuple(
    f"{floor_condition}_{disturbance_label}"
    for floor_condition in ("grippy", "slippery")
    for disturbance_label in ("undisturbed", "low_disturbance", "high_disturbance")
)

_RUN_URL_PATTERN = re.compile(
    r"^https?://wandb\.ai/(?P<entity>[^/]+)/(?P<project>[^/]+)/runs/(?P<run_id>[^/?#]+)"
)

_DATASET_ARTIFACT_PATTERN = re.compile(r"dataset-(?P<percentage>\d+)")

_DARK_INK = "#0b0b0b"
_LIGHT_INK = "#ffffff"

# A sequential blue ramp, from the step nearest the page to the darkest one.
# A cell color is read off this ramp at the exact reward, not at a rounded
# step. The reward runs from 0 to 1, so the scale is fixed to that range rather
# than to the values on the page: a cell keeps its color when a run is
# replaced, and the tables stay comparable. In dark mode the ramp is reversed,
# so a reward of 0 recedes into the dark page instead of glowing on it.
_REWARD_RAMP = (
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)


@dataclass(frozen=True)
class Cell:
    condition_label: str
    dataset_percentage: int
    method_key: str


@dataclass(frozen=True)
class Result:
    url: str
    run_name: str
    reward_mean: float
    reward_standard_deviation: float


def main() -> None:
    urls = _read_urls(file_path=RUNS_FILE_PATH)
    logger.info("Read %d run URLs from %s", len(urls), RUNS_FILE_PATH)

    api = wandb.Api()
    results: dict[Cell, Result] = {}
    for url in urls:
        run = api.run(path=_run_path(url=url))
        cell = _locate(run=run)
        already_placed = results.get(cell)
        if already_placed is not None:
            raise ValueError(
                f"Two runs fall into the same cell {cell}: "
                f"{already_placed.url} and {url}"
            )
        results[cell] = _read_result(run=run, url=url)
        logger.info("%s -> %s", url, cell)

    _log_missing_cells(results=results)

    OUTPUT_FILE_PATH.write_text(_render_page(results=results))
    logger.info("Wrote %s", OUTPUT_FILE_PATH)


def _read_urls(*, file_path: Path) -> list[str]:
    urls = []
    for line in file_path.read_text().splitlines():
        stripped_line = line.split("#")[0].strip()
        if stripped_line:
            urls.append(stripped_line)
    return urls


def _run_path(*, url: str) -> str:
    match = _RUN_URL_PATTERN.match(url)
    if match is None:
        raise ValueError(f"Not a Weights & Biases run URL: {url}")
    return f"{match['entity']}/{match['project']}/{match['run_id']}"


def _locate(*, run: Run) -> Cell:
    return Cell(
        condition_label=_read_condition_label(run=run),
        dataset_percentage=_read_dataset_percentage(run=run),
        method_key=_read_method_key(run=run),
    )


def _read_condition_label(*, run: Run) -> str:
    floor_condition = run.config["floor_condition"].lower()
    disturbance_level = run.config["disturbance_level"]
    if disturbance_level is None:
        return f"{floor_condition}_undisturbed"
    return f"{floor_condition}_{disturbance_level.lower()}_disturbance"


def _read_dataset_percentage(*, run: Run) -> int:
    artifact_name = run.config["expert_dataset_artifact"]
    match = _DATASET_ARTIFACT_PATTERN.search(artifact_name)
    if match is None:
        raise ValueError(
            f"Run {run.url} trained on {artifact_name}, which carries no dataset "
            "percentage."
        )
    return int(match["percentage"])


def _read_method_key(*, run: Run) -> str:
    if run.job_type == "gpi":
        return "gpi"
    if run.job_type not in ("bc", "dp"):
        raise ValueError(f"Run {run.url} has unknown job type {run.job_type}.")
    augmentation = run.config["augmentation"]
    if augmentation is None:
        return run.job_type
    # The two augmentations are told apart by the stage they configure: CCIL
    # relabels transitions, TaCIL creates new data.
    if "labels" in augmentation:
        return "ccil"
    if "data" in augmentation:
        return f"tacil-{run.job_type}"
    raise ValueError(f"Run {run.url} has an unknown augmentation.")


def _read_result(*, run: Run, url: str) -> Result:
    key_prefix = f"{EVALUATION_STAGE_NAME}/{_read_condition_label(run=run)}"
    try:
        reward_mean = run.summary[f"{key_prefix}/reward_mean"]
        reward_standard_deviation = run.summary[f"{key_prefix}/reward_std"]
    except KeyError as error:
        raise ValueError(f"Run {url} carries no evaluation summary.") from error
    return Result(
        url=url,
        run_name=run.name,
        reward_mean=reward_mean,
        reward_standard_deviation=reward_standard_deviation,
    )


def _log_missing_cells(*, results: dict[Cell, Result]) -> None:
    cell_count = len(CONDITION_LABELS) * len(DATASET_PERCENTAGES) * len(METHODS)
    logger.info(
        "Filled %d of %d cells, %d still missing.",
        len(results),
        cell_count,
        cell_count - len(results),
    )


def _render_page(*, results: dict[Cell, Result]) -> str:
    tables = "\n".join(
        _render_table(condition_label=condition_label, results=results)
        for condition_label in CONDITION_LABELS
    )
    generation_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Results</title>
<style>{_STYLE}</style>
</head>
<body>
<h1>Results</h1>
<p class="subtitle">Evaluation reward, mean over 100 episodes with its standard
deviation. Every cell links to its run. Generated {generation_time}.</p>
{_render_legend()}
{tables}
</body>
</html>
"""


def _render_legend() -> str:
    return (
        '<div class="legend"><span>reward 0</span>'
        '<span class="scale"></span>'
        "<span>1</span></div>"
    )


def _render_table(*, condition_label: str, results: dict[Cell, Result]) -> str:
    rows = "\n".join(
        _render_row(
            condition_label=condition_label,
            dataset_percentage=dataset_percentage,
            results=results,
        )
        for dataset_percentage in DATASET_PERCENTAGES
    )
    return f"""<h2>{condition_label.replace("_", " ")}</h2>
<div class="table-wrapper">
<table>
<colgroup><col class="label-column"><col span="{len(METHODS)}"></colgroup>
{_render_head()}
<tbody>
{rows}
</tbody>
</table>
</div>"""


def _render_head() -> str:
    augmentation_headings = ""
    policy_headings = ""
    for augmentation, methods in _group_by_augmentation().items():
        if any(method.policy for method in methods):
            augmentation_headings += f'<th colspan="{len(methods)}">{augmentation}</th>'
            policy_headings += "".join(
                f"<th>{method.policy}</th>" for method in methods
            )
        else:
            augmentation_headings += f'<th rowspan="2">{augmentation}</th>'
    return f"""<thead>
<tr><th rowspan="2">Data</th>{augmentation_headings}</tr>
<tr>{policy_headings}</tr>
</thead>"""


def _group_by_augmentation() -> dict[str, list[Method]]:
    groups: dict[str, list[Method]] = {}
    for method in METHODS:
        groups.setdefault(method.augmentation, []).append(method)
    return groups


def _render_row(
    *, condition_label: str, dataset_percentage: int, results: dict[Cell, Result]
) -> str:
    cells = "".join(
        _render_cell(
            result=results.get(
                Cell(
                    condition_label=condition_label,
                    dataset_percentage=dataset_percentage,
                    method_key=method.key,
                )
            )
        )
        for method in METHODS
    )
    return f"<tr><th>{dataset_percentage} %</th>{cells}</tr>"


def _render_cell(*, result: Result | None) -> str:
    if result is None:
        return '<td class="empty"></td>'
    fill = _reward_color(reward_mean=result.reward_mean, ramp=_REWARD_RAMP)
    dark_fill = _reward_color(
        reward_mean=result.reward_mean, ramp=tuple(reversed(_REWARD_RAMP))
    )
    style = (
        f"--fill:{fill};--ink:{_ink(fill=fill)};"
        f"--dark-fill:{dark_fill};--dark-ink:{_ink(fill=dark_fill)}"
    )
    return (
        f'<td class="reward" style="{style}">'
        f'<a href="{escape(result.url)}" title="{escape(result.run_name)}">'
        f'<span class="mean">{result.reward_mean:.3f}</span>'
        f'<span class="deviation">±{result.reward_standard_deviation:.3f}</span>'
        f"</a></td>"
    )


def _reward_color(*, reward_mean: float, ramp: tuple[str, ...]) -> str:
    position = min(max(reward_mean, 0.0), 1.0) * (len(ramp) - 1)
    lower_index = min(int(position), len(ramp) - 2)
    return _to_hex(
        color=tuple(
            round(lower_value + (upper_value - lower_value) * (position - lower_index))
            for lower_value, upper_value in zip(
                _to_channels(hex_color=ramp[lower_index]),
                _to_channels(hex_color=ramp[lower_index + 1]),
            )
        )
    )


def _to_channels(*, hex_color: str) -> tuple[int, ...]:
    return tuple(int(hex_color[index : index + 2], 16) for index in (1, 3, 5))


def _to_hex(*, color: tuple[int, ...]) -> str:
    return "#" + "".join(f"{value:02x}" for value in color)


def _ink(*, fill: str) -> str:
    """Picks the ink that reads on the fill.

    Both inks reach at least 4.4:1 anywhere on the ramp, and over 5:1 on all
    but its middle. A single hue that spans the whole lightness range has that
    middle, where neither black nor white can do better.
    """
    if _contrast(first_color=fill, second_color=_DARK_INK) >= _contrast(
        first_color=fill, second_color=_LIGHT_INK
    ):
        return _DARK_INK
    return _LIGHT_INK


def _contrast(*, first_color: str, second_color: str) -> float:
    luminances = sorted(
        (
            _relative_luminance(hex_color=first_color),
            _relative_luminance(hex_color=second_color),
        ),
        reverse=True,
    )
    return (luminances[0] + 0.05) / (luminances[1] + 0.05)


def _relative_luminance(*, hex_color: str) -> float:
    linear_channels = []
    for value in _to_channels(hex_color=hex_color):
        channel = value / 255
        linear_channels.append(
            channel / 12.92
            if channel <= 0.04045
            else ((channel + 0.055) / 1.055) ** 2.4
        )
    return (
        0.2126 * linear_channels[0]
        + 0.7152 * linear_channels[1]
        + 0.0722 * linear_channels[2]
    )


_STYLE = f"""
:root {{
  color-scheme: light dark;
  --background: #ffffff;
  --surface: #f6f7f9;
  --border: #d9dce1;
  --text: #14161a;
  --muted: #6b7280;
  --scale: linear-gradient(to right, {", ".join(_REWARD_RAMP)});
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --background: #16181d;
    --surface: #1e2128;
    --border: #333842;
    --text: #e8eaee;
    --muted: #9aa1ac;
    --scale: linear-gradient(to right, {", ".join(reversed(_REWARD_RAMP))});
  }}
}}
body {{
  margin: 0 auto;
  padding: 2rem 1.5rem 3rem;
  max-width: 58rem;
  background: var(--background);
  color: var(--text);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  line-height: 1.4;
}}
h1 {{ font-size: 1.5rem; margin: 0 0 0.25rem; }}
h2 {{ font-size: 1rem; margin: 1.6rem 0 0.4rem; font-weight: 600; }}
.subtitle {{ color: var(--muted); margin: 0; font-size: 0.85rem; }}
.legend {{
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 0.9rem;
  color: var(--muted);
  font-size: 0.78rem;
}}
.legend .scale {{ width: 18rem; height: 0.6rem; background: var(--scale); }}
.table-wrapper {{ overflow-x: auto; }}
table {{
  border-collapse: collapse;
  table-layout: fixed;
  width: 100%;
  min-width: 34rem;
  font-size: 0.82rem;
}}
th, td {{ border: 1px solid var(--border); padding: 0; text-align: center; }}
thead th, tbody th {{
  background: var(--surface);
  color: var(--text);
  padding: 0.25rem 0.5rem;
  font-weight: 600;
  white-space: nowrap;
}}
col.label-column {{ width: 4rem; }}
tbody th {{ text-align: right; }}
/* One height for every cell, so the grid stays even wherever a run is
   missing. */
td {{ height: 1.9rem; }}
td.reward {{ background: var(--fill); color: var(--ink); }}
@media (prefers-color-scheme: dark) {{
  td.reward {{ background: var(--dark-fill); color: var(--dark-ink); }}
}}
td a {{
  display: flex;
  align-items: baseline;
  justify-content: center;
  gap: 0.3rem;
  height: 100%;
  color: inherit;
  text-decoration: none;
}}
td a:hover {{ outline: 2px solid var(--text); outline-offset: -2px; }}
td.empty {{ background: repeating-linear-gradient(
  45deg, transparent, transparent 5px, var(--surface) 5px, var(--surface) 10px); }}
.mean {{ font-variant-numeric: tabular-nums; }}
.deviation {{ font-size: 0.72rem; font-variant-numeric: tabular-nums; }}
"""


if __name__ == "__main__":
    main()
