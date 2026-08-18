"""Builds one HTML page that shows every evaluation result as a linked table.

The page is built from `results.txt` alone, which is written by hand:

    uv run create_results_page.py
"""

import logging
from dataclasses import dataclass
from html import escape
from pathlib import Path

from matplotlib import colormaps
from matplotlib.colors import to_hex

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(name=__name__)

RESULTS_FILE_PATH = Path(__file__).parent / "results.txt"
OUTPUT_FILE_PATH = Path(__file__).parent / "results.html"


@dataclass(frozen=True)
class Method:
    key: str
    # The two heading rows: the policy groups the columns, the augmentation
    # names the single column. A method that is neither, such as GPI, leaves
    # the augmentation empty and takes both rows.
    policy: str
    augmentation: str


# Also fixes the column order of every table.
METHODS = (
    Method(key="bc", policy="BC", augmentation="none"),
    Method(key="ccil", policy="BC", augmentation="CCIL"),
    Method(key="tacil-bc", policy="BC", augmentation="TaCIL"),
    Method(key="gpi", policy="GPI", augmentation=""),
    Method(key="dp", policy="DP", augmentation="none"),
    Method(key="tacil-dp", policy="DP", augmentation="TaCIL"),
)

DATASET_PERCENTAGES = (10, 25, 50, 100)

CONDITION_LABELS = tuple(
    f"{floor_condition}_{disturbance_label}"
    for floor_condition in ("grippy", "slippery")
    for disturbance_label in ("undisturbed", "low_disturbance", "high_disturbance")
)

_DARK_INK = "#0b0b0b"
_LIGHT_INK = "#ffffff"

# A cell color comes straight out of the Blues colormap, asked at the reward
# itself, in both page themes. The reward runs from 0 to 1, which is also the
# range of the colormap, so the scale is fixed rather than fitted to the values
# on the page: a cell keeps its color when a run is replaced, and the tables
# stay comparable.
_REWARD_COLORMAP = colormaps["Blues"]


@dataclass(frozen=True)
class Cell:
    condition_label: str
    dataset_percentage: int
    method_key: str


@dataclass(frozen=True)
class Result:
    url: str
    reward_mean: float
    reward_standard_deviation: float


def main() -> None:
    results = _read_results(file_path=RESULTS_FILE_PATH)
    cell_count = len(CONDITION_LABELS) * len(DATASET_PERCENTAGES) * len(METHODS)
    logger.info(
        "Read %d of %d cells from %s, %d still missing.",
        len(results),
        cell_count,
        RESULTS_FILE_PATH,
        cell_count - len(results),
    )

    OUTPUT_FILE_PATH.write_text(_render_page(results=results))
    logger.info("Wrote %s", OUTPUT_FILE_PATH)


def _read_results(*, file_path: Path) -> dict[Cell, Result]:
    """Reads the hand-written table.

    A line in brackets opens a condition, every line below it holds one cell:
    the dataset percentage, the method, the reward mean, its standard deviation
    and the run URL.
    """
    results: dict[Cell, Result] = {}
    condition_label = None
    for line_number, line in enumerate(file_path.read_text().splitlines(), start=1):
        location = f"{file_path}:{line_number}"
        content = line.split("#")[0].strip()
        if not content:
            continue
        if content.startswith("[") and content.endswith("]"):
            condition_label = _read_condition_label(
                text=content[1:-1].strip(), location=location
            )
            continue
        if condition_label is None:
            raise ValueError(f"{location}: no condition opened above this line.")
        cell, result = _read_cell(
            content=content, condition_label=condition_label, location=location
        )
        already_read = results.get(cell)
        if already_read is not None:
            raise ValueError(
                f"{location}: this cell is already filled by {already_read.url}."
            )
        results[cell] = result
    return results


def _read_condition_label(*, text: str, location: str) -> str:
    if text not in CONDITION_LABELS:
        raise ValueError(
            f"{location}: unknown condition {text}. "
            f"Known are {', '.join(CONDITION_LABELS)}."
        )
    return text


def _read_cell(
    *, content: str, condition_label: str, location: str
) -> tuple[Cell, Result]:
    fields = content.split()
    if len(fields) != 4 + 1:
        raise ValueError(
            f"{location}: expected 5 fields "
            "(percentage, method, mean, standard deviation, URL), "
            f"got {len(fields)}."
        )
    percentage_text, method_key, mean_text, standard_deviation_text, url = fields

    if not percentage_text.isdigit() or int(percentage_text) not in DATASET_PERCENTAGES:
        raise ValueError(
            f"{location}: unknown dataset percentage {percentage_text}. Known are "
            f"{', '.join(str(percentage) for percentage in DATASET_PERCENTAGES)}."
        )
    if method_key not in {method.key for method in METHODS}:
        raise ValueError(
            f"{location}: unknown method {method_key}. Known are "
            f"{', '.join(method.key for method in METHODS)}."
        )

    return (
        Cell(
            condition_label=condition_label,
            dataset_percentage=int(percentage_text),
            method_key=method_key,
        ),
        Result(
            url=url,
            reward_mean=_read_number(text=mean_text, location=location),
            reward_standard_deviation=_read_number(
                text=standard_deviation_text, location=location
            ),
        ),
    )


def _read_number(*, text: str, location: str) -> float:
    try:
        return float(text)
    except ValueError as error:
        raise ValueError(f"{location}: {text} is not a number.") from error


def _render_page(*, results: dict[Cell, Result]) -> str:
    tables = "\n".join(
        _render_table(condition_label=condition_label, results=results)
        for condition_label in CONDITION_LABELS
    )
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
{tables}
</body>
</html>
"""


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
    policy_headings = ""
    augmentation_headings = ""
    for policy, methods in _group_by_policy().items():
        if any(method.augmentation for method in methods):
            policy_headings += f'<th colspan="{len(methods)}">{policy}</th>'
            augmentation_headings += "".join(
                f"<th>{method.augmentation}</th>" for method in methods
            )
        else:
            policy_headings += f'<th rowspan="2">{policy}</th>'
    return f"""<thead>
<tr><th rowspan="2">Data</th>{policy_headings}</tr>
<tr>{augmentation_headings}</tr>
</thead>"""


def _group_by_policy() -> dict[str, list[Method]]:
    groups: dict[str, list[Method]] = {}
    for method in METHODS:
        groups.setdefault(method.policy, []).append(method)
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
    fill = _reward_color(reward_mean=result.reward_mean)
    return (
        f'<td class="reward" style="--fill:{fill};--ink:{_ink(fill=fill)}">'
        f'<a href="{escape(result.url)}" target="_blank" rel="noopener">'
        f'<span class="mean">{result.reward_mean:.3f}</span>'
        f'<span class="deviation">±{result.reward_standard_deviation:.3f}</span>'
        f"</a></td>"
    )


def _reward_color(*, reward_mean: float) -> str:
    return to_hex(c=_REWARD_COLORMAP(min(max(reward_mean, 0.0), 1.0)))


def _to_channels(*, hex_color: str) -> tuple[int, ...]:
    return tuple(int(hex_color[index : index + 2], 16) for index in (1, 3, 5))


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


_STYLE = """
:root {
  color-scheme: light dark;
  --background: #ffffff;
  --surface: #f6f7f9;
  --border: #d9dce1;
  --text: #14161a;
  --muted: #6b7280;
}
@media (prefers-color-scheme: dark) {
  :root {
    --background: #16181d;
    --surface: #1e2128;
    --border: #333842;
    --text: #e8eaee;
    --muted: #9aa1ac;
  }
}
body {
  margin: 0 auto;
  padding: 2rem 1.5rem 3rem;
  max-width: 58rem;
  background: var(--background);
  color: var(--text);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  line-height: 1.4;
}
h1 { font-size: 1.5rem; margin: 0 0 0.25rem; }
h2 { font-size: 1rem; margin: 1.6rem 0 0.4rem; font-weight: 600; }
.table-wrapper { overflow-x: auto; }
table {
  border-collapse: collapse;
  table-layout: fixed;
  width: 100%;
  min-width: 34rem;
  font-size: 0.82rem;
}
th, td { border: 1px solid var(--border); padding: 0; text-align: center; }
thead th, tbody th {
  background: var(--surface);
  color: var(--text);
  padding: 0.25rem 0.5rem;
  font-weight: 600;
  white-space: nowrap;
}
col.label-column { width: 4rem; }
/* One height for every cell, so the grid stays even wherever a run is
   missing. */
td { height: 2.7rem; }
td.reward { background: var(--fill); color: var(--ink); }
td a {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  line-height: 1.15;
  color: inherit;
  text-decoration: none;
}
td a:hover { outline: 2px solid var(--text); outline-offset: -2px; }
td.empty { background: repeating-linear-gradient(
  45deg, transparent, transparent 5px, var(--surface) 5px, var(--surface) 10px); }
.mean { font-size: 0.92rem; font-weight: 600; font-variant-numeric: tabular-nums; }
.deviation { font-size: 0.76rem; font-variant-numeric: tabular-nums; }
"""


if __name__ == "__main__":
    main()
