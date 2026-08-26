"""Builds the augmentation-study results page straight from wandb.

Unlike `create_results_page.py` (which reads a hand-written table), this pulls
the final runs of the project below, groups them into (dataset %, method,
config), and keeps each method's best config (chosen once by the validation
condition). One table per dataset size: methods are rows (split into the BC and
DP+GPI families), the six evaluation conditions are columns of reward mean ±
std, and two trailing columns give the augmentation size (invented transitions
as a percent of the expert data) and the realised chain length. Grippy columns
use a blue scale, slippery ones an orange scale.

    uv run create_augmentation_results_page.py
"""

import logging
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path

import wandb
from matplotlib import colormaps
from matplotlib.colors import to_hex

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(name=__name__)

ENTITY = "bryanboateng-team"
PROJECT = "2026-08-24-final"
OUTPUT_FILE_PATH = Path(__file__).parent / "augmentation.html"

STAGE = "4_evaluation"
SELECTION_CONDITION = "grippy_low_disturbance"
SYNTHETIC_LENGTH = "2_data/4_synthetic_length"


@dataclass(frozen=True)
class Method:
    tag: str
    family: str  # "BC" or "DP+GPI"
    label: str
    param_keys: tuple[str, ...]  # config keys that identify one config


_TACIL_BOUNDED = (
    "augmentation.dynamics.window_length",
    "augmentation.data.action_noise_range_fraction",
    "augmentation.data.replay_error_limit_step_multiple",
    "augmentation.dynamics.lipschitz_constant_upper_bound",
)
_TACIL_UNBOUNDED = _TACIL_BOUNDED[:-1]
_CCIL = (
    "augmentation.dynamics.lipschitz_type",
    "augmentation.dynamics.lipschitz_constraint",
    "augmentation.labels.rejection_quantile",
    "augmentation.labels.type",
    "augmentation.labels.action_noise_std",
)
_GPI = ("policy.progression_weight", "policy.attraction_weight", "policy.plan_length")

METHODS = (
    Method("bc-noaug", "BC", "BC, none", ()),
    Method("ccil-bc", "BC", "BC, CCIL", _CCIL),
    Method("tacil-bc-bounded", "BC", "BC, TACIL bounded", _TACIL_BOUNDED),
    Method("tacil-bc-unbounded", "BC", "BC, TACIL unbounded", _TACIL_UNBOUNDED),
    Method("gpi", "DP+GPI", "GPI", _GPI),
    Method("dp-noaug", "DP+GPI", "DP, none", ()),
    Method("tacil-dp-bounded", "DP+GPI", "DP, TACIL bounded", _TACIL_BOUNDED),
    Method("tacil-dp-unbounded", "DP+GPI", "DP, TACIL unbounded", _TACIL_UNBOUNDED),
)
FAMILIES = ("BC", "DP+GPI")

CONDITION_LABELS = tuple(
    f"{floor}_{disturbance}"
    for floor in ("grippy", "slippery")
    for disturbance in ("undisturbed", "low_disturbance", "high_disturbance")
)
DISTURBANCE_HEADS = ("undisturbed", "low", "high")

_COLORMAPS = {"grippy": colormaps["Blues"], "slippery": colormaps["Oranges"]}
_DARK_INK = "#0b0b0b"
_LIGHT_INK = "#ffffff"


@dataclass(frozen=True)
class BestConfig:
    rewards: dict[str, tuple[float, float]]  # condition -> (mean, standard deviation)
    length: tuple[float, float, float] | None  # (p10, median, p90)
    # Invented (synthetic) transitions as a percent of the recorded expert
    # transitions -- the copied expert run-out is not counted.
    invented_percent: float | None
    url: str


def _invented_and_expert(summary) -> tuple[float | None, float | None]:
    """Invented and recorded-expert transition counts, whichever the augmentation
    logged (TACIL and CCIL use different keys)."""
    invented = summary.get("2_data/5_dataset/2_invented_transitions")
    expert = summary.get("2_data/5_dataset/1_recorded_expert_transitions")
    if invented is None or expert is None:
        invented = summary.get("2_data/3_dataset/synthetic_transition_count")
        expert = summary.get("2_data/1_labels/expert_transition_count")
    return invented, expert


def main() -> None:
    runs = list(wandb.Api().runs(f"{ENTITY}/{PROJECT}", per_page=500))
    logger.info("Read %d runs from %s/%s.", len(runs), ENTITY, PROJECT)

    percentages, best = _collect(runs=runs)
    OUTPUT_FILE_PATH.write_text(_render_page(percentages=percentages, best=best))
    logger.info("Wrote %s (%d dataset sizes).", OUTPUT_FILE_PATH, len(percentages))


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and the (sample) standard deviation across the seeds."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, math.sqrt(variance)


def _get(config: dict, dotted: str):
    if dotted in config:
        return config[dotted]
    current = config
    for part in dotted.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _collect(*, runs: list) -> tuple[list[int], dict[tuple[int, str], BestConfig]]:
    method_by_tag = {method.tag: method for method in METHODS}
    groups: dict = {}
    percentages: set[int] = set()

    for run in runs:
        if run.state != "finished":
            continue
        tag = next((t for t in run.tags if t in method_by_tag), None)
        pct_tag = next((t for t in run.tags if t.endswith("pct")), None)
        if tag is None or pct_tag is None:
            continue
        pct = int(pct_tag[:-3])
        percentages.add(pct)
        method = method_by_tag[tag]
        signature = tuple(_get(run.config, key) for key in method.param_keys)
        group = groups.setdefault(
            (pct, tag, signature),
            {
                "rewards": {c: [] for c in CONDITION_LABELS},
                "len": [],
                "aug": [],
                "url": run.url,
            },
        )
        for condition in CONDITION_LABELS:
            value = run.summary.get(f"{STAGE}/{condition}/reward_mean")
            if value is not None:
                group["rewards"][condition].append(value)
        triple = tuple(
            run.summary.get(f"{SYNTHETIC_LENGTH}/{q}") for q in ("p10", "median", "p90")
        )
        if all(v is not None for v in triple):
            group["len"].append(triple)
        invented, expert = _invented_and_expert(run.summary)
        if invented is not None and expert:
            group["aug"].append(100.0 * invented / expert)

    best: dict[tuple[int, str], BestConfig] = {}
    best_selection: dict[tuple[int, str], float] = {}
    for (pct, tag, _signature), group in groups.items():
        selection_values = group["rewards"][SELECTION_CONDITION]
        if not selection_values:
            continue
        mean = sum(selection_values) / len(selection_values)
        key = (pct, tag)
        if key in best_selection and mean <= best_selection[key]:
            continue
        best_selection[key] = mean
        length = None
        if group["len"]:
            columns = list(zip(*group["len"]))
            length = tuple(sum(c) / len(c) for c in columns)  # type: ignore[assignment]
        invented_percent = (
            sum(group["aug"]) / len(group["aug"]) if group["aug"] else None
        )
        best[key] = BestConfig(
            rewards={
                condition: _mean_std(values)
                for condition, values in group["rewards"].items()
                if values
            },
            length=length,
            invented_percent=invented_percent,
            url=group["url"],
        )
    return sorted(percentages), best


# ---- rendering -------------------------------------------------------------


def _render_page(
    *, percentages: list[int], best: dict[tuple[int, str], BestConfig]
) -> str:
    tables = "\n".join(
        _render_dataset_table(percentage=percentage, best=best)
        for percentage in percentages
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Augmentation results</title>
<style>{_STYLE}</style>
</head>
<body>
<h1>Augmentation results</h1>
<p class="note">Reward mean ± std over 3 seeds. In each condition column the best
method of each family (BC, DP+GPI) is bold. <b>synth</b> = invented transitions
as a percent of the expert data; <b>len</b> = chain length, median (p10–p90).</p>
{tables}
</body>
</html>
"""


def _render_dataset_table(
    *, percentage: int, best: dict[tuple[int, str], BestConfig]
) -> str:
    # For each condition, the winning method of each family (to make it bold).
    winners: dict[str, dict[str, str]] = {}
    for condition in CONDITION_LABELS:
        winners[condition] = {}
        for family in FAMILIES:
            winner, winning_value = None, -1.0
            for method in METHODS:
                if method.family != family:
                    continue
                config = best.get((percentage, method.tag))
                if config and condition in config.rewards:
                    value = config.rewards[condition][0]
                    if value > winning_value:
                        winning_value, winner = value, method.tag
            if winner is not None:
                winners[condition][family] = winner

    column_count = 1 + len(CONDITION_LABELS) + 2
    body = ""
    for family in FAMILIES:
        body += f'<tr class="group"><th colspan="{column_count}">{family}</th></tr>\n'
        for method in (m for m in METHODS if m.family == family):
            body += _render_method_row(
                percentage=percentage, method=method, best=best, winners=winners
            )
    return f"""<h2>{percentage} % of the expert data</h2>
<div class="table-wrapper">
<table>
{_render_head()}
<tbody>
{body}</tbody>
</table>
</div>"""


def _render_head() -> str:
    disturbance_heads = "".join(f"<th>{head}</th>" for head in DISTURBANCE_HEADS)
    return f"""<thead>
<tr>
  <th rowspan="2" class="method-col">Method</th>
  <th colspan="3" class="floor grippy">grippy floor</th>
  <th colspan="3" class="floor slippery">slippery floor</th>
  <th rowspan="2">synth</th>
  <th rowspan="2">len</th>
</tr>
<tr>{disturbance_heads}{disturbance_heads}</tr>
</thead>"""


def _render_method_row(
    *,
    percentage: int,
    method: Method,
    best: dict[tuple[int, str], BestConfig],
    winners: dict[str, dict[str, str]],
) -> str:
    config = best.get((percentage, method.tag))
    if config is not None:
        label = f'<a href="{escape(config.url)}" target="_blank" rel="noopener">{method.label}</a>'
    else:
        label = method.label
    cells = f'<th class="method-col">{label}</th>'
    for condition in CONDITION_LABELS:
        cells += _render_reward_cell(
            config=config,
            condition=condition,
            is_best=winners[condition].get(method.family) == method.tag,
            colormap=_COLORMAPS[condition.split("_")[0]],
        )

    if config is not None and config.invented_percent is not None:
        synth = f"{config.invented_percent:.0f}%"
    else:
        synth = '<span class="muted">—</span>'
    if config is not None and config.length is not None:
        p10, median, p90 = config.length
        length = f"{median:.0f} <span class='muted'>({p10:.0f}–{p90:.0f})</span>"
    else:
        length = '<span class="muted">—</span>'
    cells += f'<td class="aug">{synth}</td><td class="aug">{length}</td>'
    return f"<tr>{cells}</tr>\n"


def _render_reward_cell(
    *, config: BestConfig | None, condition: str, is_best: bool, colormap
) -> str:
    if config is None or condition not in config.rewards:
        return '<td class="empty"></td>'
    mean, standard_deviation = config.rewards[condition]
    fill = to_hex(c=colormap(min(max(mean, 0.0), 1.0)))
    best_class = " best" if is_best else ""
    return (
        f'<td class="reward{best_class}" style="--fill:{fill};--ink:{_ink(fill=fill)}">'
        f'<span class="mean">{mean:.3f}</span>'
        f'<span class="deviation">±{standard_deviation:.3f}</span></td>'
    )


def _ink(*, fill: str) -> str:
    if _contrast(fill, _DARK_INK) >= _contrast(fill, _LIGHT_INK):
        return _DARK_INK
    return _LIGHT_INK


def _contrast(first: str, second: str) -> float:
    luminances = sorted(
        (_relative_luminance(first), _relative_luminance(second)), reverse=True
    )
    return (luminances[0] + 0.05) / (luminances[1] + 0.05)


def _relative_luminance(hex_color: str) -> float:
    channels = []
    for index in (1, 3, 5):
        channel = int(hex_color[index : index + 2], 16) / 255
        channels.append(
            channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        )
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


_STYLE = """
:root {
  color-scheme: light dark;
  --background: #ffffff; --surface: #f6f7f9; --border: #d9dce1;
  --text: #14161a; --muted: #6b7280;
  --grippy-tint: #eef4fb; --slippery-tint: #fdf0e6;
}
@media (prefers-color-scheme: dark) {
  :root {
    --background: #16181d; --surface: #1e2128; --border: #333842;
    --text: #e8eaee; --muted: #9aa1ac;
    --grippy-tint: #1b2430; --slippery-tint: #2a2119;
  }
}
body {
  margin: 0 auto; padding: 2rem 1.5rem 3rem; max-width: 70rem;
  background: var(--background); color: var(--text);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; line-height: 1.4;
}
h1 { font-size: 1.5rem; margin: 0 0 0.5rem; }
h2 { font-size: 1.05rem; margin: 2.2rem 0 0.5rem; font-weight: 600; }
.note { color: var(--muted); font-size: 0.82rem; margin: 0 0 0.6rem; max-width: 52rem; }
.note b { color: var(--text); font-weight: 600; }
.table-wrapper { overflow-x: auto; }
table {
  border-collapse: collapse; width: 100%; min-width: 50rem; font-size: 0.82rem;
}
th, td { border: 1px solid var(--border); text-align: center; }
thead th { background: var(--surface); padding: 0.3rem 0.5rem; font-weight: 600; white-space: nowrap; }
th.floor.grippy { background: var(--grippy-tint); }
th.floor.slippery { background: var(--slippery-tint); }
tbody th { background: var(--surface); padding: 0.35rem 0.6rem; font-weight: 600; }
th.method-col { text-align: left; white-space: nowrap; }
th.method-col a { color: inherit; text-decoration: none; }
th.method-col a:hover { text-decoration: underline; }
tr.group th {
  text-align: left; background: var(--background); border-left: none; border-right: none;
  font-size: 0.9rem; padding: 0.7rem 0.2rem 0.25rem;
}
td.reward {
  background: var(--fill); color: var(--ink); padding: 0.3rem 0.4rem; line-height: 1.15;
}
td.reward .mean, td.reward .deviation { display: block; }
.mean { font-size: 0.9rem; font-weight: 600; font-variant-numeric: tabular-nums; }
td.best .mean { font-weight: 800; text-decoration: underline; text-underline-offset: 2px; }
.deviation { font-size: 0.72rem; font-variant-numeric: tabular-nums; }
td.empty { background: repeating-linear-gradient(
  45deg, transparent, transparent 5px, var(--surface) 5px, var(--surface) 10px); }
td.aug { padding: 0.3rem 0.5rem; font-variant-numeric: tabular-nums; white-space: nowrap; }
.muted { color: var(--muted); }
"""


if __name__ == "__main__":
    main()
