"""The names the experiment repo writes its evaluation results under.

The experiment repo is not importable from here, so these are copied rather
than shared. They must match `scripts/evaluation` of
`imitation_learning_and_data_control_augmentation`.

The two results pages predate this file and still carry the unnumbered labels
that older runs were written with, such as `grippy_undisturbed`. The labels
below are the current ones.
"""

STAGE = "4_evaluation"

# The mean over every condition, which the run writes alongside the six.
ALL_CONDITIONS_LABEL = "0_all_conditions"

CONDITION_LABELS = (
    "1_grippy_undisturbed",
    "2_grippy_low_disturbance",
    "3_grippy_high_disturbance",
    "4_slippery_undisturbed",
    "5_slippery_low_disturbance",
    "6_slippery_high_disturbance",
)

# The metric a sweep maximises and the final runs are ranked by.
RANKING_METRIC = f"{STAGE}/{ALL_CONDITIONS_LABEL}/1_reward_mean"
