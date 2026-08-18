# Master's thesis tools

Everything around the experiments of
[imitation_learning_and_data_control_augmentation](../imitation_learning_and_data_control_augmentation)
that is not part of the method itself.

## Results page

`create_results_page.py` builds one HTML page that shows every evaluation
result: one table per evaluation condition, the dataset percentages as rows and
the six methods as columns. A cell shows the reward mean with its standard
deviation, is colored by the reward, and links to its run.

Paste the URL of a run into `runs.txt`, one per line, then run:

    uv run create_results_page.py

The script reads the dataset percentage, the method and the evaluation
condition from each run itself, so the order of the lines does not matter. It
writes `results.html` beside itself.

## Cluster

`sweep_agent.sbatch` runs Weights & Biases sweep agents on the Slurm cluster.
Copy it to the cluster and submit it there:

    sbatch --export=ALL,SWEEP_PATH=<entity>/<project>/<sweep_id> sweep_agent.sbatch
