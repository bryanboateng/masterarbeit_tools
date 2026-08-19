# Master's thesis tools

Everything around the experiments of
[imitation_learning_and_data_control_augmentation](../imitation_learning_and_data_control_augmentation)
that is not part of the method itself.

## Results page

`create_results_page.py` builds one HTML page that shows every evaluation
result: one table per evaluation condition, the dataset percentages as rows and
the six methods as columns. A cell shows the reward mean with its standard
deviation, is colored by the reward, and links to its run.

The numbers and the links are kept by hand in `results.txt`, one line per cell:

    [grippy_undisturbed]
    10 tacil-dp 0.351 0.360 https://wandb.ai/…/runs/3ohsiome

Then build the page:

    uv run create_results_page.py

It writes `index.html` beside itself and reports how many cells are still
missing. That file is the hosted page, so commit it and push: GitHub Pages
serves it at the root of the site. An unknown method, a percentage that is not 10, 25, 50 or 100, or a
cell filled twice stops the build and names the line.

## Cluster

`sweep_agent.sbatch` runs Weights & Biases sweep agents on the Slurm cluster.
Copy it to the cluster and submit it there:

    sbatch --export=ALL,SWEEP_PATH=<entity>/<project>/<sweep_id> sweep_agent.sbatch
