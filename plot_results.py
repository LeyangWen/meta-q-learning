import wandb
import pandas as pd

api = wandb.Api()
entity = this_run.entity
project = this_run.project
run_id = this_run.id
runs = api.run(f"{entity}/{project}/{run_id}")
runs.history(keys=["train/values/Productivity (br_per_hr)", "train/episode"])
data = [[1, 2], [4, 5], [7, 8]]
table = wandb.Table(data=data, columns=["x", "y"])
this_run.log(
    {
        "my_custom_plot_id": wandb.plot.line(
            table, "x", "y", title="Custom Y vs X Line Plot"
        )
    }
)

entity, project = "qaq37", "HRC_model_based_rl_2"
runs = api.runs(entity + "/" + project)

latest_run = runs[0]

# Get logged data
history = latest_run.history()
name = latest_run.name


summary_list, config_list, name_list = [], [], []
for run in runs:
    print(run.name)
    name = run.name
    config = run.config
    summary = run.summary
    history = run.history()

    keys = history.columns.values.tolist()
    episode = history['train/episode']
    episode_start_idx = episode[episode == 0].index.tolist()

    # .summary contains the output keys/values
    #  for metrics such as accuracy.
    #  We call ._json_dict to omit large files
    run.config


    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

runs_df.to_csv("project.csv")
