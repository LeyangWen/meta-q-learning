import wandb
import pandas as pd

api = wandb.Api()

def log_precentage(wandb_run, GT, key):
    api = wandb.Api()
    run = api.run(f"{wandb_run.entity}/{wandb_run.project}/{wandb_run.id}")
    data = run.history(keys=["train/episode",key])
    data[key + " %"] = data[key]/GT*100
    table = wandb.Table(data=data)
    wandb_run.log(
        {
            f"{key}_precentage": wandb.plot.line(
                table,"train/episode", key + " %",  title=f"{key} %"
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
