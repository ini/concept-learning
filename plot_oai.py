from plot import *
from lightning_ray import filter_results

if __name__ == "__main__":
    PLOT_FUNCTIONS = {
        "neg_intervention": plot_negative_interventions,
        "pos_intervention": plot_positive_interventions,
        "random": plot_random_concepts_residual,
        #'disentanglement': plot_disentanglement,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-dir",
        type=str,
        default=os.environ.get("CONCEPT_SAVE_DIR", "./saved"),
        help="Experiment directory",
    )
    parser.add_argument(
        "--mode", nargs="+", default=PLOT_FUNCTIONS.keys(), help="Evaluation modes"
    )
    parser.add_argument(
        "--groupby",
        nargs="+",
        default=["model_type"],
        help="Config keys to group plots by",
    )

    args = parser.parse_args()

    # Recursively search for 'tuner.pkl' file within the provided directory
    # If multiple are found, use the most recently modified one
    experiment_paths = Path(args.exp_dir).resolve().glob("**/eval/tuner.pkl")
    experiment_path = sorted(experiment_paths, key=os.path.getmtime)[-1].parent.parent

    # Load evaluation results
    tuner = tune.Tuner.restore(str(experiment_path / "eval"), trainable=evaluate)
    results = group_results(tuner.get_results(), groupby="dataset")
    names = ["latent_residual", "decorrelated_residual", "mi_residual", "iter_norm"]
    settings = [
        ("latent_residual", None),
        ("decorrelated_residual", None),
        ("mi_residual", None),
        ("latent_residual", "iter_norm"),
    ]
    for name, setting in zip(names, settings):
        # Plot results for each dataset
        for dataset_name, dataset_results in results.items():
            for mode in args.mode:
                # print(dataset_results._experiment_analysis.trials)
                def fun(config):
                    return (
                        config["model_type"] == setting[0]
                        and config["norm_type"] == setting[1]
                    )

                filtered_results = filter_results(fun, dataset_results)
                PLOT_FUNCTIONS[mode](
                    filtered_results,
                    dataset_name,
                    groupby=args.groupby,
                    save_dir=experiment_path / "plots",
                    name=name,
                )
