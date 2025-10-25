import os
import numpy as np
import yaml
import argparse
import json
from pyprojroot import here

class Config:
    def __init__(
        self,
        setting,
        num_dims,
        num_tasks,
        context_length=100,
        num_eval_tasks=500,
        prior_params=None,
        random_seed=1,
        model_type="gpt-neo-x",
        num_hidden_layers=1,
        num_attention_heads=1,
        hidden_size=128,
        mlp_expansion_factor=3,
        attention_dropout=0.0,
        learning_rate=1e-3,
        max_steps=100_000,
        max_grad_norm=1,
        weight_decay=0.0,
        warmup_steps=0,
        lr_scheduler_type=None,
        lr_scheduler_kwargs={},
        batch_size=32,
        gradient_accumulation_steps=1,
        logging_steps=500,
        eval_steps=500,
        save_steps=[],
        name_suffix="",
        zipf_param=None,
        make_run_dirs=False,
        cache_dir=None,
    ):
        """
        Initialize the configuration based on the number of dimensions and setting.
        """
        # Validate setting
        if setting not in [
            "categorical-sequence",
            "linear-regression",
            "classification",
        ]:
            raise ValueError(
                "setting must be either 'categorical-sequence', 'linear-regression', or 'classification'"
            )

        # General Config
        self.setting = setting
        self.num_tasks = int(num_tasks)
        self.num_dims = int(num_dims)
        self.context_length = int(context_length)
        self.model_type = model_type
        self.random_seed = random_seed
        self.rng = np.random.default_rng(self.random_seed)
        self.name_suffix = name_suffix
        if zipf_param is None:
            self.zipf_param = 0
        else:
            self.zipf_param = zipf_param

        # set setting specific configuration
        self.set_setting_specific_params(setting, prior_params)

        # Transformer Configuration
        self.hidden_size = hidden_size  # Embedding size
        self.mlp_expansion_factor = mlp_expansion_factor
        self.intermediate_size = int(
            self.hidden_size * self.mlp_expansion_factor
        )  # MLP size
        self.num_hidden_layers = num_hidden_layers  # Number of transformer layers
        self.num_attention_heads = num_attention_heads  # Number of attention heads
        self.num_key_value_heads = self.num_attention_heads
        self.max_position_embeddings = self.block_size  # Maximum context length
        self.attention_dropout = attention_dropout

        # Training Configuration
        self.per_device_train_batch_size = batch_size
        self.per_device_eval_batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.logging_steps = logging_steps
        self.max_steps = max_steps
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        if lr_scheduler_type is None:
            self.lr_scheduler_kwargs = {}
            if self.warmup_steps > 0:
                self.lr_scheduler_type = "constant_with_warmup"
            else:
                self.lr_scheduler_type = "constant"
        else:
            self.lr_scheduler_type = lr_scheduler_type
            self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.eval_steps = eval_steps
        if save_steps:
            self.save_steps = save_steps
        else:
            self.save_steps = [self.max_steps]

        # Evaluation Configuration
        self.num_eval_tasks = num_eval_tasks

        # Filepaths Configuration
        if cache_dir is not None:

            # Create base cache directory
            self.cache_dir = cache_dir
            os.makedirs(cache_dir, exist_ok=True)
            self.setting_dir = os.path.join(cache_dir, setting)
            os.makedirs(self.setting_dir, exist_ok=True)

            # Create data directories
            self.data_dir = os.path.join(self.setting_dir, "data")
            os.makedirs(self.data_dir, exist_ok=True)
            # Training and evaluation data directories
            self.training_data_dir = os.path.join(self.data_dir, "train-data")
            os.makedirs(self.training_data_dir, exist_ok=True)
            self.eval_data_dir = os.path.join(self.data_dir, "eval-data")
            os.makedirs(self.eval_data_dir, exist_ok=True)

            # train and eval data
            self.train_datapath = os.path.join(
                self.training_data_dir,
                f"{self.num_dims}dims-{self.num_tasks}tasks-{self.random_seed}seed"
                + "_data.npz",
            )
            self.eval_random_seed = self.random_seed + 1
            self.eval_rng = np.random.default_rng(self.eval_random_seed)
            self.eval_datapath = os.path.join(
                self.eval_data_dir,
                f"{self.num_dims}dims-{self.num_eval_tasks}tasks-{self.eval_random_seed}seed"
                + "_eval_data.npz",
            )

            # name run directories
            self.results_dir = os.path.join(self.setting_dir, "transformers")
            # define directories for saving checkpoints and logs
            run_dir_name = (
                f"{int(self.num_dims)}dims-{int(self.num_tasks)}tasks-{int(self.context_length)}context-{int(self.num_hidden_layers)}layers-{float(self.mlp_expansion_factor) if self.mlp_expansion_factor < 1 else int(self.mlp_expansion_factor)}expansionfactor-{int(self.per_device_train_batch_size)}batchsize-{float(self.learning_rate)}lr-{int(self.max_steps)}steps-{int(self.random_seed)}seed"
                + ("-" + self.name_suffix if self.name_suffix != "" else "")
            )

            self.run_results_dir = os.path.join(self.results_dir, run_dir_name)

            # only create the directory below when training begins
            self.output_dir = os.path.join(self.run_results_dir, "checkpoints")
            if make_run_dirs:
                os.makedirs(self.results_dir, exist_ok=True)
                os.makedirs(self.run_results_dir, exist_ok=True)

                # make directory for saving model checkpoints and avoid duplicates
                os.makedirs(self.output_dir, exist_ok=False)

                # make yaml for run
                yaml_str = self.make_yaml_str()
                with open(os.path.join(self.run_results_dir, "config.yaml"), "w") as f:
                    f.write(yaml_str)

    def set_setting_specific_params(self, setting, prior_params):
        """Set setting specific parameters based on the setting chosen. 
           When adding a new setting, add an if clause for its specific parameters here.
        """
        if setting == "categorical-sequence":
            if prior_params is None:
                self.prior_params = [1 for _ in range(self.num_dims)]
            else:
                self.prior_params = prior_params
            vocab = {str(i): i for i in range(self.num_dims)}
            self.start_token = "s"
            vocab.update(
                {
                    self.start_token: self.num_dims,  # start token
                }
            )
            self.tokenizer_vocab = vocab
            self.block_size = self.context_length + 1  # +1 for start token
            self.inputs_key = "input_ids"
            self.labels_key = "next_token_distribution"
            self.metric_name = "KL Divergence"
            self.algo_names_dict = {
                    "memorized": "$M$",
                    "generalized": "$G$",
                }

        elif setting == "linear-regression":
            if prior_params is None:
                self.prior_params = [0, 1]
            else:
                self.prior_params = prior_params
            self.noise_variance = self.num_dims / 256
            self.block_size = 2 * self.context_length
            self.inputs_key = "xs"
            self.labels_key = "labels"
            self.metric_name = "MSE"
            self.algo_names_dict = {
                "memorized": "$M$",
                "generalized": "$G$",
            }
        elif setting == "classification":
            if prior_params is None:
                self.prior_params = [0, 1 / self.num_dims]
            else:
                self.prior_params = prior_params
            self.block_size = self.context_length
            self.inputs_key = "pairs"
            self.labels_key = "labels"
            self.num_labels = 2
            self.metric_name = "Cross Entropy"
            self.algo_names_dict = {
                "memorized": "$M$",
                "generalized": "$G$",
            }
            self.noisy_items = True
            self.within_class_variance = 0.5

    def make_yaml_str(self):
        """Make a yaml string from the config."""
        # Convert parameters to native Python types before saving to YAML
        parameters = vars(self).copy()  # Make a copy to avoid modifying self

        for key, value in parameters.items():
            if isinstance(value, (np.integer, np.floating)):
                parameters[key] = (
                    value.item()
                )  # Convert NumPy scalar to native Python type
            elif isinstance(value, np.ndarray):
                parameters[key] = value.tolist()  # Convert NumPy array to list
            elif isinstance(value, str):
                parameters[key] = str(value)  # Ensure it's a Python string
            elif isinstance(value, dict):
                # Convert any NumPy values in dictionaries
                parameters[key] = {
                    str(k): (
                        v.item() if isinstance(v, (np.integer, np.floating)) else v
                    )
                    for k, v in value.items()
                }
        # remove random number generator if it exists
        parameters.pop("rng", None)
        parameters.pop("eval_rng", None)
        yaml_str = yaml.dump(parameters)
        return yaml_str

    @classmethod
    def from_yaml(cls, yaml_filepath, cache_dir, make_run_dirs=False):
        with open(yaml_filepath, "r") as f:
            data = yaml.safe_load(f)
        # call __init__ with the data from the yaml file
        return cls(
            cache_dir=cache_dir,
            setting=data.get("setting"),
            num_dims=data.get("num_dims"),
            num_tasks=data.get("num_tasks"),
            context_length=data.get("context_length"),
            num_eval_tasks=data.get("num_eval_tasks"),
            prior_params=data.get("prior_params"),
            random_seed=data.get("random_seed"),
            model_type=data.get("model_type"),
            num_hidden_layers=data.get("num_hidden_layers"),
            num_attention_heads=data.get("num_attention_heads"),
            hidden_size=data.get("hidden_size"),
            mlp_expansion_factor=data.get("mlp_expansion_factor"),
            attention_dropout=data.get("attention_dropout"),
            learning_rate=data.get("learning_rate"),
            max_steps=data.get("max_steps"),
            max_grad_norm=data.get("max_grad_norm"),
            weight_decay=data.get("weight_decay"),
            warmup_steps=data.get("warmup_steps"),
            lr_scheduler_type=data.get("lr_scheduler_type"),
            lr_scheduler_kwargs = data.get("lr_scheduler_kwargs"),
            batch_size=data.get("per_device_train_batch_size"),
            gradient_accumulation_steps=data.get("gradient_accumulation_steps"),
            logging_steps=data.get("logging_steps"),
            eval_steps=data.get("eval_steps"),
            save_steps=data.get("save_steps"),
            name_suffix=data.get("name_suffix"),
            make_run_dirs=make_run_dirs,
        )
    
    def asdict(self):
        return vars(self)
    

def create_sqrt_checkpoint_schedule(total_steps, num_checkpoints, start_step=20):
    """
    Generate a checkpoint schedule where the checkpoints are spaced out in a manner
    that is approximately linear in the square root of the training steps.

    Args:
        total_steps (int): Total number of training steps.
        num_checkpoints (int): Total number of checkpoints desired.

    Returns:
        list: List of steps at which checkpoints should be saved.
    """
    # Generate `num_checkpoints` evenly spaced points in square root space
    sqrt_space = np.linspace(np.sqrt(start_step), np.sqrt(total_steps), num_checkpoints)

    # Map the square root spaced points back to training steps
    if start_step < 0 or total_steps <= start_step:
        raise ValueError("start_step must be non-negative and less than total_steps.")
    checkpoint_steps = [int(step) for step in (sqrt_space**2)]

    # Ensure uniqueness and inclusion of the final step
    checkpoint_steps = sorted(set(checkpoint_steps))

    if checkpoint_steps[-1] != total_steps:
        checkpoint_steps.append(total_steps)

    return checkpoint_steps


def write_yaml_files(
    experiments_dir: str,
    exp_name: str,
    num_dims_lst: list,
    num_tasks_lst: list,
    context_length_lst: list,
    mlp_expansion_factor_lst: list,
    setting: str,
    random_seed: int,
    num_hidden_layers: int,
    hidden_size: int,
    max_steps: int,
    name_suffix: str,
    save_steps: list,
    batch_size: int,
    learning_rate: float,
    warmup_steps: int,
    lr_scheduler_type: str,
    lr_scheduler_kwargs: dict,
):
    """
    Write yaml files directly to an experiment directory.
    """
    # Create the experiment and yamls directories
    exp_dir = os.path.join(experiments_dir, setting, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Write experiment parameters to the experiment directory
    with open(os.path.join(exp_dir, "experiment_params.txt"), "w") as f:
        experiment_params = {
            "setting": setting,
            "num_dims_lst": num_dims_lst,
            "num_tasks_lst": num_tasks_lst,
            "context_length_lst": context_length_lst,
            "mlp_expansion_factor_lst": mlp_expansion_factor_lst,
            "random_seed": random_seed,
            "num_hidden_layers": num_hidden_layers,
            "hidden_size": hidden_size,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_steps": max_steps,
            "save_steps": save_steps,
            "name_suffix": name_suffix,
            "warmup_steps": warmup_steps,
            "lr_scheduler_type": lr_scheduler_type,
            "lr_scheduler_kwargs": lr_scheduler_kwargs,
        }
        f.write(json.dumps(experiment_params) + "\n")

    # Define yaml folder path
    yaml_dir = os.path.join(exp_dir, "yaml-configs")
    os.makedirs(yaml_dir, exist_ok=True)

    # if folder exists, delete all files inside it
    if os.path.exists(yaml_dir):
        for file in os.listdir(yaml_dir):
            os.remove(os.path.join(yaml_dir, file))

    # Write yaml files to yaml folder
    for num_dims in num_dims_lst:
        for num_tasks in num_tasks_lst:
            for context_length in context_length_lst:
                for mlp_expansion_factor in mlp_expansion_factor_lst:
                    config = Config(
                        setting=setting,
                        num_dims=num_dims,
                        num_tasks=num_tasks,
                        context_length=context_length,
                        mlp_expansion_factor=mlp_expansion_factor,
                        random_seed=random_seed,
                        num_hidden_layers=num_hidden_layers,
                        hidden_size=hidden_size,
                        max_steps=max_steps,
                        save_steps=save_steps,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        warmup_steps=warmup_steps,
                        lr_scheduler_type=lr_scheduler_type,
                        lr_scheduler_kwargs=lr_scheduler_kwargs,
                        name_suffix=name_suffix,
                    )
                    yaml_str = config.make_yaml_str()
                    filename = f"{int(num_dims)}dims-{int(num_tasks)}tasks-{int(context_length)}context-{float(mlp_expansion_factor) if mlp_expansion_factor < 1 else int(mlp_expansion_factor)}expansionfactor-{int(random_seed)}seed{'-'+name_suffix if name_suffix != '' else ''}.yaml"
                    with open(os.path.join(yaml_dir, filename), "w") as f:
                        f.write(yaml_str)

    print(
        f"Created experiment parameters at: {os.path.join(exp_dir, 'experiment_params.txt')}"
    )
    print(f"Created yaml files at: {yaml_dir}")


if __name__ == "__main__":
    # load env variables
    from dotenv import load_dotenv

    load_dotenv()

    # argparse variables
    parser = argparse.ArgumentParser()

    # add positional argument
    parser.add_argument("exp_name", type=str)

    # add keyword arguments
    parser.add_argument("--experiments_dir", type=str, default=here("experiments"))
    parser.add_argument("--setting", type=str, default="linear-regression")
    parser.add_argument("--num_dims_lst", type=int, default=[8], nargs="+")
    parser.add_argument("--num_tasks_lst", type=int, default=[4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096], nargs="+")
    parser.add_argument("--context_length_lst", type=int, default=[16], nargs="+")
    parser.add_argument("--mlp_expansion_factor_lst", type=float, default=[4], nargs="+")
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=100_000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--num_save_checkpoints", type=int, default=100)
    parser.add_argument("--lr_scheduler_type", type=str, default="inverse_sqrt")
    parser.add_argument("--lr_scheduler_kwargs", type=dict, default={})
    parser.add_argument("--name_suffix", type=str, default="inverse_sqrt_less_warmup")
    args = parser.parse_args()

    write_yaml_files(
        experiments_dir=args.experiments_dir,
        exp_name=args.exp_name,
        num_dims_lst=args.num_dims_lst,
        num_tasks_lst=args.num_tasks_lst,
        context_length_lst=args.context_length_lst,
        mlp_expansion_factor_lst=args.mlp_expansion_factor_lst,
        setting=args.setting,
        random_seed=args.random_seed,
        num_hidden_layers=args.num_hidden_layers,
        hidden_size=args.hidden_size,
        max_steps=args.max_steps,
        name_suffix=args.name_suffix,
        save_steps=create_sqrt_checkpoint_schedule(args.max_steps, args.num_save_checkpoints),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        lr_scheduler_kwargs=args.lr_scheduler_kwargs,
    )