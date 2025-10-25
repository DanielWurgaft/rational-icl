from .analysis_utils import *
from .utils import *
from .models import *
from .make_figs import *
from dotenv import load_dotenv
from pyprojroot import here
import argparse
load_dotenv()

# suppress logging and warnings
import warnings
import logging
from pandas.errors import SettingWithCopyWarning

logging.getLogger("utils").setLevel(
    logging.WARNING
)  # suppress all but most severe warnings from train_utils
logging.getLogger("models").setLevel(
    logging.WARNING
)  # suppress logging of model parameters from models.py
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

class AnalysisPipeline:
    def __init__(self, setting, exp_name, **kwargs):
                
        self.cache_dir = os.getenv("CACHE_DIR")
        self.setting = setting
        self.exp = exp_name
        self.exp_dir = here(f"experiments/{self.setting}/{self.exp}")
        # read experiment params from first line
        with open(os.path.join(self.exp_dir, "experiment_params.txt"), "r") as f:
            self.exp_params = json.loads(f.readline())
            # close file
            f.close()

        # max sequences to evaluate
        self.num_eval_sequences = kwargs.get("num_eval_sequences", 500)
        if self.setting == "classification":
            self.num_eval_sequences_nll_computation = 50000
        else:
            self.num_eval_sequences_nll_computation = 10000

        # define evaluation need based on task
        if self.setting == "categorical-sequence":
            self.remove_last_prediction = True
        else:
            self.remove_last_prediction = False

        # define metric for evaluation
        if self.setting == "categorical-sequence":
            self.metric_name = "KL"
        elif self.setting == "linear-regression":
            self.metric_name = "MSE"
        elif self.setting == "classification":
            self.metric_name = "KL"

        # set seeds for reproducibility
        self.RNG = np.random.default_rng(self.exp_params["random_seed"])
        torch.manual_seed(self.exp_params["random_seed"])

        print("experiment params: ", self.exp_params)

        # choose variables to evaluate
        self.context_length_lst = kwargs.get("context_length_to_eval", self.exp_params["context_length_lst"])
        self.mlp_expansion_factor_lst = kwargs.get("mlp_expansion_factor_to_eval", self.exp_params["mlp_expansion_factor_lst"])
        self.num_dims_lst = kwargs.get("num_dims_to_eval", self.exp_params["num_dims_lst"])
        self.num_tasks_lst = kwargs.get("num_tasks_to_eval", self.exp_params["num_tasks_lst"])
        # read in data
        self.transformer_df_raw, self.algo_df_raw = make_results_dfs(self.exp_params, self.cache_dir)

        self.algo_names_dict = self.algo_df_raw.iloc[0]['config'].algo_names_dict
  
        self.include_optimal_constant_solution = kwargs.get("include_optimal_constant_solution", False)
        if self.include_optimal_constant_solution:
            self.algo_names_dict["optimal_constant"] = "$C$"


        # initial parameter guesses for Bayesian model from each setting
        if self.setting == "classification":
            self.params_init = {"log_beta": torch.log(torch.tensor(0.12)), "log_alpha": torch.log(torch.tensor(0.8)), "log_gamma": torch.log(torch.tensor(1.5))}
        elif self.setting == "linear-regression":
            if len(self.algo_names_dict) == 2:
                self.params_init = {"log_beta": torch.log(torch.tensor(0.3)), "log_alpha": torch.log(torch.tensor(0.65)), "log_gamma": torch.log(torch.tensor(0.01))}
            else:
                self.params_init = {"log_beta": torch.log(torch.tensor(0.45)), "log_alpha": torch.log(torch.tensor(0.65)), "log_gamma": torch.log(torch.tensor(0.01))}
        elif self.setting == "categorical-sequence":
            self.params_init = {"log_beta": torch.log(torch.tensor(0.2)), "log_alpha": torch.log(torch.tensor(0.6)), "log_gamma": torch.log(torch.tensor(0.85))}
        else:
            raise ValueError(f"Setting {self.setting} not supported")


    def process_algo_df(self, algo_df_raw, num_dims_to_eval, context_length_to_eval, 
                        num_tasks_to_eval, load_saved_evaluation, increase_generalized_code_complexity=False):
        """
        Process the algo df.
        """
        algo_df = algo_df_raw[algo_df_raw["num_dims"].isin(num_dims_to_eval) \
                              & (algo_df_raw["context_length"].isin(context_length_to_eval)) \
                              & (algo_df_raw["num_tasks"].isin(num_tasks_to_eval))]

        algo_df = run_evaluation_algorithms(
            algo_df,
            load_saved_evaluation=load_saved_evaluation,  
            batch_size=16,
            exp_params=self.exp_params,
            num_eval_sequences=self.num_eval_sequences,
            num_eval_sequences_nll_computation=self.num_eval_sequences_nll_computation,
            cache_dir=self.cache_dir,
            include_optimal_constant_solution=True # add in order to compute optimal constant solution
        )

        algo_df = get_algorithm_complexities(algo_df, increase_generalized_code_complexity=increase_generalized_code_complexity,
                                              include_optimal_constant_solution=self.include_optimal_constant_solution)
        # Convert all *_complexity columns to float64
        for col in algo_df.columns:
            if col.endswith("_complexity"):
                algo_df[col] = algo_df[col].astype(np.float64)

        # For each *_train_metrics column, create a new column with id_all_nll values
        for col in algo_df.columns:
            if col.endswith("_train_metrics"):
                algo_name = col.replace("_train_metrics", "")
                algo_df[f"{algo_name}_id_all_nll"] = algo_df[col].apply(
                    lambda x: x.get("train_all_nll", 0) if isinstance(x, dict) else 0
                )

        # For each *_eval_metrics column, create a new column with ood_all_nll values
        for col in algo_df.columns:
            if col.endswith("_eval_metrics"):
                algo_name = col.replace("_eval_metrics", "")
                algo_df[f"{algo_name}_ood_all_nll"] = algo_df[col].apply(
                    lambda x: x.get("eval_all_nll", 0) if isinstance(x, dict) else 0
                )
        
        return algo_df
    
    
    def process_transformer_df(self, transformer_df_raw, algo_df, num_dims_to_eval, context_length_to_eval, 
                               num_tasks_to_eval, mlp_expansion_factor_to_eval, 
                               load_saved_evaluation, compute_distance_from_algos, random_transformer_baseline=False):
        """
        Process the transformer df.
        """
        # processing transformer df
        transformer_df_all_checkpoints = transformer_df_raw[(transformer_df_raw["num_dims"].isin(num_dims_to_eval))
                                        & (transformer_df_raw["context_length"].isin(context_length_to_eval))
                                        & (transformer_df_raw["mlp_expansion_factor"].isin(mlp_expansion_factor_to_eval))
                                        & (transformer_df_raw["num_tasks"].isin(num_tasks_to_eval))
                                        ]

        transformer_df_all_checkpoints = run_evaluation_transformer(
            transformer_df_all_checkpoints,
            load_saved_evaluation=load_saved_evaluation,  
            only_final_models=False,
            exp_params=self.exp_params,
            num_eval_sequences=self.num_eval_sequences,
            random_transformer_baseline=random_transformer_baseline
        )

        if compute_distance_from_algos:
            # compute distances between transformer and algorithms
            transformer_df_all_checkpoints = compute_distances(
                transformer_df_all_checkpoints,
                algo_df,
                remove_last_prediction=self.remove_last_prediction,
                distance_function=self.metric_name,
                include_optimal_constant_solution=self.include_optimal_constant_solution
            )

        return transformer_df_all_checkpoints
    

    def main(self, load_saved_evaluation, fit_Bayesian_model, load_saved_Bayesian_model_params, make_figs, increase_generalized_code_complexity=False):

        num_tasks_to_eval = self.exp_params["num_tasks_lst"]

        transformer_df_raw, algo_df_raw = make_results_dfs(self.exp_params, self.cache_dir)

        # iterate over context length, mlp expansion factor, num dims and evaluate
        for context_length in self.context_length_lst:
            for mlp_expansion_factor in self.mlp_expansion_factor_lst:
                for num_dims in self.num_dims_lst:
                    print(f"context_length: {context_length}, mlp_expansion_factor: {mlp_expansion_factor}, num_dims: {num_dims}")

                    algo_df = self.process_algo_df(algo_df_raw, num_dims_to_eval=[num_dims],
                                                   context_length_to_eval=[context_length],
                                                   num_tasks_to_eval=num_tasks_to_eval,
                                                   load_saved_evaluation=load_saved_evaluation,
                                                   increase_generalized_code_complexity=increase_generalized_code_complexity)
                    

                    # process transformer df
                    transformer_df_all_checkpoints = self.process_transformer_df(transformer_df_raw, algo_df, num_dims_to_eval=[num_dims],
                                                                  context_length_to_eval=[context_length],
                                                                  num_tasks_to_eval=num_tasks_to_eval,
                                                                  mlp_expansion_factor_to_eval=[mlp_expansion_factor],
                                                                  load_saved_evaluation=load_saved_evaluation,
                                                                  compute_distance_from_algos=True if fit_Bayesian_model else False)
                    if fit_Bayesian_model:
                        transformer_df = find_approximate_interpolation_threshold(transformer_df_all_checkpoints, threshold_percentile = 0.1 if self.setting == "linear-regression" else 0.2).query("included_in_interpolation_analysis == 1", engine="python")
                        model_fitter = HierarchicalBayesianModelFitter(
                            transformer_df=transformer_df,
                            algo_df=algo_df,
                            mlp_expansion_factor=mlp_expansion_factor,
                            context_length=context_length,
                            num_dims=num_dims,
                            params_init=self.params_init,
                            metric_name=self.metric_name,
                            load_saved_evaluation=load_saved_Bayesian_model_params,
                            remove_last_prediction=self.remove_last_prediction,
                            add_to_df=True,
                            baseline_lst=["optimal_constant_baseline"]
                        )
                        transformer_df, history, model = model_fitter.fit()
                    if make_figs:
                        fig_gen = FigureGenerator(transformer_df, transformer_df_all_checkpoints, algo_df, self.setting, self.algo_names_dict)
                        fixed_values = {"context_length": context_length, "mlp_expansion_factor": mlp_expansion_factor, "num_dims": num_dims}
                        fixed_values_task_div_threshold_plot = {"context_length": context_length, "mlp_expansion_factor": mlp_expansion_factor, "num_dims": num_dims, "checkpoint": self.exp_params["save_steps"][-1]}
                        if self.setting == "classification":
                            train_mode = "iwl"
                        else:
                            train_mode = "train"
                        fig_configs = {
                                    "prediction comparison": {"fixed_values": fixed_values}, 
                                    "posterior odds": {"fixed_values": fixed_values}, 
                                    "algorithm probabilities": {"fixed_values": fixed_values, "task_diversity_values": [16, 128, 512]}, 
                                    "sublinear evidence accumulation": {"fixed_values": fixed_values,"compare_with_predicted": False,  "tasks_to_label": [4, 64, 256, 1024]}, 
                                    "transience predictions": {"fixed_values": fixed_values, "two_hypotheses_cutoff": transformer_df["approximate_interpolation_threshold"].iloc[0]}, 
                                    "transience appendix": {"fixed_values": fixed_values, "mode": "eval", "x_scale": "log"}, 
                                    "task diversity threshold appendix": {"fixed_values": fixed_values_task_div_threshold_plot, "train_mode": train_mode}, 
                                    "task diversity threshold": {"fixed_values": fixed_values_task_div_threshold_plot, "train_mode": train_mode},
                                    "relative and absolute distance": {"fixed_values": fixed_values},
                                    }
                        fig_gen.make_figs(fig_configs)

                        # clear memory
                        del transformer_df
                        del algo_df
                        del history
                        del model
                        gc.collect()
                        

if __name__ == "__main__":
    # set task and number of eval sequences with named command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--num_eval_sequences", type=int, default=500)
    parser.add_argument("--exp_name", type=str, default="mlp-titration-exp")
    parser.add_argument("--load_saved_evaluation", type=bool, default=True)
    parser.add_argument("--fit_Bayesian_model", type=bool, default=True)
    parser.add_argument("--load_saved_Bayesian_model_params", type=bool, default=True)
    parser.add_argument("--make_figs", type=bool, default=True)
    parser.add_argument("--increase_generalized_code_complexity", type=bool, default=False)
    args = parser.parse_args()

    analysis_pipeline = AnalysisPipeline(args.task, args.exp_name, args.num_eval_sequences)
    analysis_pipeline.main(args.load_saved_evaluation, args.fit_Bayesian_model, args.load_saved_Bayesian_model_params, args.make_figs, args.increase_generalized_code_complexity)