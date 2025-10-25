import os
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import TrainerCallback, default_data_collator
import numpy as np
import logging
from .config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


############################
# initializing and creating datasets
############################


def make_dataset(config, mode):
    """
    Make a dataset from a given config.
    """
    prior_params = config.prior_params
    setting = config.setting
    if mode == "train":
        rng = config.rng
        num_tasks = config.num_tasks
        data_path = config.train_datapath
    else:
        rng = config.eval_rng
        num_tasks = config.num_eval_tasks
        data_path = config.eval_datapath
    if setting == "categorical-sequence":
        CategoricalSequenceDataset.make_dataset(
            num_tasks=num_tasks,
            data_path=data_path,
            prior_params=prior_params,
            rng=rng,
        )
    elif setting == "linear-regression":
        LinearRegressionDataset.make_dataset(
            num_tasks=num_tasks,
            data_path=data_path,
            prior_params=prior_params,
            rng=rng,
            num_dims=config.num_dims,
        )
    elif setting == "classification":
        ClassificationDataset.make_dataset(
            num_tasks=num_tasks,
            data_path=data_path,
            prior_params=prior_params,
            rng=rng,
            num_dims=config.num_dims,
            num_labels=config.num_labels,
        )


def init_dataset(config, mode, dataset_length=None, iwl=False):
    """
    Initialize a dataset from a given config.
    """
    if mode == "train":
        data_path = config.train_datapath
    else:
        data_path = config.eval_datapath
    if config.setting == "categorical-sequence":
        return CategoricalSequenceDataset(
            data_path, config, dataset_length=dataset_length
        )
    elif config.setting == "linear-regression":
        return LinearRegressionDataset(data_path, config, dataset_length=dataset_length)
    elif config.setting == "classification":
        if iwl and mode == "train":
            return ClassificationDataset(data_path, config, dataset_length=dataset_length, mode="iwl")
        else:
            return ClassificationDataset(data_path, config, dataset_length=dataset_length)

#########################
# general ICL dataset class
#########################


class ICLDataset(Dataset):
    def __init__(
        self, tasks_path: str, config: Config, dataset_length: Optional[int] = None
    ):
        self.config = config
        self.num_dims = config.num_dims
        self.context_length = config.context_length

        dataset = np.load(tasks_path)
        self.tasks = dataset["tasks"]
        self.len_tasks = self.tasks.shape[0]

        # set up random number generator
        self.rng = np.random.default_rng(config.random_seed)
        self.initial_rng_state = self.rng.bit_generator.state

        # set up zipfian distribution
        self.zipf_param = config.zipf_param
        self.class_probabilities = self.get_class_probabilities()

        # determines number of batches to collect for an epoch, if not defined, use len_tasks
        if dataset_length is None:
            self.dataset_length = self.len_tasks
        else:
            self.dataset_length = dataset_length

        logger.info(f"Shape of dataset (# tasks, # dims): {self.tasks.shape}")

    def __len__(self):
        """altered __len__ to allow arbitrary dataset length"""
        return self.dataset_length

    def get_class_probabilities(self):
        """
        Get the class probabilities for the dataset.
        """
        if self.zipf_param is None:
            return np.ones(len(self.tasks)) / len(self.tasks)
        else:
            unnormalized_probabilities = np.power(
                np.arange(len(self.tasks)), self.zipf_param
            )
            return unnormalized_probabilities / np.sum(unnormalized_probabilities)

    def reset_rng(self):
        self.rng.bit_generator.state = self.initial_rng_state

    def __gettask__(self):
        """
        Returns a task from the dataset.
        """
        i = self.rng.choice(self.len_tasks, p=self.class_probabilities)
        return self.tasks[i, :], i



#########################
# categorical sequences dataset
#########################


class CategoricalSequenceDataset(ICLDataset):
    def __init__(
        self, tasks_path: str, config: Config, dataset_length: Optional[int] = None
    ):
        super().__init__(tasks_path, config, dataset_length)
        self.tokenizer_vocab = config.tokenizer_vocab
        self.start_token = config.start_token

    def __getitem__(self, i):
        task, i = self.__gettask__()
        # sample from categorical distribution based on distributions provided in task
        task_sample = self.rng.choice(
            self.num_dims, size=self.context_length, replace=True, p=task
        )
        # insert start token
        task_sample = np.insert(task_sample, 0, self.tokenizer_vocab[self.start_token])
        # convert to tensor
        input_ids = torch.tensor(task_sample, dtype=torch.long)
        # copy input ids to labels
        labels = input_ids.clone()
        # create distribution labels for each token in the sequence
        distribution = (
            torch.tensor(task, dtype=torch.float)
            .unsqueeze(0)
            .tile(input_ids.shape[0], 1)
        )
        return {
            "labels": labels,
            "input_ids": input_ids,
            "next_token_distribution": distribution,
        }

    @staticmethod
    def make_dataset(num_tasks, data_path, prior_params, rng, save=True):
        """
        Draws and saves samples from a Dirichlet distribution for a categorical-sequence dataset.

        Args:
            num_tasks (int): Number of tasks to generate.
            data_path (str): Path to save the generated dataset.
            prior_params (list or None): Parameters for the Dirichlet distribution. Defaults to config.prior_params.
            rng (np.random.Generator): Random number generator.
        """
        # Get row probabilities for num_tasks tasks
        task_probs = rng.dirichlet(prior_params, size=num_tasks)

        # Save arrays
        if save:
            np.savez(data_path, tasks=task_probs)
        else:
            return task_probs

    @staticmethod
    def compute_metrics(
        eval_pred: Tuple[np.ndarray, np.ndarray],
        compute_softmax: bool = True,
        mode: str = "eval",
        return_all: bool = False,
        compute_nll: bool = False,
        return_raw_loss: bool = False,
    ) -> Dict[str, Any]:
        """
        Computes the KL divergence loss as an evaluation metric.
        """
        # get shape
        logits, labels = eval_pred
        logits = torch.as_tensor(logits, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.float32)

        if labels.shape != logits.shape:
            labels = labels.reshape(logits.shape)

        # remove last token from labels and logits since it is not being trained on
        labels = labels[:, :-1]
        logits = logits[:, :-1]

        batch_size, context_length, num_dims = logits.shape

        if compute_softmax:
            probs = nn.functional.softmax(logits, dim=-1)

            log_probs = torch.log(
                probs.flatten(end_dim=-2)
            )  # need logprobs for KL divergence calculation
        else:
            log_probs = logits.flatten(end_dim=-2)

        labels = labels.flatten(end_dim=-2)
        kl_div_fn = nn.KLDivLoss(reduction="none")
        with torch.no_grad():
            metric = (
                kl_div_fn(log_probs, labels)
                .sum(axis=-1)
                .reshape((batch_size, context_length))
            )

        mean_metric = torch.mean(metric)
        if return_raw_loss:
            return mean_metric
        else:
            metric = metric.detach().cpu()
            out = {}
            metric_per_trial = torch.mean(metric, axis=0)
            out[f"{mode}_metric"] = mean_metric.item()
            out[f"{mode}_metric_per_trial"] = metric_per_trial.numpy().tolist()
            if return_all:
                out[f"{mode}_all_metric"] = metric.numpy().tolist()
            if compute_nll:
                out[f"{mode}_nll"] = mean_metric.item()  # KL divergence is -log likelihood + constant term (which we do not care about in a bayes factor)
                out[f"{mode}_all_nll"] = metric.numpy().tolist()
            return out

#########################
# linear regression dataset
#########################


# Define custom dataset for linear regression
class LinearRegressionDataset(ICLDataset):
    def __init__(
        self, tasks_path: str, config: Config, dataset_length: Optional[int] = None
    ):
        super().__init__(tasks_path, config, dataset_length)
        self.noise_variance = config.noise_variance
        self.prior_params = config.prior_params

    def __getitem__(self, i):
        task, i = self.__gettask__()

        # 1) Sample xs from a multivariate Gaussian
        xs = self.rng.multivariate_normal(
            mean=self.prior_params[0] * np.ones(self.num_dims),
            cov=self.prior_params[1] * np.eye(self.num_dims),
            size=self.context_length,
        )  # shape: (context_length, num_dims)

        # 2) Sample noise from a univariate normal distribution
        noise = self.rng.normal(
            loc=0.0, scale=np.sqrt(self.noise_variance), size=self.context_length
        )
        # shape: (context_length,)

        # 3) Compute ys
        ys = xs @ task + noise
        # shape: (context_length,)

        # 4) Convert to float32 Tensors for PyTorch
        xs = torch.tensor(xs, dtype=torch.float32)
        ys = torch.tensor(ys, dtype=torch.float32)

        # Return dictionary compatible with model's forward (xs, labels)
        return {"xs": xs, "labels": ys}

    @staticmethod
    def make_dataset(num_tasks, data_path, prior_params, rng, num_dims, save=True):
        """
        Draws and saves samples from a Multivariate Gaussian distribution for a linear-regression dataset.

        Args:
            num_tasks (int): Number of tasks to generate.
            data_path (str): Path to save the generated dataset.
            prior_params (list or None): Parameters for the Gaussian distribution. Defaults to config.prior_params.
            rng (np.random.Generator): Random number generator.
            num_dims (int): Number of dimensions for the linear regression.
        """
        # Get row probabilities for num_tasks tasks
        task_weights = rng.multivariate_normal(
            prior_params[0] * np.ones(num_dims),
            prior_params[1] * np.eye(num_dims),
            size=num_tasks,
        )

        # Save arrays
        if save:
            np.savez(data_path, tasks=task_weights)
        else:
            return task_weights

    @staticmethod
    def compute_metrics(
        eval_pred,
        mode: str = "eval",
        return_all: bool = False,
        compute_nll: bool = False,
        noise_variance=None,
        return_raw_loss: bool = False,
    ):
        """
        Computes the Mean Squared Error (MSE) loss as an evaluation metric.

        Args:
                - eval_pred: An EvalPrediction object with two attributes:
                - predictions: The model's output predictions.
                - label_ids: The true labels.
                - mode: The mode of the evaluation.
                - return_all: Whether to return all the metrics.
                - compute_nll: Whether to compute the negative log likelihood.
                - noise_variance: The variance of the noise.
                - return_raw_loss: Whether to return the raw loss.

        Returns:
            A dictionary with a single key "mse" whose value is the computed MSE.
        """
        predictions, labels = eval_pred

        # Calculate mean squared error
        mse = nn.MSELoss(reduction="none")
        metric = mse(predictions, labels)

        if return_raw_loss:
            return torch.mean(metric)
        else:
            metric = metric.detach().cpu()
            mean_metric = torch.mean(metric)
            metric_per_trial = metric.mean(dim=0)

            out = {}
            out[f"{mode}_metric"] = mean_metric.item()
            out[f"{mode}_metric_per_trial"] = metric_per_trial.tolist()
            if return_all:
                out[f"{mode}_all_metric"] = metric.tolist()
            if compute_nll:
                out[f"{mode}_nll"] = mean_metric.item() / (2 * noise_variance)
                out[f"{mode}_all_nll"] = (metric.numpy() / (2 * noise_variance)).tolist()
            return out


#########################
# classification dataset
#########################


class ClassificationDataset(ICLDataset):
    def __init__(
        self, tasks_path: str, config: Config, dataset_length: Optional[int] = None, mode="icl"
    ):
        super().__init__(tasks_path, config, dataset_length)
        self.prior_params = config.prior_params
        self.within_class_variance = config.within_class_variance
        self.mode = mode
        self.noisy_items = config.noisy_items
        self.num_labels = config.num_labels
        self.noise_rng = np.random.default_rng(config.random_seed+1000)
        
    def __getitem__(self, i):

        if self.mode == "icl":
            # sample rows from self.tasks with replacement
            tasks = self.rng.choice(self.tasks, size=self.context_length - 1, replace=True) #TODO: implement zipfian behavior

            # choose a target task uniformly from tasks
            target_task = np.copy(self.rng.choice(tasks))

        elif self.mode == "iwl":
            # sample a task uniformly from self.tasks
            target_task = np.copy(self.rng.choice(self.tasks)) #TODO: implement zipfian behavior

            # create a copy of self.tasks and remove target_task
            available_tasks = np.copy(self.tasks)
            mask = ~np.all(available_tasks == target_task, axis=1)
            available_tasks = available_tasks[mask]

            # sample tasks uniformly from available_tasks with replacement
            tasks = self.rng.choice(available_tasks, size=self.context_length - 1, replace=True)

        target_label = target_task[-1]

        # set the final label of target_task to -1
        target_task[-1] = -1

        # append target_task to pairs sequence
        pairs = np.concatenate((tasks, target_task.reshape(1, -1)), axis=0)

        if self.noisy_items:
            # create noise: sample from the same normal distribution as all items were sampled from 
            noise = self.noise_rng.multivariate_normal(
                    self.prior_params[0] * np.ones(self.num_dims),
                    self.prior_params[1] * np.eye(self.num_dims),
                    size=self.context_length,
                )

            # noise the items: multiply noise by within_class_var, add to items and normalize to get final items
            pairs[:, :-1] = (pairs[:, :-1] + noise * np.sqrt(self.within_class_variance)) / np.sqrt(1 + self.within_class_variance)

        # convert pairs to tensor
        pairs = torch.tensor(pairs, dtype=torch.float)

        # make labels tensor only the true label of the last pair
        label = torch.tensor(target_label, dtype=torch.long)

        return {
            "pairs": pairs,
            "labels": label,
        }  # labels is the name required by HF trainer

    @staticmethod
    def make_dataset(num_tasks, data_path, prior_params, rng, num_dims, num_labels, save=True):
        """
        Sample `num_tasks` item-label pairs in NumPy.
        Each pair contains an item sampled from a Gaussian, and a target label randomly sampled from {0,1,...,num_labels-1}.

        Args:
            num_tasks (int): Number of pairs to sample.
            prior_params (1D array): Parameters for the Gaussian.
            rng (np.random.Generator): Random number generator.
            num_dims (int): Number of dimensions for the Gaussian.
            num_labels (int): Number of labels.
        """
        # Sample items from a Gaussian
        items = rng.multivariate_normal(
            prior_params[0] * np.ones(num_dims),
            prior_params[1] * np.eye(num_dims),
            size=num_tasks,
        )

        # Sample labels from num_labels classes
        labels = rng.choice(np.arange(num_labels), size=num_tasks)

        # concatenate items and labels
        pairs = np.concatenate((items, labels.reshape(-1, 1)), axis=1)

        # Save arrays
        if save:
            np.savez(data_path, tasks=pairs)
        else:
            return pairs

    @staticmethod
    def compute_metrics(
        eval_pred,
        mode: str = "eval",
        return_all: bool = False,
        compute_nll=False,
        return_raw_loss = False,
    ):
        """
        Computes the cross entropy loss as an evaluation metric.
        """
        logits, labels = eval_pred
        # convert to tensors    
        logits = torch.as_tensor(logits, dtype=torch.float)
        labels = torch.as_tensor(labels, dtype=torch.long)

        # compute cross entropy loss
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        with torch.no_grad():
            metric = loss_fn(logits, labels).squeeze()

        if return_raw_loss:
            return torch.mean(metric)
        else:
            metric = metric.detach().cpu()
            mean_metric = torch.mean(metric)
            out = {f"{mode}_metric": mean_metric.item()}
            if compute_nll:
                out[f"{mode}_nll"] = mean_metric.item()  
                out[f"{mode}_all_nll"] = metric.tolist()
            if return_all:
                out[f"{mode}_all_metric"] = metric.tolist()
            # compute the accuracy
            # For binary classification, accuracy is the proportion of correct predictions
            predictions = (logits.argmax(dim=-1) == labels).float()  # Convert probabilities to predictions and check if they are correct
            accuracy = torch.sum(predictions).float() / len(predictions)  # Manually compute mean as float
            out[f"{mode}_accuracy"] = accuracy.item()  # Get accuracy value
            return out


#########################
# helper functions for training
#########################


def collect_outputs_and_labels(
    model,
    dataset,
    labels_key,
    inputs_key,
    batch_data=False,
    batch_size=16,
    collect_all_batches=False,
    num_batches_to_collect=1,
    data_collator=default_data_collator,
):
    """
    Run the model on the eval dataset and collect outputs and labels.

    Args:
        model: The model to evaluate.
        dataset: Dataset for evaluation.
        batch_size: Batch size for evaluation.
        num_dims: Number of dimensions for the task.
        labels_key: Key for labels in the dataset.
        inputs_key: Key for inputs in the dataset.
        num_batches_to_collect: Number of batches to collect.
        data_collator: Data collator for the dataset.
    Returns:
        A tuple (outputs, labels) containing tensors for all batches.
    """
    def eval_batch(batch):
        """
        A helper function to move batch to the device and forward pass. 

        returns:
            outputs: model outputs
            labels: labels
        """
        batch_on_device = {
            key: value.to(device)
            for key, value in batch.items()
            if torch.is_tensor(value)
        }
        labels = batch_on_device[labels_key]
        inputs = batch_on_device[inputs_key]

        # Forward pass
        if labels_required_for_inference:
            outputs = model(inputs, labels)
        else:
            outputs = model(inputs)

        # Collect outputs and labels
        if isinstance(outputs, dict):
            outputs = outputs[model_output_key]
        else:
            outputs = getattr(outputs, model_output_key)

        # remove irrelevant dimensions
        # check if model has start_token
        if hasattr(model, "config") and hasattr(model.config, "bos_token_id"):
            start_token_idx = model.config.bos_token_id
            # Remove the start token dimension from outputs
            # Create a mask that's False at the start_token_idx position and True elsewhere
            mask = torch.ones(outputs.shape[-1], dtype=torch.bool, device=outputs.device)
            mask[start_token_idx] = False
            # Use the mask to select all dimensions except the start token
            outputs = outputs[..., mask]                

        return outputs, labels 
    
    device = model.device
    all_outputs = []
    all_labels = []
    model_output_key = model.output_key
    labels_required_for_inference = model.labels_required_for_inference

    if batch_data:
        return eval_batch(dataset)
    else:
        # Create a dataloader for the evaluation dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=0,
        )

        with torch.no_grad():
            num_batches_evaluated = 0
            for batch in dataloader:
                outputs, labels = eval_batch(batch)
                all_outputs.append(outputs)
                all_labels.append(labels)
                num_batches_evaluated += 1
                if (
                    not collect_all_batches
                    and num_batches_evaluated >= num_batches_to_collect
                ):
                    break

        # Concatenate all outputs and labels into tensors on the device
        outputs_tensor = torch.cat(all_outputs, dim=0).to(model.device)
        labels_tensor = torch.cat(all_labels, dim=0).to(model.device)

        return outputs_tensor, labels_tensor


class StepMetricsCallback(TrainerCallback):
    def __init__(self, step_interval, trainer, config):
        self.trainer = trainer
        self.config = config
        self.step_interval = step_interval  # Interval at which to execute

    def on_step_end(self, args, state, control, **kwargs):
        """
        Called at the end of each training step. Triggers metric computation
        every `step_interval` steps.
        """
        # Check if it's time to execute based on step interval
        if (
            state.global_step % self.step_interval == 0
            and self.trainer.compute_metrics is not None
        ):
            trainer = self.trainer

            trainer.model.eval()  # Set model to evaluation mode

            # Perform evaluation and compute metrics
            train_outputs, train_labels = collect_outputs_and_labels(
                model=trainer.model,
                dataset=trainer.train_dataset,
                batch_size=trainer.args.per_device_eval_batch_size,
                labels_key=self.config.labels_key,
                inputs_key=self.config.inputs_key,
            )

            train_metrics = trainer.compute_metrics(
                (train_outputs, train_labels), mode="train"
            )

            eval_outputs, eval_labels = collect_outputs_and_labels(
                model=trainer.model,
                dataset=trainer.eval_dataset,
                batch_size=trainer.args.per_device_eval_batch_size,
                labels_key=self.config.labels_key,
                inputs_key=self.config.inputs_key,
            )
            eval_metrics = trainer.compute_metrics(
                (eval_outputs, eval_labels), mode="eval"
            )

            trainer.model.train()  # Return model to training mode

            # Save the step and metrics
            eval_metrics["step"] = state.global_step
            train_metrics["step"] = state.global_step

            # Log the metrics to trainer.state.log_history
            trainer.state.log_history.append(train_metrics)
            trainer.state.log_history.append(eval_metrics)

            # print the metrics
            print(
                f"Eval metrics at Step {state.global_step}: {eval_metrics['eval_metric']}"
            )
            print(
                f"Train metrics at Step {state.global_step}: {train_metrics['train_metric']}"
            )


class CustomSaveCallback(TrainerCallback):
    def __init__(self, steps_to_save, trainer):
        self.steps_to_save = set(steps_to_save)
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        """
        Save the model at specific intervals without saving unnecessary files.
        """
        trainer = self.trainer
        model = trainer.model
        if (
            state.global_step in self.steps_to_save
            or state.global_step == args.max_steps
        ):
            output_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(os.path.join(output_dir, "model"))
            print(f"Model saved at {os.path.join(output_dir, 'model')}")