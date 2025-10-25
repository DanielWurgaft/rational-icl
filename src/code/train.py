from transformers import TrainingArguments, Trainer
import os
import argparse
import torch
import numpy as np
import csv
from .models import init_transformer
from .config import Config
from .utils import (
    init_dataset,
    CustomSaveCallback,
    StepMetricsCallback,
)


def print_current_gpus():
    if torch.cuda.is_available():
        print("Current GPUs:")
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPUs available.")


def run_training(config):
    print_current_gpus()

    model = init_transformer(config)  # initialize model

    # Initialize the dataset
    train_dataset = init_dataset(
        config,
        "train",
        dataset_length=max(config.num_tasks, config.per_device_train_batch_size),
    )  # set dataset_length to allow batches greater than num_tasks

    eval_dataset = init_dataset(config, "eval")

    # define training args
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        logging_steps=config.logging_steps,
        max_steps=config.max_steps,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        seed=config.random_seed,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        lr_scheduler_type=config.lr_scheduler_type,
        lr_scheduler_kwargs=config.lr_scheduler_kwargs,
        save_strategy="no",  # Disable default saving
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=train_dataset.compute_metrics,
    )

    # add saving callback to the Trainer
    trainer.add_callback(
        CustomSaveCallback(
            config.save_steps,
            trainer=trainer,
        )
    )

    # Add the metrics callback to the Trainer
    trainer.add_callback(
        StepMetricsCallback(
            trainer=trainer, step_interval=config.eval_steps, config=config
        )
    )

    trainer.train()
    logs = trainer.state.log_history

    os.makedirs(config.run_results_dir, exist_ok=True)
    all_keys = set().union(*(d.keys() for d in logs))
    with open(
        os.path.join(config.run_results_dir, "logs.csv"),
        "w+",
        encoding="utf8",
        newline="",
    ) as file:
        csv_writer = csv.DictWriter(file, fieldnames=all_keys, restval="NA")
        csv_writer.writeheader()
        csv_writer.writerows(logs)


if __name__ == "__main__":
    # load cache directory from env
    from dotenv import load_dotenv

    load_dotenv()
    cache_dir = os.getenv("CACHE_DIR")

    # arg parse yaml file
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str)
    args = parser.parse_args()

    # if end in .yaml just run that file
    if args.experiment.endswith(".yaml"):
        config = Config.from_yaml(
            args.experiment, cache_dir=cache_dir, make_run_dirs=True
        )
        run_training(config)
    else:
        # iterate over each yaml in the folder
        yaml_dir = os.path.join(args.experiment, "yaml-configs")
        for file in os.listdir(yaml_dir):
            if file.endswith(".yaml"):
                try:
                    config = Config.from_yaml(
                        os.path.join(yaml_dir, file),
                        cache_dir=cache_dir,
                        make_run_dirs=True,
                    )
                    run_training(config)
                except FileExistsError as e:
                    print(e)
                    print("Skipping...")
                    continue
