import os
from tqdm import tqdm
import numpy as np
from .utils import make_dataset
import argparse
from .config import Config


def main(args):
    for num_dims in tqdm(args.num_dims):
        for i in tqdm(
            range(int(np.log2(args.max_num_tasks)), -1, -1), desc="Creating datasets"
        ):
            config = Config(
                setting=args.setting,
                num_dims=num_dims,
                num_tasks=2**i,
                num_eval_tasks=args.num_eval_tasks,
                random_seed=args.random_seed,
                cache_dir=args.cache_dir,
            )
            make_dataset(config, mode="train")

        if args.make_eval_data:
            print("making eval data...")
            make_dataset(config, mode="eval")

        if len(args.additional_num_tasks) > 0:
            print("making additional datasets...")
            for num_tasks in tqdm(
                args.additional_num_tasks, desc="making additional datasets"
            ):
                config = Config(
                    setting=args.setting,
                    num_dims=num_dims,
                    num_tasks=num_tasks,
                    num_eval_tasks=args.num_eval_tasks,
                    random_seed=args.random_seed,
                    cache_dir=args.cache_dir,
                )

            make_dataset(
                config.train_datapath, num_tasks=config.num_tasks, config=config
            )


if __name__ == "__main__":
    # load env variables
    from dotenv import load_dotenv

    load_dotenv()
    cache_dir = os.getenv("CACHE_DIR")

    # set up variables for config based on command line arguments
    # read command line arguments
    parser = argparse.ArgumentParser()
    # dataset type
    parser.add_argument("--setting", type=str, default="linear-regression")
    # list of number of dimensions to run
    parser.add_argument("--num_dims", type=int, nargs="+", default=[8])
    # set number of eval tasks
    parser.add_argument("--num_eval_tasks", type=int, default=500)
    # set max number of tasks
    parser.add_argument("--max_num_tasks", type=int, default=2**16)
    # set additional tasks to add
    parser.add_argument("--additional_num_tasks", type=int, nargs="+", default=[])
    # set random seeds
    parser.add_argument("--random_seed", type=int, default=1)
    # set make eval data
    parser.add_argument("--make_eval_data", type=bool, default=True)

    args = parser.parse_args()

    # add cache dir as an arg
    args.cache_dir = cache_dir

    main(args)
