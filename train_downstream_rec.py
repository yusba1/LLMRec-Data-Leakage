import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn

import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank, init_distributed_mode
from minigpt4.common.logger import setup_logger
from minigpt4.common.registry import registry
from minigpt4.common.utils import now

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
from torch.distributed.elastic.multiprocessing.errors import *

def parse_args():
    parser = argparse.ArgumentParser(description="Training Downstream Recommendation Model")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--options", nargs="+", help="override some settings in the used config")
    return parser.parse_args()

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def get_runner_class(cfg):
    return registry.get_runner_class(cfg.run_cfg.get("runner", "rec_runner_base"))

@record
def main():
    job_id = now()
    cfg = Config(parse_args())
    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    setup_logger()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    
    # 获取数据路径并计算 user/item num
    try:
        # 尝试获取 movie_ood (ML-1M) 或 amazon_ood
        data_dir = cfg.datasets_cfg.get("movie_ood", cfg.datasets_cfg.get("amazon_ood")).path
    except AttributeError:
        # Fallback 逻辑
        data_name = list(datasets.keys())[0]
        data_dir = cfg.datasets_cfg[data_name].path

    print(f"Data directory: {data_dir}")
    
    # 加载数据以计算统计信息
    # 注意：这里假设目录包含 .pkl 文件
    train_ = pd.read_pickle(f"{data_dir}/train_ood2.pkl")
    valid_ = pd.read_pickle(f"{data_dir}/valid_ood2.pkl")
    test_ = pd.read_pickle(f"{data_dir}/test_ood2.pkl")
    
    user_num = max(train_.uid.max(), valid_.uid.max(), test_.uid.max()) + 1
    item_num = max(train_.iid.max(), valid_.iid.max(), test_.iid.max()) + 1

    cfg.model_cfg.rec_config.user_num = int(user_num)
    cfg.model_cfg.rec_config.item_num = int(item_num)
    
    cfg.pretty_print()

    model = task.build_model(cfg)
    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()

if __name__ == "__main__":
    main()
