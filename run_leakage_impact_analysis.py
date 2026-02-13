#!/usr/bin/env python3
"""
Benchmark Leakage Experiment: Main Script
此脚本用于复现论文 "Benchmark Leakage Trap" 中的核心实验。
它模拟了数据泄漏场景，并评估不同方法在泄漏前后的性能变化。
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import argparse
import logging
import subprocess
import re
import time
from pathlib import Path
from typing import Dict

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("experiment.log")]
)
logger = logging.getLogger(__name__)

class LeakageExperiment:
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_path = Path(config['data_path'])
        
    def load_dataset(self):
        """加载原始数据集"""
        logger.info(f"Loading dataset from {self.data_path}")
        try:
            train_data = pd.read_pickle(self.data_path / "train_ood2.pkl")
            valid_data = pd.read_pickle(self.data_path / "valid_ood2.pkl")
            test_data = pd.read_pickle(self.data_path / "test_ood2.pkl")
            return train_data, valid_data, test_data
        except FileNotFoundError:
            logger.error(f"Dataset files not found in {self.data_path}. Please check the path.")
            sys.exit(1)

    def simulate_leakage(self, train_data, valid_data, test_data, ratio: float) -> Path:
        """模拟数据泄漏：从Train+Valid+Test随机抽取ratio比例数据混合进训练集"""
        if ratio <= 0:
            logger.info("No leakage simulation (Baseline). Using original data.")
            return self.data_path

        logger.info(f"Simulating {ratio*100}% data leakage...")
        
        # 1. 合并所有数据
        all_data = pd.concat([train_data, valid_data, test_data], ignore_index=True)
        
        # 2. 随机抽取泄漏数据
        n_leak = int(len(all_data) * ratio)
        np.random.seed(42) 
        leaked_indices = np.random.choice(len(all_data), size=n_leak, replace=False)
        leaked_data = all_data.iloc[leaked_indices].copy()
        
        # 3. 构建新的数据集目录
        leaked_dataset_dir = self.output_dir / f"dataset_leak_{ratio}"
        leaked_dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存：训练集被污染，验证集和测试集保持不变
        leaked_data.to_pickle(leaked_dataset_dir / "train_ood2.pkl")
        valid_data.to_pickle(leaked_dataset_dir / "valid_ood2.pkl")
        test_data.to_pickle(leaked_dataset_dir / "test_ood2.pkl")
        
        return leaked_dataset_dir

    def get_config_template(self, method_name):
        """根据方法名获取对应的 YAML 配置文件模板"""
        templates = {
            'CoLLM-MF': 'train_configs/collm_pretrain_mf_ood.yaml',
            'CoLLM-DIN': 'train_configs/collm_pretrain_din_ood_cc.yaml',
            'TALLRec': 'train_configs/personlized_prompt_ood_cc.yaml', 
        }
        
        if self.config['dataset_name'] == 'amazon-book':
            templates = {k: v.replace(".yaml", "_amazon.yaml") if v.endswith(".yaml") else v for k, v in templates.items()}

        path = templates.get(method_name)
        if not path:
             return None

        if os.path.exists(path):
            return path
        elif os.path.exists(os.path.join(os.getcwd(), path)):
             return os.path.join(os.getcwd(), path)
        return None

    def run_training(self, method_name, dataset_path, leakage_ratio):
        """运行单次训练评估"""
        template_path = self.get_config_template(method_name)
        if not template_path:
            logger.error(f"Skipping {method_name}: Config not found.")
            return None

        # 1. 读取并修改 YAML 配置
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to read config: {e}")
            return None
            
        dataset_keys = list(yaml_config.get('datasets', {}).keys())
        if not dataset_keys:
            return None
        target_dataset_key = dataset_keys[0]
        
        yaml_config['datasets'][target_dataset_key]['path'] = str(dataset_path) + "/" 
        
        run_id = f"{method_name}_leak{leakage_ratio}_{int(time.time())}"
        yaml_config['run']['output_dir'] = str(self.output_dir / "checkpoints" / run_id)
        
        temp_config_path = self.output_dir / f"config_{run_id}.yaml"
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_config, f)
            
        # 2. 调用训练脚本
        train_script = "train_downstream_rec.py"
        cmd = [sys.executable, train_script, "--cfg-path", str(temp_config_path)]
        
        logger.info(f"Starting training for {method_name} (Leak {leakage_ratio})...")
        try:
            result = subprocess.run(cmd, check=True, text=True, capture_output=True)
            output = result.stdout
            
            # 3. 解析结果 (Regex 查找最后的 AUC/UAUC)
            auc = 0.0
            uauc = 0.0
            auc_matches = re.findall(r"\*\*\*auc:\s*([\d\.]+)", output)
            uauc_matches = re.findall(r"\*\*\*uauc:\s*([\d\.]+)", output)
            
            if auc_matches: auc = float(auc_matches[-1])
            if uauc_matches: uauc = float(uauc_matches[-1])
                
            logger.info(f"Finished {method_name} (Leak {leakage_ratio}): AUC={auc}, UAUC={uauc}")
            return {'AUC': auc, 'UAUC': uauc}
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed for {method_name}. Error:\n{e.stderr}")
            return None

    def run(self):
        logger.info("Starting Experiment Run...")
        train_data, valid_data, test_data = self.load_dataset()
        results_summary = []

        for ratio in self.config['leakage_ratios']:
            current_data_path = self.simulate_leakage(train_data, valid_data, test_data, ratio)
            for method in self.config['methods']:
                res = self.run_training(method, current_data_path, ratio)
                if res:
                    results_summary.append({
                        'Method': method, 'Leakage': ratio, 'AUC': res['AUC'], 'UAUC': res['UAUC']
                    })
        
        df = pd.DataFrame(results_summary)
        csv_path = self.output_dir / "final_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Experiment completed. Results saved to {csv_path}")
        print("\nFinal Results:")
        print(df)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ml-1m", choices=["ml-1m", "amazon-book"])
    parser.add_argument("--data-path", default="dataset/ml-1m", help="Path to original dataset")
    parser.add_argument("--methods", nargs="+", default=["CoLLM-MF", "TALLRec"], help="Methods to evaluate")
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.0, 0.1], help="Leakage ratios")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    
    args = parser.parse_args()
    exp = LeakageExperiment(vars(args))
    exp.run()

if __name__ == "__main__":
    main()
