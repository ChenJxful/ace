import sys
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from models.networks import Runner
from data.data_loader import create_train_val_loaders


@hydra.main(version_base=None, config_path="configs", config_name="ACE")
def main(cfg: DictConfig) -> None:
    device = torch.device(cfg.device)
    print('\033[1;32m' + f"Using device: {device}" + '\033[0m')
    kwargs = {"num_workers": cfg.num_workers, "pin_memory": True} if cfg.device == 'cuda' else {"num_workers": 0}

    # 初始化数据集
    print(f"Dataset: {cfg.dataset}")
    train_loader, val_loader = create_train_val_loaders(
        cfg.dataset, cfg.batch_size, cfg.val_split, kwargs)
    print(f"Training set size: {len(train_loader.dataset)}\nValidation set size: {len(val_loader.dataset)}")

    # 初始化 Runner
    runner = Runner(cfg)
    # 开始训练
    runner.train(train_loader, val_loader)


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    sys.argv.append(f'+device={"cuda" if use_cuda else "cpu"}')  # 写在这里可以自动写入 hydra 备份的 config 文件
    # sys.argv.append('batch_size=3')
    main()
