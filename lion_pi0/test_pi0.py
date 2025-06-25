from vlaholo.datasets.lerobot_dataset import LeRobotDataset
import torch
from vlaholo.models.pretrained import PreTrainedConfig
from vlaholo.models.build_model import make_policy
from vlaholo.utils.utils import auto_select_torch_device
from loguru import logger
from huggingface_hub import login


def main():
    hf_token = "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"
    login(token=hf_token)
    
    # device = auto_select_torch_device()
    # Specify GPU index (0-7) here, for example GPU 2:
    gpu_id = 0  # Change this to your desired GPU (0-7)
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    # paligemma doesn't support bf16?
    # dtype = torch.float32
    logger.info(f"##info, device: {device}, dtype: {dtype}")

    # dataset_repo_id = "danaaubakirova/koch_test"
    dataset_repo_id = "/pfs/data/xiongxiao/robotwin2lerobot/block_hammer_beat"
    # ckpt_torch_dir = "/pfs/data/fgang/vla_holo/checkpoints/pi0"
    ckpt_torch_dir = "/pfs/data/fgang/outputs_models/pi0-1-20000/pretrained_model"
    
    # dataset = LeRobotDataset(repo_id, episodes=[0, 10, 11, 23])
    dataset = LeRobotDataset(dataset_repo_id, episodes=[0])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=1,
    )
  
    batch = next(iter(dataloader))
    # Print batch keys and their types
    
    # To device
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device=device, dtype=dtype)
    print(f'dataset.meta: {dataset.meta}')
    
    cfg = PreTrainedConfig.from_pretrained(ckpt_torch_dir, device=device)
    cfg.pretrained_path = ckpt_torch_dir
    policy = make_policy(cfg, ds_meta=dataset.meta)
    policy.to(dtype)
    print(policy)

    with torch.amp.autocast(device_type=device):
        benchmark_iters = 30
        for _ in range(benchmark_iters):
            print(batch)
            action = policy.select_action(batch)
            print("##info, action:", action.shape, action.dtype, action.device, action)


if __name__ == "__main__":
    main()
