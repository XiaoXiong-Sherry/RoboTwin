# Your_Policy - PI0 Model Deployment

This directory contains the deployment configuration for your PyTorch-based PI0 model in the RoboTwin framework.

## Files Overview

- `pi0_model.py`: Model wrapper class that interfaces your PI0 model with RoboTwin
- `deploy_policy.py`: Main deployment script that loads and uses your model
- `deploy_policy.yml`: Configuration file with model parameters
- `eval.sh`: Evaluation script for running tasks
- `conda_env.yaml`: Conda environment with required dependencies

## Setup

1. **Install Dependencies**:
   ```bash
   conda env create -f conda_env.yaml
   conda activate pi0_policy
   ```

2. **Verify Model Path**:
   Make sure your model is located at `my_policies/pretrained_model-pi0_beat/`

## Usage

### Basic Evaluation
```bash
cd policy/Your_Policy
./eval.sh <task_name> <task_config> <ckpt_setting> <seed> <gpu_id>
```

### Example
```bash
./eval.sh beat_block_hammer _config_template pi0_beat 0 0
```

### With Custom Parameters
```bash
./eval.sh beat_block_hammer _config_template pi0_beat 0 0 my_policies/pretrained_model-pi0_beat 2
```

## Parameters

- `task_name`: Name of the task to evaluate (e.g., `beat_block_hammer`)
- `task_config`: Task configuration file (e.g., `_config_template`)
- `ckpt_setting`: Checkpoint setting identifier (e.g., `pi0_beat`)
- `seed`: Random seed for reproducibility
- `gpu_id`: GPU ID to use (0-7)
- `model_path`: Path to your pretrained model (optional, defaults to `my_policies/pretrained_model-pi0_beat`)
- `pi0_step`: Number of action steps to predict per inference (optional, defaults to 1)

## Model Customization

You may need to modify `pi0_model.py` to match your model's exact input/output format:

1. **Input Format**: Adjust `_process_observation()` to match your model's expected input
2. **Output Format**: Modify `get_action()` to handle your model's output format
3. **Dataset Meta**: Update the dummy_meta in `_load_model()` with your actual dimensions

## Troubleshooting

1. **Import Errors**: Make sure all required packages are installed in your conda environment
2. **Model Loading**: Verify the model path and that all model files are present
3. **GPU Issues**: Check that the specified GPU is available and has sufficient memory
4. **Input Format**: Ensure the observation processing matches your model's expected format

## Notes

- The model wrapper assumes your PI0 model follows the VLAHolo interface
- You may need to adjust the image preprocessing and action post-processing based on your specific model
- The observation cache size can be adjusted in `update_obs()` method 