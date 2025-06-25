import torch
import numpy as np
import os
import sys
from vlaholo.models.pretrained import PreTrainedConfig
from vlaholo.models.build_model import make_policy
from vlaholo.utils.utils import auto_select_torch_device


class PI0Model:
    def __init__(self, model_path, gpu_id=0, pi0_step=1):
        """
        Initialize the PI0 model wrapper for RoboTwin
        
        Args:
            model_path: Path to the pretrained model directory
            gpu_id: GPU ID to use (0-7)
            pi0_step: Number of action steps to predict
        """
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.pi0_step = pi0_step
        
        # Setup device
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        
        print(f"Loading PI0 model from: {model_path}")
        print(f"Device: {self.device}, dtype: {self.dtype}")
        
        # Load model
        self._load_model()
        
        # Initialize observation cache
        self.obs_cache = []
        self.instruction = None
        
    def _load_model(self):
        """Load the pretrained PI0 model"""
        try:
            # Load config and model
            cfg = PreTrainedConfig.from_pretrained(self.model_path, device=self.device)
            cfg.pretrained_path = self.model_path
            
            # Create a dummy dataset meta for compatibility
            # You may need to adjust this based on your model's expected format
            dummy_meta = {
                "action_dim": 14,  # Adjust based on your action space
                "state_dim": 14,   # Adjust based on your state space
                "image_size": (480, 640, 3),  # Adjust based on your image size
            }
            
            self.policy = make_policy(cfg, ds_meta=dummy_meta)
            self.policy.to(dtype=self.dtype)
            self.policy.eval()
            
            print("PI0 model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def set_language(self, instruction):
        """Set the language instruction for the current episode"""
        self.instruction = instruction
        print(f"Instruction set: {instruction}")
    
    def update_obs(self, observation):
        """Update the observation cache with new observation"""
        # Convert observation to the format expected by your model
        # This is a placeholder - you need to adapt this to your model's input format
        processed_obs = self._process_observation(observation)
        self.obs_cache.append(processed_obs)
        
        # Keep only recent observations if needed
        if len(self.obs_cache) > 10:  # Adjust window size as needed
            self.obs_cache = self.obs_cache[-10:]
    
    def _process_observation(self, observation):
        """
        Process observation to match your model's expected input format
        This is a placeholder - you need to implement this based on your model
        """
        # Extract images from observation
        head_rgb = observation["observation"]["head_camera"]["rgb"]
        right_rgb = observation["observation"]["right_camera"]["rgb"] 
        left_rgb = observation["observation"]["left_camera"]["rgb"]
        
        # Extract joint state
        joint_state = observation["joint_action"]["vector"]
        
        # Convert to tensors and move to device
        images = torch.stack([
            torch.from_numpy(head_rgb).permute(2, 0, 1),  # HWC -> CHW
            torch.from_numpy(right_rgb).permute(2, 0, 1),
            torch.from_numpy(left_rgb).permute(2, 0, 1)
        ]).to(device=self.device, dtype=self.dtype)
        
        state = torch.from_numpy(joint_state).to(device=self.device, dtype=self.dtype)
        
        return {
            "images": images,
            "state": state,
            "instruction": self.instruction
        }
    
    def get_action(self):
        """Get action from the model based on current observation cache"""
        if not self.obs_cache:
            raise ValueError("No observations in cache. Call update_obs() first.")
        
        # Get the latest observation
        latest_obs = self.obs_cache[-1]
        
        # Create batch format expected by your model
        batch = {
            "image": latest_obs["images"].unsqueeze(0),  # Add batch dimension
            "state": latest_obs["state"].unsqueeze(0),
            "instruction": latest_obs["instruction"]
        }
        
        # Get action from model
        with torch.no_grad():
            action = self.policy.select_action(batch)
        
        # Convert to numpy and return
        action_np = action.cpu().numpy()
        
        # Return pi0_step actions
        if len(action_np.shape) == 3:  # [batch, horizon, action_dim]
            return action_np[0, :self.pi0_step]  # Remove batch dim, take first pi0_step
        else:  # [batch, action_dim]
            return action_np[0:1]  # Remove batch dim
    
    def reset_obsrvationwindows(self):
        """Reset the observation cache and instruction"""
        self.obs_cache = []
        self.instruction = None
        print("Observation cache and instruction reset") 