# import packages and module here
import numpy as np
import torch
import os
import sys

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)

from pi0_model import PI0Model


def encode_obs(observation):  # Post-Process Observation
    """Post-process observation - keeping original format for now"""
    obs = observation
    # ...
    return obs


def get_model(usr_args):  # from deploy_policy.yml and eval.sh (overrides)
    """Load the PI0 model based on arguments from deploy_policy.yml and eval.sh"""
    model_path = usr_args["model_path"]
    gpu_id = usr_args.get("gpu_id", 0)
    pi0_step = usr_args.get("pi0_step", 1)
    
    return PI0Model(model_path=model_path, gpu_id=gpu_id, pi0_step=pi0_step)


def eval(TASK_ENV, model, observation):
    """
    Main evaluation function that follows the RoboTwin interface
    """
    obs = encode_obs(observation)  # Post-Process Observation
    instruction = TASK_ENV.get_instruction()

    if len(model.obs_cache) == 0:  # Force an update of the observation at the first frame
        model.set_language(instruction)
        model.update_obs(obs)

    actions = model.get_action()  # Get Action according to observation chunk

    for action in actions:  # Execute each step of the action
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)  # Update Observation


def reset_model(
        model):  # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    """Clean the model cache at the beginning of every evaluation episode"""
    model.reset_obsrvationwindows()
