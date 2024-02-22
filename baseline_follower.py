"""Follower deployed as a proof of concept in initial (pilot) experiment."""

import logging
import pathlib

from typing import List
from dataclasses import dataclass

from config import SamplingStrategy, EnsemblingStrategy
from models import FollowerEnsemble, DecisionTransformerFromPath
from cb2_tensors import TensorFromCb2Actions, Cb2ActionsFromTensor, PadActionTensor

from cb2game.agents.agent import Agent, Role
from cb2game.pyclient.game_endpoint import Action, GameState
from mashumaro.mixins.json import DataClassJSONMixin

logger = logging.getLogger(__name__)

BASE_MODEL_PATH = pathlib.Path("experiments/pretraining/deployment_models")

@dataclass
class BaselineFollowerConfig(DataClassJSONMixin):
    """Configuration for demo follower bot."""
    # Strategy to follow for sampling follower actions.
    sampling_strategy: SamplingStrategy = SamplingStrategy.BOLTZMANN_MULTIPLICATION
    ensembling_strategy: EnsemblingStrategy = EnsemblingStrategy.AVERAGE
    # If more than one model is used, ensembling is enabled.
    model_paths: List[pathlib.Path] = [
        BASE_MODEL_PATH / "run_1",
        BASE_MODEL_PATH / "run_2",
        BASE_MODEL_PATH / "run_3",
    ]

class BaselineFollower(Agent):
    def __init__(self, config: BaselineFollowerConfig):
        # Initialize your agent here.
        self.config = config

        # Load model(s)
        self.models = []
        for model_path in config.model_paths:
            model = DecisionTransformerFromPath(model_path)
            self.models.append(model)
        
        # If only one model was loaded, set it as the follower.
        # Otherwise, create an ensemble of models and set it as the follower.
        if len(self.follower_models) == 1:
            self.follower = self.models[0]
        else:
            self.follower = FollowerEnsemble(self.models, config.ensembling_strategy)
        
        # Keep a history of states, actions, and timesteps.
        self.states = []
        self.actions = []
        self.timesteps = []

    # OVERRIDES role
    def role(self) -> Role:
        # This function should be a one-liner.
        # Return the role of your agent (Role.LEADER or Role.FOLLOWER).
        raise Role.FOLLOWER

    # OVERRIDES choose_action
    def choose_action(self, game_state: GameState, action_mask=None) -> Action:
        # Create a Ax1 tensor containing the action codes, where A is the number
        # of actions taken so far.  Adds a padding action to the end of the
        # tensor. See cb2_tensors.py for more.
        action_tensor = PadActionTensor(TensorFromCb2Actions(self.actions))

        




