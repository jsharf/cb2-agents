""" This file contains utilities for converting from CB2 data structures to pytorch tensors.

We maintain a set of enums that mirror the CB2 enums, but provide a more
convenient interface for pytorch, and allow for more flexibility in model training.

"""
from enum import Enum
from typing import List

import torch

import cb2game.pyclient.game_endpoint as game_endpoint
import torch.nn.functional as F

from cb2game.game_endpoint import GameState
from cb2game.server.messages.map_update import MapUpdate
from cb2game.server.messages.prop import Prop, PropType
from cb2game.server.card import Card

MAP_SIZE = 25

class ActionCode(Enum):
    MF = 0
    MB = 1
    TR = 2
    TL = 3
    DONE = 4
    PAD = 5
    END_TURN = 6

    @staticmethod
    def FromCb2ActionCode(action_code: game_endpoint.Action.ActionCode):
        if action_code == game_endpoint.Action.ActionCode.FORWARDS:
            return ActionCode.MF
        elif action_code == game_endpoint.Action.ActionCode.BACKWARDS:
            return ActionCode.MB
        elif action_code == game_endpoint.Action.ActionCode.TURN_RIGHT:
            return ActionCode.TR
        elif action_code == game_endpoint.Action.ActionCode.TURN_LEFT:
            return ActionCode.TL
        elif action_code == game_endpoint.Action.ActionCode.DONE:
            return ActionCode.DONE
        elif action_code == game_endpoint.Action.ActionCode.END_TURN:
            return ActionCode.END_TURN
        else:
            raise ValueError(f"Cannot translate action code: {action_code}")
        
    def cb2_action(self) -> game_endpoint.Action.ActionCode:
        if self == ActionCode.MF:
            return game_endpoint.Action.ActionCode.FORWARDS
        elif self == ActionCode.MB:
            return game_endpoint.Action.ActionCode.BACKWARDS
        elif self == ActionCode.TR:
            return game_endpoint.Action.ActionCode.TURN_RIGHT
        elif self == ActionCode.TL:
            return game_endpoint.Action.ActionCode.TURN_LEFT
        elif self == ActionCode.DONE:
            return game_endpoint.Action.ActionCode.DONE
        elif self == ActionCode.END_TURN:
            return game_endpoint.Action.ActionCode.END_TURN
        else:
            raise ValueError(f"Cannot translate action code: {self}")

class PaddingCode(Enum):
    PAD = 0

class RotationCode(Enum):
    ROT_0 = 1
    ROT_60 = 2
    ROT_120 = 3
    ROT_180 = 4
    ROT_240 = 5
    ROT_300 = 6

class LayerCode(Enum):
    LAYER_0 = 7
    LAYER_1 = 8
    LAYER_2 = 9

class TileCode(Enum):
     # Tile contents
    GROUND_TILE = 10
    ROCKY = 11
    STONES = 12
    TREES = 13
    HOUSES = 14
    STREETLIGHT = 15
    PATH = 16
    WATER = 17
    MOUNTAIN = 18
    RAMP = 19

    # Stone types
    STONE_TYPE_0 = 20
    STONE_TYPE_1 = 21
    STONE_TYPE_2 = 22
    STONE_TYPE_3 = 23

    # Tree types
    TREE_DEFAULT = 24
    TREE_BROWN = 25
    TREE_DARKGREEN = 26
    TREE_SOLIDBROWN = 27
    TREE_TREES = 28
    TREE_TREES_2 = 29
    TREE_FOREST = 30

    # House properties
    HOUSE_TRIPLE = 31
    HOUSE_COLOR_DEFAULT = 32
    HOUSE_COLOR_RED = 33
    HOUSE_COLOR_BLUE = 34
    HOUSE_COLOR_GREEN = 35
    HOUSE_COLOR_ORANGE = 36
    HOUSE_COLOR_PINK = 37
    HOUSE_COLOR_YELLOW = 38
    
    # Snow
    SNOW = 39

def TensorFromCb2Actions(actions: List[game_endpoint.Action]) -> torch.Tensor:
    action_codes = [action.action_code() for action in actions]
    return [ActionCode.FromCb2ActionCode(code) for code in action_codes]

def PadActionTensor(action_tensor: torch.Tensor) -> torch.Tensor:
    return F.pad(input=action_tensor, pad=(0, 1), mode='constant', value=ActionCode.PAD)

def Cb2ActionsFromTensor(action_tensor: torch.Tensor) -> List[game_endpoint.Action.ActionCode]:
    return [ActionCode(action).cb2_action() for action in action_tensor if action != ActionCode.PAD]

def TensorFromCb2State(state: GameState) -> torch.Tensor:
    """Converts a CB2 state into a tensor.

    Args:
        state: A CB2 state.

    Returns:
        A tensor representing the CB2 state.
    """
