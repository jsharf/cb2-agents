"""This file implements the models for all the bots in this repo.

DecisionTransformer is adapted from https://github.com/Louiealbp/TDT
"""
from baseline_follower import BaselineFollowerConfig

import pathlib

from typing import List

class FollowerModel(object):
    ...

# File: follower_transformers.py
# ------------------------------
# Contains transformer architectures for the follower model. Adapted from
# the Text Decision Transformer, https://github.com/Louiealbp/TDT

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

from dataclasses import dataclass
from enum import Enum

import transformers
from transformers import GPT2Model, GPT2Tokenizer,  BertModel, BertTokenizer

from constants import GPT_EMB_DIM, TEXT_PAD_IDX, CNN_EMB_DIM, NUM_PROPERTIES, STATE_PAD_IDX, TORCH_DEVICE
from data_utils.data_classes import ActionEnums
from models.hex_conv import HexConv

import ray

from config import SamplingStrategy, EnsemblingStrategy

class ConvolutionType(Enum):
    SHALLOW = 1
    DEEP = 2
    RESNET = 3

@dataclass
class LegacyDecisionTransformerConfig:
    act_dim: int
    state_embed_dim: int
    num_layers: int
    cnn_option: int
    max_ep_len: int
    use_timesteps: bool
    freeze_embeddings: bool
    inference_temperature: float
    sampling_strat: str
    
@dataclass
class DecisionTransformerConfig:
    # Number of layers to use in the GPT2 model. If -1, use all layers.
    gpt2_layer_prefix: int = -1
    inference_temperature: float = 1.0
    # Maximum number of actions that the network can take.
    number_of_actions: int
    state_embedding_dimension: int
    convolution_type: ConvolutionType
    use_timesteps: bool = True
    sampling_strategy: SamplingStrategy = SamplingStrategy.ARGMAX
    max_episode_length: int = 4096
    # Applies Tanh() activation function to action prediction.
    action_tanh: bool = False
    layer_normalization: bool = True
    freeze_embeddings: bool = True
    embedding_size: int = GPT_EMB_DIM

class DecisionTransformer(nn.Module):

    """
    This model uses GPT to model (text_1, ..., text_n, state_1, action_1, state_2, action_2, ...)
    """

    def __init__(
            self,
            config: DecisionTransformerConfig,
            device: torch.device
    ):
        super().__init__()

        # Class config & device setup.
        self.config = config
        self.device = device

        # Used to cache hidden state.
        self.past_output = None

        # GPT2 Setup.
        gpt_config = transformers.GPT2Config(
            n_embd=self.config.embedding_size,
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.transformer = GPT2Model.from_pretrained('gpt2')
        if self.config.gpt2_layer_prefix != -1:
            # Prune GPT2
            self.transformer.config.n_layer = self.config.gpt2_layer_prefix
            self.transformer.h = self.transformer.h[:self.config.gpt2_layer_prefix]

        # Initialize embedding layers.
        self.embed_timestep = nn.Embedding(self.config.max_episode_length+1,
                                           self.config.embedding_size,
                                           padding_idx=self.config.max_episode_length)
        nn.init.xavier_normal_(self.embed_timestep.weight)
        state_convolution_layer = StateConvolutionFromType(self.config.convolution_type)
        self.embed_state = nn.Sequential(*[StateEmbedder(NUM_PROPERTIES, self.config.state_embedding_dimension),
                                        state_convolution_layer(self.config.state_embedding_dimension)])
        self.embed_state[0].initialize_weights()
        self.embed_action = nn.Sequential(nn.Embedding(self.act_dim, config.n_embd, padding_idx=ActionEnums['PAD'].value),
                                          nn.Tanh()) # TODO: Why am I using tanh here?
        nn.init.xavier_normal_(self.embed_action[0].weight)

        # The added words are for the PAD and the SEP tokens
        self.construct_text_embeddings(gpt_config.vocab_size + 1, gpt_config.n_embd, TEXT_PAD_IDX)
        self.embed_ln = nn.Identity()
        if self.config.layer_normalization:
            self.embed_ln = nn.LayerNorm(self.config.embedding_size)

        # note: we don't predict states or returns for the paper
        self.predict_action = nn.Sequential(
            *([nn.Linear(config.n_embd, self.act_dim)] + ([nn.Tanh()] if self.config.action_tanh else []))
        )
        nn.init.kaiming_normal_(self.predict_action[0].weight)
        nn.init.constant_(self.predict_action[0].bias, 0)

    def construct_text_embeddings(self, vocab_size, n_embd, pad_idx):
        assert(vocab_size - 1 == pad_idx)

        # Construct normally initialilzed text embeddings
        self.embed_text = nn.Embedding(vocab_size, n_embd, padding_idx=pad_idx)

        # Copy over GPT-2 embedding weights
        with torch.no_grad():
            self.embed_text.weight[:-1, :] = self.transformer.wte.weight

        # Freeze embeddings if needed
        if self.config.freeze_embeddings:
            self.embed_text.weight.requires_grad = not self.config.freeze_embeddings

    def forward(self, states, actions, timesteps, text_conditioning,
                pos_idx, attention_mask, text_attention_mask, action_mask):

        batch_size, seq_length = states.shape[0], states.shape[1]

        # Pushing vectors to device.
        states = states.to(self.device)
        actions = actions.to(self.device)
        timesteps = timesteps.to(self.device)
        text_conditioning = text_conditioning.to(self.device)
        pos_idx = pos_idx.to(self.device)
        attention_mask = attention_mask.to(self.device)
        text_attention_mask = text_attention_mask.to(self.device)
        action_mask = action_mask.to(self.device)

        # Embed each modality
        text_embeddings = self.embed_text(text_conditioning) # B x T' X hidden
        state_embeddings = self.embed_state(states) # B x T x hidden
        action_embeddings = self.embed_action(actions) # B x T x hidden
        text_context_size = text_embeddings.shape[1]

        # time embeddings are treated similar to positional embeddings
        if self.use_timesteps:
            time_embeddings = self.embed_timestep(timesteps) # B x T x hidden
            state_embeddings = state_embeddings + time_embeddings
            action_embeddings = action_embeddings + time_embeddings

        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 2*seq_length, self.hidden_size)


        # adds the text-conditioning to the front
        # new view is (T_1, T_2, T_3, ..., s_1, a_1, etc.)
        stacked_inputs = torch.cat([text_embeddings, stacked_inputs], dim = 1)

        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 2*seq_length)

        stacked_attention_mask = torch.cat([text_attention_mask, stacked_attention_mask], dim = 1) # B x (T' + 2T)


        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            position_ids=pos_idx
        )
        x = transformer_outputs['last_hidden_state']
        x = x[:, text_context_size:]

        # reshape x so that the second dimension corresponds to
        # predicting actions (0), or states (1)
        x = x.reshape(batch_size, seq_length, 2, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        action_preds = self.predict_action(x[:,0])

        # Apply the action mask on the predictions
        action_preds.masked_fill_(action_mask, -float('inf'))

        return action_preds

    def compute_probabilities(self, states, actions, timesteps, text_conditioning,
                              pos_idx, attention_mask, text_attention_mask, action_mask):
        # Assume an output of BxTx6
        action_logits = self.forward(states, actions, timesteps, text_conditioning,
                                     pos_idx, attention_mask, text_attention_mask, action_mask)
        action_probs = F.softmax(action_logits, dim=2)
        return action_probs

    def sample_action(self, states, actions, timesteps, text_conditioning,
                      pos_idx, attention_mask, text_attention_mask,
                      action_mask):
        # Get the logits
        final_prob = self.compute_logits_with_past(states, actions, timesteps, text_conditioning,
                                                   pos_idx, attention_mask, text_attention_mask,
                                                   action_mask)        

        if self.config.sampling_strategy == SamplingStrategy.SOFTMAX:
            m = Categorical(F.softmax(final_prob, dim=1))
            action = m.sample()
        elif self.config.sampling_strategy == SamplingStrategy.ARGMAX:
            action = torch.argmax(final_prob, dim=1)
        else:
            assert(False, "Input an invalid sampling strategy")

        return action

    def compute_logits_with_past(self, states, actions, timesteps, text_conditioning, pos_idx,
                                 attention_mask, text_attention_mask, action_mask):
        # Pushing to cuda
        states = states.to(self.device)
        actions = actions.to(self.device)
        timesteps = timesteps.to(self.device)
        text_conditioning = text_conditioning.to(self.device)
        pos_idx = pos_idx.to(self.device)
        attention_mask = attention_mask.to(self.device)
        text_attention_mask = text_attention_mask.to(self.device)
        action_mask = action_mask.to(self.device)

        # Two cases requiring different approaches
        if self.past_output is None:
            a_probs, self.past_output = self.rollout_first_sample(states, actions, timesteps, text_conditioning,
                                                                  pos_idx, attention_mask, text_attention_mask)
        else:
            a_probs, self.past_output = self.rollout_with_past(states, actions, timesteps, text_conditioning,
                                                               pos_idx, attention_mask, text_attention_mask)

        # Determine the final probabilities
        # Assume that the PAD token is the final index
        final_prob = a_probs[:, -1, :-1] / self.inference_temperature
        final_prob.masked_fill_(action_mask, -float('inf'))
        final_prob = final_prob.cpu()

        return final_prob
        

    def rollout_first_sample(self, states, actions, timesteps, text_conditioning, pos_idx,
                         attention_mask, text_attention_mask):
        batch_size, seq_length = states.shape[0], states.shape[1]            

        # Embed each modality
        text_embeddings = self.embed_text(text_conditioning)
        state_embeddings = self.embed_state(states)

        text_context_size = text_embeddings.shape[1]
        pos_idx = pos_idx[:, :-1] # B x T' + 1

        # time embeddings are treated similar to positional embeddings
        if self.use_timesteps:
            time_embeddings = self.embed_timestep(timesteps)
            state_embeddings = state_embeddings + time_embeddings

        # Add text encoding to the front
        stacked_inputs = torch.cat([text_embeddings, state_embeddings], dim=1)
        stacked_inputs = self.embed_ln(stacked_inputs)
        stacked_attention_mask = torch.cat([text_attention_mask, attention_mask], dim=1)

        transformer_outputs = self.transformer(
            inputs_embeds = stacked_inputs,
            attention_mask=stacked_attention_mask,
            position_ids = pos_idx,
            use_cache = True            
        )
        x = transformer_outputs['last_hidden_state']
        x = x[:, text_context_size:] # B x 1 x hidden
        past_values = transformer_outputs['past_key_values']

        action_preds = self.predict_action(x)
        return action_preds, past_values

    def rollout_with_past(self, states, actions, timesteps, text_conditioning, pos_idx,
                          attention_mask, text_attention_mask):
        batch_size, seq_length = states.shape[0], states.shape[1]

        # Embed the state and the action
        state_embeddings = self.embed_state(states[:, -1:, :]) # Bx1xhidden
        action_embeddings = self.embed_action(actions[:, -2:-1]) # Bx1xhidden
        pos_idx = pos_idx[:, -3:-1]

        # Add time embeddings
        if self.use_timesteps:
            time_embeddings = self.embed_timestep(timesteps[:, -2:]) # Bx2xhidden
            state_embeddings = state_embeddings + time_embeddings[:, -1:, :]
            action_embeddings = action_embeddings + time_embeddings[:, -2:-1, :]

        stacked_inputs = torch.stack(
            (action_embeddings, state_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 2, self.hidden_size) # Bx2xhidden
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Get the complete attention masks
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 2*timesteps.shape[1])[:, :-1] # Exclude the unpredicted action
        stacked_attention_mask = torch.cat([text_attention_mask, stacked_attention_mask], dim=1)

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            position_ids=pos_idx,
            past_key_values=self.past_output,
            use_cache=True
        )
        x = transformer_outputs['last_hidden_state']
        past_values = transformer_outputs['past_key_values']

        x = x.reshape(batch_size, 1, 2, self.hidden_size).permute(0, 2, 1, 3)
        action_preds = self.predict_action(x[:, 1])

        return action_preds, past_values

    def reset_past_output(self):
        self.past_output = None

    def has_past_output(self):
        return self.past_output is not None

def StateConvolutionFromType(convolution_type: ConvolutionType):
    '''Selects the appropriate convolution layer for the state embedding based on the provided strategy.
    '''
    if convolution_type == ConvolutionType.SHALLOW:
        return ShallowFollowerStateCNN
    elif convolution_type == ConvolutionType.DEEP:
        return DeeperFollowerStateCNN
    elif convolution_type == ConvolutionType.RESNET:
        return ResNetFollowerStateCNN
    else:
        raise ValueError("Invalid convolution type: {}".format(convolution_type))

class StateEmbedder(nn.Module):
    '''
    Embeds the property indices held in the state representation tensor
    into P dimensional vectors, then sums the vectors up for each tile.

    Also adds a convolution layer to the end of the embedding.
    '''

    def __init__(self, num_properties, emb_dim, convolution_type: ConvolutionType, in_channels: int):
        super().__init__()
        self.embed = nn.Embedding(num_properties, emb_dim, padding_idx=STATE_PAD_IDX)

    def initialize_weights(self):
        nn.init.xavier_normal_(self.embed.weight)

    def forward(self, x):
        embed_x = self.embed(x) # BxTxPxHxWxemb_dim
        embed_x = torch.sum(embed_x, dim=2) #BxTxHxWxemb_dim
        return embed_x.permute(0, 1, 4, 2, 3)

class ShallowFollowerStateCNN(nn.Module):
    '''
    A 3 layer CNN using HexaConv as its convolution layer. It expects
    a Px15x15 output, then processes it with a kernel size of 7 and then
    with two layers of kernel size 5.
    '''

    def __init__(self, in_channels=CNN_EMB_DIM, out_channels=GPT_EMB_DIM):
        super().__init__()
        self.l1 = nn.Sequential(*[HexConv(in_channels, in_channels*2, 7), nn.LeakyReLU(),
                                 nn.InstanceNorm2d(in_channels*2)])
        self.l2 = nn.Sequential(*[HexConv(in_channels*2, in_channels*4, 5), nn.LeakyReLU(),
                               nn.InstanceNorm2d(in_channels*4)])
        self.l3 = nn.Sequential(*[HexConv(in_channels*4, out_channels, 5), nn.LeakyReLU()])

    def forward(self, x):
        B, T, P, H, W = x.shape
        x = x.view(-1, P, H, W)

        x = self.l1(x)
        x = self.l2(x)
        return self.l3(x).view(B, T, -1)

class DeeperFollowerStateCNN(nn.Module):
    '''
    A 4 layer CNN using HexaConv as its convolution layer. It expects
    an input of shape BxTxPx15x15 and processes it with 3 layers with kernel
    size 5 and one layer with kernel size 3.
    '''

    def __init__(self, in_channels=CNN_EMB_DIM, out_channels=GPT_EMB_DIM):
        super().__init__()
        self.l1 = nn.Sequential(*[HexConv(in_channels, in_channels*2, 5), nn.LeakyReLU(),
                               nn.InstanceNorm2d(in_channels*2)])
        self.l2 = nn.Sequential(*[HexConv(in_channels*2, in_channels*4, 5), nn.LeakyReLU(),
                               nn.InstanceNorm2d(in_channels*4)])
        self.l3 = nn.Sequential(*[HexConv(in_channels*4, out_channels, 5), nn.LeakyReLU(),
                               nn.InstanceNorm2d(out_channels)])
        self.l4 = nn.Sequential(*[HexConv(out_channels, out_channels, 3), nn.LeakyReLU()])

    def forward(self, x, B=None, T=None):
        if B is None:
            B, T, P, H, W = x.shape
        else:
            P, H, W = x.shape[-3:]
        x = x.view(-1, P, H, W)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return self.l4(x).view(B, T, -1)

class ResNetFollowerStateCNN(nn.Module):
    '''
    A 4 layer ResNet with HexaConv convolutions followed by a DeeperFollowerStateCNN.
    '''
    
    def __init__(self, in_channels=CNN_EMB_DIM, out_channels=GPT_EMB_DIM):
        super().__init__()
        self.l1 = nn.Sequential(*[HexConv(in_channels, in_channels, 3, padding=1), nn.LeakyReLU()])

        self.res_layers = nn.ModuleList([])
        for layer in range(3):
            curr_layer = [nn.InstanceNorm2d(in_channels),
                          HexConv(in_channels, in_channels, 3, padding=1),
                          nn.LeakyReLU()]
            if layer == 2:
                curr_layer.append(nn.InstanceNorm2d(in_channels))

            self.res_layers.append(nn.Sequential(*curr_layer))

        self.out_cnn = DeeperFollowerStateCNN(in_channels=in_channels, out_channels=out_channels)
        
    def forward(self, x):
        B, T, P, H, W = x.shape        
        x = x.view(-1, P, H, W)

        x = self.l1(x)
        for layer in self.res_layers:
            x = x + layer(x)
        return self.out_cnn(x, B=B, T=T)

class FollowerEnsemble(object):
    def __init__(self, followers: List[DecisionTransformer], ensembling_strategy: EnsemblingStrategy):
        self.followers = followers
        self.ensembling_strategy = ensembling_strategy

    # Forward pass with ensembled models
    def compute_probabilities(self, states, actions, timesteps, text_conditioning,
                              pos_idx, attention_mask, text_attention_mask, action_mask):
        # Have all models perform a forward pass
        all_action_logits = []
        with torch.no_grad():
            for follower in self.followers:
                # BxTx6 each
                all_action_logits.append(follower.forward(
                    states,
                    actions,
                    timesteps,
                    text_conditioning,
                    pos_idx,
                    attention_mask,
                    text_attention_mask,
                    action_mask))
        return self.ensembled_probabilities(all_action_logits, action_mask)

    def ensembled_probabilities(self, all_action_logits, action_mask):
        if self.ensembling_strategy == EnsemblingStrategy.MAJORITY_VOTING_RAW:
            return self.majority_voting_raw(all_action_logits)
        elif self.ensembling_strategy == EnsemblingStrategy.MAJORITY_VOTING_SOFTMAX:
            return self.majority_voting_softmax(all_action_logits, action_mask)
        elif self.ensembling_strategy == EnsemblingStrategy.BOLTZMANN_MULTIPLICATION:
            return self.boltzmann_multiplication(all_action_logits)
        else:
            assert(False, "Input an invalid ensembling strategy")

    def majority_voting_raw(self, all_action_logits):
        '''
        Input: A list of BxTxA tensors of action logits
        Output: Probability distribution obtained through majority voting without a softmax.
        '''
        votes = self.collect_votes(all_action_logits)
        totals = torch.sum(votes, dim=2).unsqueeze(2)
        return votes / totals

    def majority_voting_softmax(self, all_action_logits, action_mask):
        votes = self.collect_votes(all_action_logits)
        votes.masked_fill_(action_mask.unsqueeze(1), -float('inf'))
        return F.softmax(votes, dim=2)

    def boltzmann_multiplication(self, all_action_logits):
        # Compute probabilities for each model in the ensemble
        all_probs = [F.softmax(logits, dim=2) for logits in all_action_logits]

        # Get the product of each of these
        B, T, A = all_action_logits[-1].shape
        return_probs = torch.ones(B, T, A)
        for probs in all_probs:
            return_probs *= probs

        # Normalize the result
        totals = torch.sum(return_probs, dim=2).unsqueeze(2)
        return return_probs / totals

    def collect_votes(self, all_action_logits):
        # Initialize the return tensor
        B, T, A = all_action_logits[-1].shape
        return_probs = torch.zeros(B, T, A).to(all_action_logits[-1].device)

        # Get the argmax for each output and collect votes
        all_argmax = [torch.argmax(logits, dim=2) for logits in all_action_logits]
        for i in range(A):
            for argmax in all_argmax:
                return_probs[:, :, i] += (argmax == i).type(torch.float)
        return return_probs

    def sample_action(self, states, actions, timesteps, proc_instruction,
                      pos_idx, attention_mask, text_mask, action_mask):
        # Have all models perform a forward pass (with memory)
        all_action_logits = []
        for follower in self.followers:
            logits = follower.compute_logits_with_past(
                states,
                actions,
                timesteps,
                proc_instruction,
                pos_idx,
                attention_mask,
                text_mask,
                action_mask)
            logits = logits.unsqueeze(1)
            all_action_logits.append(logits)

        # Get the probabilities
        probs = self.ensembled_probabilities(all_action_logits, action_mask).squeeze(1)
        m = Categorical(probs)
        action = m.sample()
        return action

    def eval(self):
        # Set the models to eval
        ray.get([follower.eval.remote() for follower in self.followers])

    def train(self):
        # Set the models to train
        ray.get([follower.train.remote() for follower in self.followers])        

    def reset_past_output(self):
        ray.get([follower.reset_past_output.remote() for follower in self.followers])

    def has_past_output(self):
        return ray.get([self.followers[-1].has_past_output.remote()])[-1]

def DecisionTransformerFromPath(model_path: pathlib.Path) -> DecisionTransformer:
    '''Loads a DecisionTransformer from a path.'''
    model_args_path = model_path / "logging" / "args.json"

    # Load the arguments from legacy JSON and turn them into a DecisionTransformerConfig.
    legacy_args = LegacyDecisionTransformerConfig.from_json(model_args_path)
    config = DecisionTransformerConfig(
        number_of_actions=legacy_args.act_dim,
        state_embedding_dimension=legacy_args.state_embed_dim,
        convolution_type=ConvolutionType(legacy_args.cnn_option),
        use_timesteps=legacy_args.use_timesteps,
        sampling_strategy=SamplingStrategy[legacy_args.sampling_strat.upper()],
        max_episode_length=legacy_args.max_ep_len,
        action_tanh=False,
        layer_normalization=True,
        freeze_embeddings=legacy_args.freeze_embeddings,
        embedding_size=GPT_EMB_DIM,
        inference_temperature=legacy_args.inference_temperature,
        gpt2_layer_prefix=legacy_args.num_layers)

    # Load the model from the path.
    model = DecisionTransformer(config, TORCH_DEVICE)

    # Load the weights from the path.
    weights_path = model_path / "checkpoints" / "best_follower.pth"
    state_dict, _, _ = torch.load(weights_path)
    model.load_state_dict(state_dict)
    return model