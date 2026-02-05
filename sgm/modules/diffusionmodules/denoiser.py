from typing import Dict, Union

import torch
import torch.nn as nn

from ...util import append_dims, instantiate_from_config


class Denoiser(nn.Module):
    def __init__(self, weighting_config, scaling_config):
        super().__init__()

        self.weighting = instantiate_from_config(weighting_config)
        self.scaling = instantiate_from_config(scaling_config)

    def possibly_quantize_sigma(self, sigma):
        return sigma

    def possibly_quantize_c_noise(self, c_noise):
        return c_noise

    def w(self, sigma):
        return self.weighting(sigma)

    def forward(
        self,
        network: nn.Module,
        input: torch.Tensor,
        sigma: torch.Tensor,
        cond: Dict,
        **additional_model_inputs,
    ) -> torch.Tensor:
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma, **additional_model_inputs)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        return network(input * c_in, c_noise, cond, **additional_model_inputs) * c_out + input * c_skip


class DiscreteDenoiser(Denoiser):
    def __init__(
        self,
        weighting_config,
        scaling_config,
        num_idx,
        discretization_config,
        do_append_zero=False,
        quantize_c_noise=True,
        flip=True,
    ):
        super().__init__(weighting_config, scaling_config)
        sigmas = instantiate_from_config(discretization_config)(num_idx, do_append_zero=do_append_zero, flip=flip)
        self.sigmas = sigmas
        # self.register_buffer("sigmas", sigmas)
        self.quantize_c_noise = quantize_c_noise

    def sigma_to_idx(self, sigma):
        dists = sigma - self.sigmas.to(sigma.device)[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def idx_to_sigma(self, idx):
        return self.sigmas.to(idx.device)[idx]

    def possibly_quantize_sigma(self, sigma):
        return self.idx_to_sigma(self.sigma_to_idx(sigma))

    def possibly_quantize_c_noise(self, c_noise):
        if self.quantize_c_noise:
            return self.sigma_to_idx(c_noise)
        else:
            return c_noise



class DiscreteDenoiser_withact(Denoiser):
    def __init__(
        self,
        weighting_config,
        scaling_config,
        num_idx,
        discretization_config,
        do_append_zero=False,
        quantize_c_noise=True,
        flip=True,
        use_latent_action=False,
    ):
        super().__init__(weighting_config, scaling_config)
        sigmas = instantiate_from_config(discretization_config)(num_idx, do_append_zero=do_append_zero, flip=flip)
        self.sigmas = sigmas
        # self.register_buffer("sigmas", sigmas)
        self.quantize_c_noise = quantize_c_noise
        self.use_latent_action = use_latent_action

    def sigma_to_idx(self, sigma):
        dists = sigma - self.sigmas.to(sigma.device)[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def idx_to_sigma(self, idx):
        return self.sigmas.to(idx.device)[idx]

    def possibly_quantize_sigma(self, sigma):
        return self.idx_to_sigma(self.sigma_to_idx(sigma))

    def possibly_quantize_c_noise(self, c_noise):
        if self.quantize_c_noise:
            return self.sigma_to_idx(c_noise)
        else:
            return c_noise
    def forward(
        self,
        network: nn.Module,
        input: torch.Tensor,
        sigma: torch.Tensor,
        cond: Dict,
        **additional_model_inputs,
    ) -> torch.Tensor:
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape

        # video_sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma, **additional_model_inputs)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape)) #hacking 


        video_c_skip = append_dims(c_skip, input.ndim)
        video_c_out = append_dims(c_out, input.ndim)
        video_c_in = append_dims(c_in, input.ndim)

        additional_output_dict = additional_model_inputs["additional_output_dict"] 
        action_noise_input = additional_model_inputs["action_noise_input"] #train: gt_with_noise; test: last step noise
        action_c_skip = append_dims(c_skip, action_noise_input.ndim)
        action_c_out = append_dims(c_out, action_noise_input.ndim)
        action_c_in = append_dims(c_in, action_noise_input.ndim)
        additional_model_inputs["action_input_withdenoiser"] = action_noise_input * action_c_in
        additional_model_inputs["action_c_skip"] = action_c_skip
        additional_model_inputs["action_c_out"] = action_c_out
        additional_model_inputs["action_c_in"] = action_c_in
        
        # network(input * c_in, c_noise, cond, **additional_model_inputs) * c_out + input * c_skip
        video_pred_noise, action_pred_noise = network(input * video_c_in, c_noise, cond, **additional_model_inputs)
        video_pred_noise = video_pred_noise * video_c_out + input * video_c_skip
        action_pred_noise = action_pred_noise * action_c_out + action_noise_input * action_c_skip

        additional_output_dict["action_pred_noise"] = action_pred_noise

        return video_pred_noise
