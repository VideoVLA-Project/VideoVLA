from typing import List, Optional, Union

import torch
import torch.nn as nn
from omegaconf import ListConfig
from ...util import append_dims, instantiate_from_config
from ...modules.autoencoding.lpips.loss.lpips import LPIPS
from sat import mpu


class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config,
        type="l2",
        offset_noise_level=0.0,
        batch2model_keys: Optional[Union[str, List[str], ListConfig]] = None,
    ):
        super().__init__()

        assert type in ["l2", "l1", "lpips"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)

        self.type = type
        self.offset_noise_level = offset_noise_level

        if type == "lpips":
            self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

    def __call__(self, network, denoiser, conditioner, input, batch):
        cond = conditioner(batch)
        additional_model_inputs = {key: batch[key] for key in self.batch2model_keys.intersection(batch)}

        sigmas = self.sigma_sampler(input.shape[0]).to(input.device)
        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            noise = (
                noise + append_dims(torch.randn(input.shape[0]).to(input.device), input.ndim) * self.offset_noise_level
            )
            noise = noise.to(input.dtype)
        noised_input = input.float() + noise * append_dims(sigmas, input.ndim)
        model_output = denoiser(network, noised_input, sigmas, cond, **additional_model_inputs)
        w = append_dims(denoiser.w(sigmas), input.ndim)
        return self.get_loss(model_output, input, w)

    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
        elif self.type == "l1":
            return torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss


class VideoDiffusionLoss(StandardDiffusionLoss):
    def __init__(self, block_scale=None, block_size=None, min_snr_value=None, fixed_frames=0, **kwargs):
        self.fixed_frames = fixed_frames
        self.block_scale = block_scale
        self.block_size = block_size
        self.min_snr_value = min_snr_value
        super().__init__(**kwargs)

    def __call__(self, network, denoiser, conditioner, input, batch):
        cond = conditioner(batch)
        additional_model_inputs = {key: batch[key] for key in self.batch2model_keys.intersection(batch)}

        alphas_cumprod_sqrt, idx = self.sigma_sampler(input.shape[0], return_idx=True)
        alphas_cumprod_sqrt = alphas_cumprod_sqrt.to(input.device)
        idx = idx.to(input.device)

        noise = torch.randn_like(input)

        # broadcast noise
        mp_size = mpu.get_model_parallel_world_size()
        global_rank = torch.distributed.get_rank() // mp_size
        src = global_rank * mp_size
        torch.distributed.broadcast(idx, src=src, group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(noise, src=src, group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(alphas_cumprod_sqrt, src=src, group=mpu.get_model_parallel_group())

        additional_model_inputs["idx"] = idx

        if self.offset_noise_level > 0.0:
            noise = (
                noise + append_dims(torch.randn(input.shape[0]).to(input.device), input.ndim) * self.offset_noise_level
            )

        noised_input = input.float() * append_dims(alphas_cumprod_sqrt, input.ndim) + noise * append_dims(
            (1 - alphas_cumprod_sqrt**2) ** 0.5, input.ndim
        )

        if "concat_images" in batch.keys():
            cond["concat"] = batch["concat_images"]

        # [2, 13, 16, 60, 90],[2] dict_keys(['crossattn', 'concat'])  dict_keys(['idx'])
        model_output = denoiser(network, noised_input, alphas_cumprod_sqrt, cond, **additional_model_inputs)
        w = append_dims(1 / (1 - alphas_cumprod_sqrt**2), input.ndim)  # v-pred

        if self.min_snr_value is not None:
            w = min(w, self.min_snr_value)
        return self.get_loss(model_output, input, w)

    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
        elif self.type == "l1":
            return torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss



class VideoDiffusionLoss_withact(StandardDiffusionLoss):
    def __init__(self, block_scale=None, block_size=None, min_snr_value=None, fixed_frames=0, vd_lw=1, act_lw=1, use_latent_action=False, latent_action_dim=50, **kwargs):
        self.fixed_frames = fixed_frames
        self.block_scale = block_scale
        self.block_size = block_size
        self.min_snr_value = min_snr_value
        self.vd_lw = vd_lw
        self.act_lw = act_lw
        self.use_latent_action = use_latent_action
        self.latent_action_dim = latent_action_dim
        super().__init__(**kwargs)

    def __call__(self, network, denoiser, conditioner, input, batch):
        cond = conditioner(batch)
        additional_model_inputs = {key: batch[key] for key in self.batch2model_keys.intersection(batch)}

        alphas_cumprod_sqrt, idx = self.sigma_sampler(input.shape[0], return_idx=True)
        alphas_cumprod_sqrt = alphas_cumprod_sqrt.to(input.device)
        idx = idx.to(input.device)

        noise = torch.randn_like(input)
        
        # action noise
        if self.use_latent_action:
            action_bs, action_len, _ = batch["actions"].shape # B, action_len, action_dim
            action_input = torch.randn(action_bs, action_len, self.latent_action_dim).to(device=batch["actions"].device,dtype=batch["actions"].dtype) # B, action_len, action_dim
        else:
            action_input = batch["actions"] # B, action_len, action_dim
        
        action_noise = torch.randn_like(action_input) 

        # # broadcast noise # hacking !!!yichao close model_parallel
        # mp_size = mpu.get_model_parallel_world_size()
        # global_rank = torch.distributed.get_rank() // mp_size
        # src = global_rank * mp_size
        # torch.distributed.broadcast(idx, src=src, group=mpu.get_model_parallel_group())
        # torch.distributed.broadcast(noise, src=src, group=mpu.get_model_parallel_group())
        # torch.distributed.broadcast(alphas_cumprod_sqrt, src=src, group=mpu.get_model_parallel_group())

        additional_model_inputs["idx"] = idx

        if self.offset_noise_level > 0.0:
            noise = (
                noise + append_dims(torch.randn(input.shape[0]).to(input.device), input.ndim) * self.offset_noise_level
            )

        noised_input = input.float() * append_dims(alphas_cumprod_sqrt, input.ndim) + noise * append_dims(
            (1 - alphas_cumprod_sqrt**2) ** 0.5, input.ndim
        )

        action_noise_input = action_input.float() * append_dims(alphas_cumprod_sqrt, action_input.ndim) + action_noise * append_dims(
            (1 - alphas_cumprod_sqrt**2) ** 0.5, action_input.ndim
        )
        # additional_model_inputs["action_input"] = action_input
        additional_model_inputs["robot_ids"] = batch["robot_ids"] if "robot_ids" in batch.keys() else None
        # additional_model_inputs["action_noise"] = action_noise
        additional_model_inputs["action_noise_input"] = action_noise_input
        additional_output_dict = {}
        additional_model_inputs["additional_output_dict"] = additional_output_dict


        if "concat_images" in batch.keys():
            cond["concat"] = batch["concat_images"]
        if "action_concat" in batch.keys():
            additional_model_inputs["action_concat"] = batch['action_concat']
        if "vision_feature" in batch.keys():
            additional_model_inputs["vision_feature"] = batch['vision_feature']
            

        # [2, 13, 16, 60, 90],[2] dict_keys(['crossattn', 'concat'])  dict_keys(['idx'])
        model_output = denoiser(network, noised_input, alphas_cumprod_sqrt, cond, **additional_model_inputs)
        # model_output = model_output_dict["video_pred_noise"]
        w = append_dims(1 / (1 - alphas_cumprod_sqrt**2), input.ndim)  # v-pred

        action_model_output = additional_output_dict["action_pred_noise"]
        action_w = append_dims(1 / (1 - alphas_cumprod_sqrt**2), action_model_output.ndim)

        if self.min_snr_value is not None:
            w = min(w, self.min_snr_value)
            action_w = min(action_w, self.min_snr_value)
        
        video_loss = self.get_loss(model_output, input, w)
        action_loss = self.get_loss(action_model_output, action_input, action_w)

        if action_loss.mean().detach().clone().item() > 10:
            print("action_model_output", action_model_output.detach())
            print("action_loss", action_loss.detach())

        loss = video_loss * self.vd_lw + action_loss * self.act_lw 

        loss_dict = {
            "loss":video_loss.mean(),
            "video_loss":video_loss.mean(),
            "action_loss":action_loss.mean(),
        }
        
        return loss, loss_dict

    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
        elif self.type == "l1":
            return torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
