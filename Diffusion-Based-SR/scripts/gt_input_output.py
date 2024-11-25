"""make variations of input image"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import math
import copy
from scripts.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization

def space_timesteps(num_timesteps, section_counts):
	"""
	Create a list of timesteps to use from an original diffusion process,
	given the number of timesteps we want to take from equally-sized portions
	of the original process.
	For example, if there's 300 timesteps and the section counts are [10,15,20]
	then the first 100 timesteps are strided to be 10 timesteps, the second 100
	are strided to be 15 timesteps, and the final 100 are strided to be 20.
	If the stride is a string starting with "ddim", then the fixed striding
	from the DDIM paper is used, and only one section is allowed.
	:param num_timesteps: the number of diffusion steps in the original
						  process to divide up.
	:param section_counts: either a list of numbers, or a string containing
						   comma-separated numbers, indicating the step count
						   per section. As a special case, use "ddimN" where N
						   is a number of steps to use the striding from the
						   DDIM paper.
	:return: a set of diffusion steps from the original process to use.
	"""
	if isinstance(section_counts, str):
		if section_counts.startswith("ddim"):
			desired_count = int(section_counts[len("ddim"):])
			for i in range(1, num_timesteps):
				if len(range(0, num_timesteps, i)) == desired_count:
					return set(range(0, num_timesteps, i))
			raise ValueError(
				f"cannot create exactly {num_timesteps} steps with an integer stride"
			)
		section_counts = [int(x) for x in section_counts.split(",")]   #[250,]
	size_per = num_timesteps // len(section_counts)
	extra = num_timesteps % len(section_counts)
	start_idx = 0
	all_steps = []
	for i, section_count in enumerate(section_counts):
		size = size_per + (1 if i < extra else 0)
		if size < section_count:
			raise ValueError(
				f"cannot divide section of {size} steps into {section_count}"
			)
		if section_count <= 1:
			frac_stride = 1
		else:
			frac_stride = (size - 1) / (section_count - 1)
		cur_idx = 0.0
		taken_steps = []
		for _ in range(section_count):
			taken_steps.append(start_idx + round(cur_idx))
			cur_idx += frac_stride
		all_steps += taken_steps
		start_idx += size
	return set(all_steps)

def chunk(it, size):
	it = iter(it)
	return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
	print(f"Loading model from {ckpt}")
	pl_sd = torch.load(ckpt, map_location="cpu")
	if "global_step" in pl_sd:
		print(f"Global Step: {pl_sd['global_step']}")
	sd = pl_sd["state_dict"]
	model = instantiate_from_config(config.model)
	m, u = model.load_state_dict(sd, strict=False)
	if len(m) > 0 and verbose:
		print("missing keys:")
		print(m)
	if len(u) > 0 and verbose:
		print("unexpected keys:")
		print(u)

	# model.cuda()
	model.eval()
	return model

def load_img(path):
	image = Image.open(path).convert("RGB")
	w, h = image.size
	print(f"loaded input image of size ({w}, {h}) from {path}")
	w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
	image = image.resize((w, h), resample=PIL.Image.LANCZOS)
	image = np.array(image).astype(np.float32) / 255.0
	image = image[None].transpose(0, 3, 1, 2)
	image = torch.from_numpy(image)
	return 2.*image - 1.


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--img_path",
		type=str,
		nargs="?",
		help="path to the input image",
		default="/data1_ssd4t/chendu/datasets/DF2K/subimages_512",
	)
	parser.add_argument(
		"--save_path",
		type=str,
		nargs="?",
		help="dir to write results to",
		default="./GT_intput_output",
	)

	parser.add_argument('--img_numbers', type=int, default=10)

	parser.add_argument(
		"--save_samples",
		type=bool,
		default=True,
	)

	parser.add_argument(
		"--ddpm_steps",
		type=int,
		default=1000,
		help="number of ddpm sampling steps",
	)
	parser.add_argument(
		"--C",
		type=int,
		default=4,
		help="latent channels",
	)
	parser.add_argument(
		"--f",
		type=int,
		default=8,
		help="downsampling factor, most often 8 or 16",
	)
	parser.add_argument(
		"--n_samples",
		type=int,
		default=1,
		help="how many samples to produce for each given prompt. A.k.a batch size",
	)
	parser.add_argument(
		"--config",
		type=str,
		default="/data1_ssd4t/chendu/myprojects/MyStableSR/configs/GT_input_output/v2-finetune_text_T_512.yaml",
		help="path to config which constructs model",
	)
	parser.add_argument(
		"--ckpt",
		type=str,
		default="/data1_ssd4t/chendu/myprojects/MyStableSR/pretrained_models/StableSR/stablesr_000117.ckpt",
		help="path to checkpoint of model",
	)
	parser.add_argument(
		"--vqgan_ckpt",
		type=str,
		default="/data1_ssd4t/chendu/myprojects/MyStableSR/pretrained_models/StableSR/vqgan_cfw_00011_vae_only.ckpt",
		help="path to checkpoint of VQGAN model",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="the seed (for reproducible sampling)",
	)
	parser.add_argument(
		"--precision",
		type=str,
		help="evaluate at this precision",
		choices=["full", "autocast"],
		default="autocast"
	)
	parser.add_argument(
		"--input_size",
		type=int,
		default=512,
		help="input size",
	)
	parser.add_argument(
		"--dec_w",
		type=float,
		default=0,
		help="weight for combining VQGAN and Diffusion",
	)
	parser.add_argument(
		"--q_sample_step",
		type=int,
		default=500,
		help="q_sample_step",
	)

	parser.add_argument(
		"--p_sample_step",
		type=int,
		default=500,
		help="p_sample_step",
	)

	opt = parser.parse_args()

	vqgan_config = OmegaConf.load("/data1_ssd4t/chendu/myprojects/MyStableSR/configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml")
	vq_model = load_model_from_config(vqgan_config, opt.vqgan_ckpt)
	vq_model = vq_model.cuda()
	vq_model.decoder.fusion_w = opt.dec_w

	seed_everything(opt.seed)

	transform = torchvision.transforms.Compose([
		torchvision.transforms.Resize(opt.input_size),
		torchvision.transforms.CenterCrop(opt.input_size),
	])

	config = OmegaConf.load(f"{opt.config}")
	model = load_model_from_config(config, f"{opt.ckpt}")
	model = model.cuda()

	print(f'save path is {os.path.abspath(opt.save_path)}')

	save_original_gt_path = os.path.join(opt.save_path, 'originalGT')
	save_change_gt_path = os.path.join(opt.save_path, 'changeGT',f'p_sample_step{opt.p_sample_step}-q_sample_step{opt.q_sample_step}')

	os.makedirs(save_original_gt_path, exist_ok=True)
	os.makedirs(save_change_gt_path, exist_ok=True)


	model.register_schedule(given_betas=None, beta_schedule="linear", timesteps=1000,
						  linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3)
	model.num_timesteps = 1000

	sqrt_alphas_cumprod = copy.deepcopy(model.sqrt_alphas_cumprod)
	sqrt_one_minus_alphas_cumprod = copy.deepcopy(model.sqrt_one_minus_alphas_cumprod)

	use_timesteps = set(space_timesteps(1000, [opt.ddpm_steps]))
	last_alpha_cumprod = 1.0
	new_betas = []
	timestep_map = []
	for i, alpha_cumprod in enumerate(model.alphas_cumprod):
		if i in use_timesteps:
			new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
			last_alpha_cumprod = alpha_cumprod
			timestep_map.append(i)
	new_betas = [beta.data.cpu().numpy() for beta in new_betas]
	model.register_schedule(given_betas=np.array(new_betas), timesteps=len(new_betas))
	model.num_timesteps = 1000
	model.ori_timesteps = list(use_timesteps)
	model.ori_timesteps.sort()
	model = model.cuda()

	precision_scope = autocast if opt.precision == "autocast" else nullcontext

	img_list_ori = os.listdir(opt.img_path)
	img_list = copy.deepcopy(img_list_ori)

	if opt.img_numbers > 0:
		for idx, item in enumerate(img_list_ori):
			if idx < opt.img_numbers:

				image_path = os.path.join(opt.img_path, item)
				if os.path.exists(os.path.join(save_original_gt_path, item)):
					pass
				else:
					os.system(f'cp -r {image_path} {save_original_gt_path}')
				cur_image = load_img(image_path).cuda()
				cur_image = transform(cur_image)
				cur_image = cur_image.clamp(-1, 1)
				init_image_list = cur_image

				basename = os.path.splitext(item)[0]

				with torch.no_grad():
					with precision_scope("cuda"):
						with model.ema_scope():
							init_image = init_image_list
							init_latent_generator, enc_fea_lq = vq_model.encode(init_image)
							init_latent = model.get_first_stage_encoding(init_latent_generator)
							text_init = ['']*init_image.size(0)
							semantic_c = model.cond_stage_model(text_init)

							noise = torch.randn_like(init_latent)
							# If you would like to start from the intermediate steps, you can add noise to LR to the specific steps.
							t = repeat(torch.tensor([opt.q_sample_step]), '1 -> b', b=init_image.size(0))
							t = t.cuda().long()
							x_T = model.q_sample_respace(x_start=init_latent, t=t, sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, noise=noise)

							samples, _ = model.sample(cond=semantic_c, struct_cond=init_latent, p_sample_step = opt.p_sample_step, batch_size=init_image.size(0), timesteps=opt.ddpm_steps, time_replace=opt.ddpm_steps, x_T=x_T, return_intermediates=True)
							print(f"samples shape is {samples.shape}")


							if opt.save_samples:
								x_samples = vq_model.decode(samples * 1. / model.scale_factor, enc_fea_lq)
								x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
								x_sample = 255. * rearrange(x_samples[0].cpu().numpy(), 'c h w -> h w c')
								Image.fromarray(x_sample.astype(np.uint8)).save(
									os.path.join(save_change_gt_path, basename+f'p{opt.p_sample_step}q{opt.q_sample_step}.png'))




if __name__ == "__main__":
	# ===== change the spade of StableSR =====
	from ldm.modules.spade import SPADE
	def spade_wrap():
		if not hasattr(SPADE, 'spade_flag') or SPADE.spade_flag is False:
			def new_forward(self, x_dic, segmap_dic, size=None):
				return x_dic

			SPADE.forward = new_forward
		SPADE.spade_flag = True


	spade_wrap()
	# ===== ends =====
	main()
