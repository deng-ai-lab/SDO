import argparse
import os
import shutil

import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import torchvision
import open_clip
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DDIMInverseScheduler
from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer


def torch_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def center_crop(im: Image.Image) -> Image.Image:
    width, height = im.size
    min_dim = min(width, height)
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2
    return im.crop((left, top, right, bottom))


def load_image_512(path: str) -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img = center_crop(img).resize((512, 512), resample=Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS)
    return img


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cut_power=1.0):
        super().__init__()
        self.cut_size = cut_size
        self.cut_power = cut_power

    def forward(self, pixel_values, num_cutouts):
        side_y, side_x = pixel_values.shape[2:4]
        max_size = min(side_x, side_y)
        min_size = min(side_x, side_y, self.cut_size)
        cutouts = []
        for _ in range(num_cutouts):
            size = int(torch.rand([]) ** self.cut_power * (max_size - min_size) + min_size)
            offset_x = torch.randint(0, side_x - size + 1, ())
            offset_y = torch.randint(0, side_y - size + 1, ())
            cutout = pixel_values[:, :, offset_y : offset_y + size, offset_x : offset_x + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


def get_aesthetic_model(clip_model: str = "vit_l_14") -> nn.Module:
    cache_folder = ".cache/emb_reader"
    path_to_model = os.path.join(cache_folder, f"sa_0_4_{clip_model}_linear.pth")
    if clip_model == "vit_l_14":
        model = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        model = nn.Linear(512, 1)
    else:
        raise ValueError("clip_model must be vit_l_14 or vit_b_32")
    state = torch.load(path_to_model, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def aesthetic_score(image, clip_model, aesthetic_head, use_amp):
    with autocast(enabled=use_amp):
        image_features = clip_model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        prediction = aesthetic_head(image_features).clip(1, 10).mean()
        return prediction


def aesthetic_loss_fn(
    dtype,
    device,
    aesthetic_target=None,
    use_cutouts=False,
    num_cuts=64,
    cut_power=0.6,
    grad_scale=1.0,
    clip_model_str="vit_l_14",
    weights=None,
):
    target_size = 224
    normalize = torchvision.transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    cut = MakeCutouts(cut_size=target_size, cut_power=cut_power) if use_cutouts else None
    weights = weights or [1, 1]

    if clip_model_str == "both":
        model_l, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        model_b, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        model_l = model_l.to(device)
        model_b = model_b.to(device)
        model_l.eval()
        model_b.eval()
        for p in model_l.parameters():
            p.requires_grad_(False)
        for p in model_b.parameters():
            p.requires_grad_(False)
        head_l = get_aesthetic_model("vit_l_14").to(device).eval()
        head_b = get_aesthetic_model("vit_b_32").to(device).eval()
        for p in head_l.parameters():
            p.requires_grad_(False)
        for p in head_b.parameters():
            p.requires_grad_(False)
        models = [model_l, model_b]
        heads = [head_l, head_b]
    else:
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14" if clip_model_str == "vit_l_14" else "ViT-B-32",
            pretrained="openai",
        )
        model = model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        head = get_aesthetic_model(clip_model_str).to(device).eval()
        for p in head.parameters():
            p.requires_grad_(False)
        models = [model]
        heads = [head]

    use_amp = device.type == "cuda"

    def loss_fn(im_pix):
        im_pix = ((im_pix / 2) + 0.5).clamp(0, 1)
        if cut is not None:
            x_var = cut(im_pix, num_cuts)
        else:
            x_var = torchvision.transforms.Resize(target_size)(im_pix)
        x_var = normalize(x_var).to(dtype)
        predictions = [aesthetic_score(x_var, model, head, use_amp) for model, head in zip(models, heads)]
        prediction = sum([w * p for w, p in zip(weights, predictions)]) / len(predictions)
        if aesthetic_target is None:
            loss = -1 * prediction
        else:
            loss = abs(prediction - aesthetic_target)
        return loss * grad_scale

    return loss_fn


def build_unconditional_text_embedding(clip_tokenizer, clip_text: CLIPTextModel, device, dtype):
    with torch.no_grad():
        tokens = clip_tokenizer(
            "",
            padding="max_length",
            max_length=clip_tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
            return_overflowing_tokens=True,
        )
        embedding = clip_text(tokens.input_ids.to(device)).last_hidden_state
        return embedding.to(dtype)


def dmplug_aesthetic_optimize(
    unet,
    vae,
    scheduler,
    logdir,
    ref_img,
    eta,
    lr,
    pre_steps,
    optim_steps,
    aesthetic_target,
    device,
):
    use_amp = device.type == "cuda"
    os.makedirs(logdir, exist_ok=True)

    model_path_clip = "openai/clip-vit-large-patch14"
    clip_tokenizer = CLIPTokenizer.from_pretrained(model_path_clip)
    clip_model = CLIPModel.from_pretrained(model_path_clip, torch_dtype=unet.dtype)
    clip_text = clip_model.text_model.to(device)

    unet.to(device).eval()
    vae.to(device).eval()

    embedding_unconditional = build_unconditional_text_embedding(clip_tokenizer, clip_text, device, unet.dtype)

    scheduler_inversion = DDIMInverseScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        clip_sample=False,
        set_alpha_to_one=False,
    )
    scheduler_inversion.set_timesteps(50)

    with torch.no_grad():
        with autocast(enabled=use_amp):
            init_latent = vae.encode(ref_img).latent_dist.sample() * 0.18215
            x_t = None
            for i, tt in enumerate(scheduler_inversion.timesteps[:pre_steps]):
                if i == 0:
                    noise_pred = unet(init_latent, tt, encoder_hidden_states=embedding_unconditional).sample
                else:
                    noise_pred = unet(x_t, tt, encoder_hidden_states=embedding_unconditional).sample
                if i == 0:
                    x_t = scheduler_inversion.step(noise_pred, tt, init_latent, return_dict=True).prev_sample
                else:
                    x_t = scheduler_inversion.step(noise_pred, tt, x_t, return_dict=True).prev_sample
            z = x_t.detach()
    z.requires_grad = True

    loss_aes = aesthetic_loss_fn(
        dtype=vae.dtype,
        device=device,
        aesthetic_target=aesthetic_target,
        use_cutouts=True,
        num_cuts=16,
        cut_power=0.3,
        grad_scale=1.0,
        clip_model_str="both",
        weights=[0.5, 1.5],
    )

    optimizer = torch.optim.Adam([{"params": z, "lr": lr}], betas=(0.9, 0.9))
    scaler = GradScaler(enabled=use_amp)

    losses = []
    for step in range(optim_steps):
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            for i, tt in enumerate(scheduler.timesteps[pre_steps:]):
                if i == 0:
                    noise_pred = unet(z, tt, encoder_hidden_states=embedding_unconditional).sample
                else:
                    with torch.no_grad():
                        noise_pred = unet(x_t, tt, encoder_hidden_states=embedding_unconditional).sample.detach()
                if i == 0:
                    x_t = scheduler.step(noise_pred, tt, z, return_dict=True, use_clipped_model_output=True, eta=eta).prev_sample
                else:
                    x_t = scheduler.step(noise_pred, tt, x_t, return_dict=True, use_clipped_model_output=True, eta=eta).prev_sample

            x_t_scaled = x_t / 0.18215
            out = vae.decode(x_t_scaled).sample
            out = torch.clamp(out, -1, 1)
            loss = loss_aes(out)

        losses.append(loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        out_np = out.detach().cpu().squeeze().numpy()
        out_np = (out_np + 1) / 2
        out_np = np.transpose(out_np, (1, 2, 0))
        plt.imsave(os.path.join(logdir, f"rec_img_{step}.png"), out_np)

    plt.plot(np.array(losses), label="loss")
    plt.legend()
    plt.savefig(os.path.join(logdir, "loss.png"))
    plt.close()


def resolve_input_path(args) -> str:
    if args.input_image:
        return args.input_image
    if args.dataset is None or args.img is None:
        raise ValueError("Provide --input_image or --dataset/--img")
    img = str(args.img).zfill(5)
    return os.path.join(args.data_root, args.dataset, f"{img}{args.image_ext}")


def resolve_run_name(args, input_path: str) -> str:
    if args.run_name:
        return args.run_name
    if args.input_image:
        return os.path.splitext(os.path.basename(input_path))[0]
    if args.dataset is not None and args.img is not None:
        img = str(args.img).zfill(5)
        return os.path.join(args.dataset, img)
    return "run"


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="bedroom")
    parser.add_argument("--img", type=int, default=80)
    parser.add_argument("--image_ext", type=str, default=".jpg")
    parser.add_argument("--logdir", type=str, default="./results_sdo")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--custom_steps", type=int, default=50)
    parser.add_argument("--pre_steps", type=int, default=10)
    parser.add_argument("--optim_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--aesthetic_target", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--revision", type=str, default="fp16")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main():
    args = get_parser().parse_args()
    torch_seed(args.seed)

    device = torch.device(args.device)
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    unet = UNet2DConditionModel.from_pretrained(
        args.model_path,
        subfolder="unet",
        revision=args.revision,
        torch_dtype=torch_dtype,
    )
    vae = AutoencoderKL.from_pretrained(
        args.model_path,
        subfolder="vae",
        revision=args.revision,
        torch_dtype=torch_dtype,
    )

    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        clip_sample=False,
        set_alpha_to_one=False,
    )
    scheduler.set_timesteps(args.custom_steps)

    ref_path = resolve_input_path(args)
    ref_img = load_image_512(ref_path)
    ref_img = np.array(ref_img) / 255.0 * 2.0 - 1.0
    ref_img = torch.from_numpy(ref_img[np.newaxis, ...].transpose(0, 3, 1, 2)).to(device).to(unet.dtype)
    ref_img.requires_grad = False

    run_name = resolve_run_name(args, ref_path)
    logdir = os.path.join(args.logdir, run_name)
    os.makedirs(logdir, exist_ok=True)
    shutil.copy(ref_path, os.path.join(logdir, "gt.png"))

    dmplug_aesthetic_optimize(
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        logdir=logdir,
        ref_img=ref_img,
        eta=args.eta,
        lr=args.lr,
        pre_steps=args.pre_steps,
        optim_steps=args.optim_steps,
        aesthetic_target=args.aesthetic_target,
        device=device,
    )


if __name__ == "__main__":
    main()
