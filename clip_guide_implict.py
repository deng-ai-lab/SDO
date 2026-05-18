import argparse
import os
import shutil

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
import clip
import yaml
from diffusers import DDIMScheduler

from guided_diffusion.unet import create_model
from util.DiffAugment_pytorch import DiffAugment


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def torch_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_ref_image(path: str, device, dtype):
    img = Image.open(path).convert("RGB")
    ref_numpy = np.array(img) / 255.0
    x = ref_numpy * 2 - 1
    x = x.transpose(2, 0, 1)
    return torch.tensor(x, dtype=dtype, device=device).unsqueeze(0)


def make_clip_loss(clip_model, device, ref_img):
    preprocess = torchvision.transforms.Normalize(
        (0.48145466 * 2 - 1, 0.4578275 * 2 - 1, 0.40821073 * 2 - 1),
        (0.26862954 * 2, 0.26130258 * 2, 0.27577711 * 2),
    )

    def loss_fn(x, text, alpha=0.5):
        sim = (x - ref_img).abs().mean()
        img_aug = DiffAugment(x.repeat(20, 1, 1, 1), policy="color,translation,resize,cutout")
        img_aug = torch.nn.functional.interpolate(img_aug, size=224, mode="bilinear", align_corners=False)
        img_aug = preprocess(img_aug)
        text_tok = clip.tokenize([text]).to(device)
        logits_per_image, _ = clip_model(img_aug, text_tok)
        logits_per_image = logits_per_image / clip_model.logit_scale.exp()
        concept_loss = (-1.0) * logits_per_image
        return alpha * concept_loss.mean() + (1.0 - alpha) * sim.sum()

    return loss_fn


def dmplug_clip_optimize(
    model,
    scheduler,
    logdir,
    ref_img,
    prompt,
    lr,
    guidance_steps,
    alpha_warmup_steps,
    alpha_after,
    device,
):
    os.makedirs(logdir, exist_ok=True)

    image_size = ref_img.shape[-1]
    z = torch.randn((1, 3, image_size, image_size), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([{"params": z, "lr": lr}], betas=(0.9, 0.9))

    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    loss_fn = make_clip_loss(clip_model, device, ref_img)

    losses = []
    for step in range(guidance_steps):
        model.eval()
        optimizer.zero_grad(set_to_none=True)
        for i, tt in enumerate(scheduler.timesteps):
            t = (torch.ones(1) * tt).to(device)
            if i == 0:
                noise_pred = model(z, t)
            else:
                with torch.no_grad():
                    noise_pred = model(x_t, t).detach()
            noise_pred = noise_pred[:, :3]
            if i == 0:
                x_t = scheduler.step(noise_pred, tt, z, return_dict=True, use_clipped_model_output=True).prev_sample
            else:
                x_t = scheduler.step(noise_pred, tt, x_t, return_dict=True, use_clipped_model_output=True).prev_sample

        output = torch.clamp(x_t, -1, 1)
        alpha = 0.0 if step < alpha_warmup_steps else alpha_after
        loss = loss_fn(output, prompt, alpha=alpha)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        output_numpy = output.detach().cpu().squeeze().numpy()
        output_numpy = (output_numpy + 1) / 2
        output_numpy = np.transpose(output_numpy, (1, 2, 0))
        plt.imsave(os.path.join(logdir, f"rec_img_{step}.png"), output_numpy)

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
    parser.add_argument("--dataset", type=str, default="ffhq_eval")
    parser.add_argument("--img", type=int, default=98)
    parser.add_argument("--image_ext", type=str, default=".png")
    parser.add_argument("--logdir", type=str, default="./results_sdo")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="A photo of a smile face.")
    parser.add_argument("--custom_steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--guidance_steps", type=int, default=200)
    parser.add_argument("--alpha_warmup_steps", type=int, default=150)
    parser.add_argument("--alpha_after", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_config", type=str, default="configs/model_config_ffhq.yaml")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main():
    args = get_parser().parse_args()
    torch_seed(args.seed)

    device = torch.device(args.device)
    model_config = load_yaml(args.model_config)
    model = create_model(**model_config).to(device).eval()

    scheduler = DDIMScheduler()
    scheduler.set_timesteps(args.custom_steps)

    ref_path = resolve_input_path(args)
    ref_img = load_ref_image(ref_path, device, torch.float32)

    run_name = resolve_run_name(args, ref_path)
    logdir = os.path.join(args.logdir, run_name)
    os.makedirs(logdir, exist_ok=True)
    shutil.copy(ref_path, os.path.join(logdir, "gt.png"))

    dmplug_clip_optimize(
        model=model,
        scheduler=scheduler,
        logdir=logdir,
        ref_img=ref_img,
        prompt=args.prompt,
        lr=args.lr,
        guidance_steps=args.guidance_steps,
        alpha_warmup_steps=args.alpha_warmup_steps,
        alpha_after=args.alpha_after,
        device=device,
    )


if __name__ == "__main__":
    main()
