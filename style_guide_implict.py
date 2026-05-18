import argparse
import os
import shutil

import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import torchvision
import clip
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
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


def encode_image_with_features(model, x):
    x = x.type(model.dtype)
    x = model.visual.conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    x = torch.cat(
        [
            model.visual.class_embedding.to(x.dtype)
            + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x,
        ],
        dim=1,
    )
    x = x + model.visual.positional_embedding.to(x.dtype)
    x = model.visual.ln_pre(x)

    x = x.permute(1, 0, 2)
    features = []
    for block in model.visual.transformer.resblocks:
        x = block(x)
        features.append(x)

    x = x.permute(1, 0, 2)
    x = model.visual.ln_post(x[:, 0, :])

    if model.visual.proj is not None:
        x = x @ model.visual.proj

    return x, features


def get_gram_matrix_residual(ref, img, clip_feature):
    preprocess = torchvision.transforms.Normalize(
        (0.48145466 * 2 - 1, 0.4578275 * 2 - 1, 0.40821073 * 2 - 1),
        (0.26862954 * 2, 0.26130258 * 2, 0.27577711 * 2),
    )
    ref = F.interpolate(ref, size=(224, 224), mode="bicubic", align_corners=False)
    ref = preprocess(ref)

    img = F.interpolate(img, size=(224, 224), mode="bicubic", align_corners=False)
    img = preprocess(img)

    _, feats_img = encode_image_with_features(clip_feature, img)
    _, feats_ref = encode_image_with_features(clip_feature, ref)

    feat_img = feats_img[2][1:, 0, :]
    feat_ref = feats_ref[2][1:, 0, :]
    gram_img = torch.mm(feat_img.t(), feat_img)
    gram_ref = torch.mm(feat_ref.t(), feat_ref)
    return gram_img - gram_ref


def build_text_embeddings(clip_tokenizer, clip_text: CLIPTextModel, prompt: str, device, dtype):
    with torch.no_grad():
        null_prompt = ""
        tokens_uncond = clip_tokenizer(
            null_prompt,
            padding="max_length",
            max_length=clip_tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
            return_overflowing_tokens=True,
        )
        tokens_cond = clip_tokenizer(
            prompt,
            padding="max_length",
            max_length=clip_tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
            return_overflowing_tokens=True,
        )
        emb_uncond = clip_text(tokens_uncond.input_ids.to(device)).last_hidden_state
        emb_cond = clip_text(tokens_cond.input_ids.to(device)).last_hidden_state
        text_emb_all = torch.cat([emb_uncond.to(dtype), emb_cond.to(dtype)])
    return text_emb_all


def ddim_guided_sampling(unet, scheduler, z, text_emb_all, guidance_scale, timesteps, eta, device):
    x_t = None
    use_amp = device.type == "cuda"
    with torch.no_grad():
        with autocast(enabled=use_amp):
            for i, tt in enumerate(timesteps):
                t = tt
                if i == 0:
                    z_cat = torch.cat([z] * 2)
                    noise_pred = unet(z_cat, t, encoder_hidden_states=text_emb_all).sample
                    noise_uncond, noise_cond = noise_pred.chunk(2)
                    noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                else:
                    x_t_cat = torch.cat([x_t] * 2)
                    noise_pred = unet(x_t_cat, t, encoder_hidden_states=text_emb_all).sample
                    noise_uncond, noise_cond = noise_pred.chunk(2)
                    noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                if i == 0:
                    x_t = scheduler.step(noise_pred, tt, z, return_dict=True, use_clipped_model_output=True, eta=eta).prev_sample
                else:
                    x_t = scheduler.step(noise_pred, tt, x_t, return_dict=True, use_clipped_model_output=True, eta=eta).prev_sample
    return x_t


def dmplug_style_optimize(
    unet,
    vae,
    scheduler,
    logdir,
    style_image_path,
    prompt,
    eta,
    lr,
    custom_steps,
    pre_steps,
    optim_steps,
    guidance_scale,
    device,
):
    use_amp = device.type == "cuda"
    os.makedirs(logdir, exist_ok=True)

    if pre_steps <= 0 or pre_steps >= custom_steps:
        raise ValueError("pre_steps must be > 0 and < custom_steps")

    shutil.copy(style_image_path, os.path.join(logdir, "gt.png"))
    gt_img = load_image_512(style_image_path)
    ref_img = np.array(gt_img) / 255.0 * 2.0 - 1.0
    ref_img = torch.from_numpy(ref_img[np.newaxis, ...].transpose(0, 3, 1, 2)).to(device)
    ref_img = ref_img.to(unet.dtype)
    ref_img.requires_grad = False

    model_path_clip = "openai/clip-vit-large-patch14"
    clip_tokenizer = CLIPTokenizer.from_pretrained(model_path_clip)
    clip_model = CLIPModel.from_pretrained(model_path_clip, torch_dtype=unet.dtype)
    clip_text = clip_model.text_model.to(device)

    clip_feature, _ = clip.load("ViT-B/16", device=device)
    clip_feature.eval()
    for p in clip_feature.parameters():
        p.requires_grad_(False)

    unet.to(device).eval()
    vae.to(device).eval()

    text_emb_all = build_text_embeddings(clip_tokenizer, clip_text, prompt, device, unet.dtype)

    z = torch.randn(1, unet.in_channels, 512 // 8, 512 // 8, device=device, dtype=unet.dtype)
    x_t = ddim_guided_sampling(
        unet, scheduler, z, text_emb_all, guidance_scale, scheduler.timesteps[:pre_steps], eta, device
    )
    z = x_t.detach()
    z.requires_grad = True

    optimizer = torch.optim.Adam([{"params": z, "lr": lr}], betas=(0.9, 0.9))
    scaler = GradScaler(enabled=use_amp)

    losses = []
    for step in range(optim_steps):
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            for i, tt in enumerate(scheduler.timesteps[pre_steps:]):
                t = tt
                if i == 0:
                    z_cat = torch.cat([z] * 2)
                    noise_pred = unet(z_cat, t, encoder_hidden_states=text_emb_all).sample
                    noise_uncond, noise_cond = noise_pred.chunk(2)
                    noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                else:
                    with torch.no_grad():
                        x_t_cat = torch.cat([x_t] * 2)
                        noise_pred = unet(x_t_cat, t, encoder_hidden_states=text_emb_all).sample
                        noise_uncond, noise_cond = noise_pred.chunk(2)
                        noise_pred = (noise_uncond + guidance_scale * (noise_cond - noise_uncond)).detach()
                if i == 0:
                    x_t = scheduler.step(
                        noise_pred, tt, z, return_dict=True, use_clipped_model_output=True, eta=eta
                    ).prev_sample
                else:
                    x_t = scheduler.step(
                        noise_pred, tt, x_t, return_dict=True, use_clipped_model_output=True, eta=eta
                    ).prev_sample

            x_t_scaled = x_t / 0.18215
            out = vae.decode(x_t_scaled).sample
            out = torch.clamp(out, -1, 1)
            loss = (get_gram_matrix_residual(ref_img, out, clip_feature) ** 2).mean()

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
    parser.add_argument("--input_image", type=str, default=None, help="reference image path")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--img", type=int, default=None)
    parser.add_argument("--image_ext", type=str, default=".jpg")
    parser.add_argument("--prompt", type=str, default="A portrait of Caleb, a character from the Critical Role series.")
    parser.add_argument("--logdir", type=str, default="./results_sdo")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--custom_steps", type=int, default=50)
    parser.add_argument("--pre_steps", type=int, default=45)
    parser.add_argument("--optim_steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--revision", type=str, default="fp16")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"))
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
        use_auth_token=args.hf_token,
        revision=args.revision,
        torch_dtype=torch_dtype,
    )
    vae = AutoencoderKL.from_pretrained(
        args.model_path,
        subfolder="vae",
        use_auth_token=args.hf_token,
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

    input_path = resolve_input_path(args)
    run_name = resolve_run_name(args, input_path)
    logdir = os.path.join(args.logdir, run_name)
    dmplug_style_optimize(
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        logdir=logdir,
        style_image_path=input_path,
        prompt=args.prompt,
        eta=args.eta,
        lr=args.lr,
        custom_steps=args.custom_steps,
        pre_steps=args.pre_steps,
        optim_steps=args.optim_steps,
        guidance_scale=args.guidance_scale,
        device=device,
    )


if __name__ == "__main__":
    main()
