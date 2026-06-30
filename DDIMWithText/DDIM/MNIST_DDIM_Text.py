import argparse
import math
import re
import unicodedata
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


DIGIT_WORDS = {
    0: ["0", "zero", "ling", "零", "〇"],
    1: ["1", "one", "yi", "一", "壹"],
    2: ["2", "two", "er", "二", "两", "贰"],
    3: ["3", "three", "san", "三", "叁"],
    4: ["4", "four", "si", "四", "肆"],
    5: ["5", "five", "wu", "五", "伍"],
    6: ["6", "six", "liu", "六", "陆"],
    7: ["7", "seven", "qi", "七", "柒"],
    8: ["8", "eight", "ba", "八", "捌"],
    9: ["9", "nine", "jiu", "九", "玖"],
}


class DigitTextClassifier:
    """A tiny text classifier that maps prompts such as 'give digit 2' to label 2."""

    def predict(self, text):
        return classify_digit_prompt(text)


def classify_digit_prompt(text):
    normalized = unicodedata.normalize("NFKC", text).strip().lower()
    if not normalized:
        raise ValueError("Empty prompt. Example: 给出数字2")

    digit_matches = re.findall(r"[0-9]", normalized)
    if digit_matches:
        return int(digit_matches[-1])

    for digit, words in DIGIT_WORDS.items():
        for word in words:
            word = word.lower()
            if re.fullmatch(r"[a-z]+", word):
                if re.search(rf"\b{re.escape(word)}\b", normalized):
                    return digit
            elif word in normalized:
                return digit

    raise ValueError(f"Cannot find a digit in prompt: {text!r}")


def parse_digit_sequence(text):
    normalized = unicodedata.normalize("NFKC", text).strip().lower()
    if not normalized:
        raise ValueError("Empty sequence. Example: 612389")

    digit_matches = re.findall(r"[0-9]", normalized)
    if digit_matches:
        return [int(digit) for digit in digit_matches]

    token_to_digit = {
        word.lower(): digit
        for digit, words in DIGIT_WORDS.items()
        for word in words
    }
    sequence = []
    for char in normalized:
        if char in token_to_digit and not re.fullmatch(r"[a-z]", char):
            sequence.append(token_to_digit[char])

    if sequence:
        return sequence

    for token in re.findall(r"[a-z]+", normalized):
        if token in token_to_digit:
            sequence.append(token_to_digit[token])

    if sequence:
        return sequence

    raise ValueError(f"Cannot find a digit sequence in: {text!r}")


def group_norm(channels):
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


def sinusoidal_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(half, device=timesteps.device).float()
        / max(half - 1, 1)
    )
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.norm1 = group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, out_channels)
        self.norm2 = group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.cond_proj(cond)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class ConditionalUNet(nn.Module):
    def __init__(self, image_channels=1, base_channels=64, cond_dim=256, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.null_label = num_classes

        self.time_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.label_embedding = nn.Embedding(num_classes + 1, cond_dim)

        self.input = nn.Conv2d(image_channels, base_channels, 3, padding=1)
        self.down1 = ResBlock(base_channels, base_channels, cond_dim)
        self.downsample1 = nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1)
        self.down2 = ResBlock(base_channels * 2, base_channels * 2, cond_dim)
        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1)

        self.middle1 = ResBlock(base_channels * 4, base_channels * 4, cond_dim)
        self.middle2 = ResBlock(base_channels * 4, base_channels * 4, cond_dim)

        self.upsample1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        self.up1 = ResBlock(base_channels * 4, base_channels * 2, cond_dim)
        self.upsample2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.up2 = ResBlock(base_channels * 2, base_channels, cond_dim)

        self.output = nn.Sequential(
            group_norm(base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, image_channels, 3, padding=1),
        )

    def make_condition(self, timesteps, labels):
        if labels is None:
            labels = torch.full(
                (timesteps.shape[0],),
                self.null_label,
                device=timesteps.device,
                dtype=torch.long,
            )
        time_cond = self.time_mlp(sinusoidal_embedding(timesteps, self.time_mlp[0].in_features))
        label_cond = self.label_embedding(labels)
        return time_cond + label_cond

    def forward(self, x, timesteps, labels=None):
        cond = self.make_condition(timesteps, labels)

        x0 = self.input(x)
        d1 = self.down1(x0, cond)
        d2 = self.down2(self.downsample1(d1), cond)
        mid = self.middle1(self.downsample2(d2), cond)
        mid = self.middle2(mid, cond)

        u1 = self.upsample1(mid)
        u1 = self.up1(torch.cat([u1, d2], dim=1), cond)
        u2 = self.upsample2(u1)
        u2 = self.up2(torch.cat([u2, d1], dim=1), cond)
        return self.output(u2)


class Diffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def extract(self, values, timesteps, x_shape):
        out = values.gather(0, timesteps)
        return out.reshape(timesteps.shape[0], *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, timesteps, noise):
        return (
            self.extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
            + self.extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape) * noise
        )


def mnist_loader(data_dir, image_size, batch_size, num_workers):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = datasets.MNIST(data_dir, train=True, transform=transform, download=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device, but torch.cuda.is_available() is False.")
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    sample_dir = output_dir / "samples"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    loader = mnist_loader(args.data_dir, args.image_size, args.batch_size, args.num_workers)
    model = ConditionalUNet(base_channels=args.base_channels).to(device)
    diffusion = Diffusion(args.timesteps, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"training device: {device}")
    if device.type == "cuda":
        print(f"cuda device: {torch.cuda.get_device_name(device)}")
    print(f"model parameter device: {next(model.parameters()).device}")
    print(f"data loader workers: {args.num_workers}")

    start_epoch = 1
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        print(f"Resumed from {args.resume}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            timesteps = torch.randint(0, diffusion.timesteps, (images.shape[0],), device=device).long()
            noise = torch.randn_like(images)
            noisy_images = diffusion.q_sample(images, timesteps, noise)

            train_labels = labels.clone()
            if args.drop_label_prob > 0:
                drop_mask = torch.rand(labels.shape[0], device=device) < args.drop_label_prob
                train_labels[drop_mask] = model.null_label

            predicted_noise = model(noisy_images, timesteps, train_labels)
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            if batch_idx % args.log_every == 0:
                avg = running_loss / args.log_every
                print(f"epoch={epoch} batch={batch_idx}/{len(loader)} loss={avg:.4f}")
                running_loss = 0.0

            if args.limit_train_batches and batch_idx >= args.limit_train_batches:
                break

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "timesteps": args.timesteps,
            "base_channels": args.base_channels,
            "image_size": args.image_size,
            "drop_label_prob": args.drop_label_prob,
        }
        torch.save(checkpoint, checkpoint_dir / "latest.pt")
        torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch:03d}.pt")
        print(f"saved checkpoint: {checkpoint_dir / 'latest.pt'}")

        if args.sample_every > 0 and epoch % args.sample_every == 0:
            labels = torch.arange(10, device=device)
            samples = ddim_sample(
                model,
                diffusion,
                labels=labels,
                image_size=args.image_size,
                steps=args.sample_steps,
                eta=args.eta,
                guidance_scale=args.guidance_scale,
                device=device,
            )
            save_image((samples.clamp(-1, 1) + 1) * 0.5, sample_dir / f"epoch_{epoch:03d}.png", nrow=5)
            print(f"saved preview: {sample_dir / f'epoch_{epoch:03d}.png'}")


@torch.no_grad()
def ddim_sample(
    model,
    diffusion,
    labels,
    image_size=32,
    steps=50,
    eta=0.0,
    guidance_scale=1.5,
    device="cuda",
):
    model.eval()
    labels = labels.to(device).long()
    batch_size = labels.shape[0]
    x = torch.randn(batch_size, 1, image_size, image_size, device=device)

    steps = min(steps, diffusion.timesteps)
    times = torch.linspace(diffusion.timesteps - 1, 0, steps, device=device).long()
    times = torch.unique_consecutive(times)

    for index, timestep in enumerate(times):
        t = int(timestep.item())
        prev_t = int(times[index + 1].item()) if index + 1 < len(times) else -1
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

        eps_cond = model(x, t_batch, labels)
        if guidance_scale == 1.0:
            eps = eps_cond
        else:
            eps_uncond = model(x, t_batch, None)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        alpha_t = diffusion.alphas_cumprod[t]
        alpha_prev = (
            torch.tensor(1.0, device=device)
            if prev_t < 0
            else diffusion.alphas_cumprod[prev_t]
        )

        pred_x0 = (x - torch.sqrt(1.0 - alpha_t) * eps) / torch.sqrt(alpha_t)
        pred_x0 = pred_x0.clamp(-1.0, 1.0)

        if prev_t < 0:
            x = pred_x0
            continue

        sigma = eta * torch.sqrt((1.0 - alpha_prev) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_prev))
        direction = torch.sqrt(torch.clamp(1.0 - alpha_prev - sigma**2, min=0.0)) * eps
        noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
        x = torch.sqrt(alpha_prev) * pred_x0 + direction + sigma * noise

    return x.clamp(-1.0, 1.0)


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ConditionalUNet(base_channels=checkpoint.get("base_channels", 64)).to(device)
    model.load_state_dict(checkpoint["model"])
    diffusion = Diffusion(checkpoint.get("timesteps", 1000), device=device)
    image_size = checkpoint.get("image_size", 32)
    return model, diffusion, image_size


def sample(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device, but torch.cuda.is_available() is False.")
    text_classifier = DigitTextClassifier()
    digit = text_classifier.predict(args.prompt)
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, diffusion, image_size = load_model(checkpoint, device)
    print(f"sampling device: {device}")
    if device.type == "cuda":
        print(f"cuda device: {torch.cuda.get_device_name(device)}")
    print(f"model parameter device: {next(model.parameters()).device}")
    labels = torch.full((args.num_images,), digit, device=device, dtype=torch.long)
    samples = ddim_sample(
        model,
        diffusion,
        labels=labels,
        image_size=image_size,
        steps=args.sample_steps,
        eta=args.eta,
        guidance_scale=args.guidance_scale,
        device=device,
    )

    output_path = output_dir / f"prompt_digit_{digit}.png"
    save_image((samples + 1) * 0.5, output_path, nrow=min(args.num_images, 8))
    print(f"prompt: {args.prompt}")
    print(f"classified digit: {digit}")
    print(f"saved image: {output_path}")


def sample_sequence(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device, but torch.cuda.is_available() is False.")

    digits = parse_digit_sequence(args.sequence)
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, diffusion, image_size = load_model(checkpoint, device)
    print(f"sampling device: {device}")
    if device.type == "cuda":
        print(f"cuda device: {torch.cuda.get_device_name(device)}")
    print(f"model parameter device: {next(model.parameters()).device}")

    labels = torch.tensor(digits, device=device, dtype=torch.long)
    samples = ddim_sample(
        model,
        diffusion,
        labels=labels,
        image_size=image_size,
        steps=args.sample_steps,
        eta=args.eta,
        guidance_scale=args.guidance_scale,
        device=device,
    )

    sequence_text = "".join(str(digit) for digit in digits)
    output_path = output_dir / f"sequence_{sequence_text}.png"
    save_image(
        (samples + 1) * 0.5,
        output_path,
        nrow=len(digits),
        padding=args.padding,
        pad_value=args.pad_value,
    )
    print(f"sequence: {sequence_text}")
    print(f"labels: {digits}")
    print(f"saved image: {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train a text-prompted conditional DDIM on MNIST."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the conditional denoising model.")
    train_parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parent / "data"))
    train_parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "mnist_ddim_runs"))
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--image-size", type=int, default=32)
    train_parser.add_argument("--timesteps", type=int, default=1000)
    train_parser.add_argument("--sample-steps", type=int, default=50)
    train_parser.add_argument("--base-channels", type=int, default=64)
    train_parser.add_argument("--lr", type=float, default=2e-4)
    train_parser.add_argument("--eta", type=float, default=0.0)
    train_parser.add_argument("--guidance-scale", type=float, default=1.5)
    train_parser.add_argument("--drop-label-prob", type=float, default=0.1)
    train_parser.add_argument("--sample-every", type=int, default=1)
    train_parser.add_argument("--log-every", type=int, default=100)
    train_parser.add_argument("--num-workers", type=int, default=4)
    train_parser.add_argument("--limit-train-batches", type=int, default=0)
    train_parser.add_argument("--resume", default="")
    train_parser.add_argument("--device", default="cuda")
    train_parser.set_defaults(func=train)

    sample_parser = subparsers.add_parser("sample", help="Generate an MNIST digit from a text prompt.")
    sample_parser.add_argument("--prompt", required=True, help='Example: "给出数字2" or "give me digit two"')
    sample_parser.add_argument(
        "--checkpoint",
        default=str(Path(__file__).resolve().parent / "mnist_ddim_runs" / "checkpoints" / "latest.pt"),
    )
    sample_parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "mnist_ddim_runs" / "outputs"))
    sample_parser.add_argument("--num-images", type=int, default=8)
    sample_parser.add_argument("--sample-steps", type=int, default=50)
    sample_parser.add_argument("--eta", type=float, default=0.0)
    sample_parser.add_argument("--guidance-scale", type=float, default=1.5)
    sample_parser.add_argument("--device", default="cuda")
    sample_parser.set_defaults(func=sample)

    sequence_parser = subparsers.add_parser(
        "sample-sequence",
        help="Generate a left-to-right image strip for a digit sequence.",
    )
    sequence_parser.add_argument("--sequence", required=True, help='Example: "612389"')
    sequence_parser.add_argument(
        "--checkpoint",
        default=str(Path(__file__).resolve().parent / "mnist_ddim_runs" / "checkpoints" / "latest.pt"),
    )
    sequence_parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "mnist_ddim_runs" / "outputs"))
    sequence_parser.add_argument("--sample-steps", type=int, default=50)
    sequence_parser.add_argument("--eta", type=float, default=0.0)
    sequence_parser.add_argument("--guidance-scale", type=float, default=1.5)
    sequence_parser.add_argument("--padding", type=int, default=2)
    sequence_parser.add_argument("--pad-value", type=float, default=1.0)
    sequence_parser.add_argument("--device", default="cuda")
    sequence_parser.set_defaults(func=sample_sequence)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
