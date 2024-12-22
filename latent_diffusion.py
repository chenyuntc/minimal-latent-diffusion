from tqdm import trange
import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, AutoencoderKL, DDIMScheduler
from util import dist, dist_sync_grad, set_seed_all

dist.init()
torch.cuda.set_device(dist.local_rank())
device = torch.device("cuda")
set_seed_all(dist.rank())

# Hyperparameters
BATCH_SIZE = 32 * 3 # adjust if needed 
LEARNING_RATE = 4e-4
NUM_EPOCHS = 200
IMAGE_SIZE = 256
LATENT_CHANNELS = 4
AUTOENCODER_CKPT_PATH = "./autoencoder_checkpoint_epoch_7.pt"
DATA_ROOT = "/home/yun/imgs/"
NUM_WORKERS = 2

# Data Loading
img_transform = tv.transforms.Compose(
    [
        tv.transforms.Resize(IMAGE_SIZE),
        tv.transforms.CenterCrop(IMAGE_SIZE),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
dataset = tv.datasets.ImageFolder(DATA_ROOT, transform=img_transform)
data_sampler = torch.utils.data.distributed.DistributedSampler(
    dataset, shuffle=True, rank=dist.rank(), num_replicas=dist.size()
)
dataloader = DataLoader(
    dataset,
    sampler=data_sampler,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

# Load pretrained autoencoder
autoencoder = (
    AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=[
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ],
        up_block_types=[
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ],
        block_out_channels=(128, 256, 512, 512),
        latent_channels=4,
        sample_size=IMAGE_SIZE,
    )
    .cuda()
    .eval()
)
state_dict = torch.load(AUTOENCODER_CKPT_PATH, map_location=device)["model_state_dict"]
autoencoder.load_state_dict(state_dict)

# Create UNet for diffusion in latent space
latent_size = IMAGE_SIZE // 8  # Because autoencoder downsamples by factor of 8
model = UNet2DModel(
    sample_size=latent_size,
    in_channels=LATENT_CHANNELS,  # Matches autoencoder latent channels
    out_channels=LATENT_CHANNELS,
    layers_per_block=4,
    block_out_channels=(128, 256, 512, 512),
    down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
).cuda()
for param in list(model.parameters()) + list(model.buffers()):
    dist.broadcast(param.data, src=0)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# noise schedule
num_train_timesteps = 1000
num_inference_timesteps = 50
scheduler = DDIMScheduler(
    num_train_timesteps=num_train_timesteps,
    clip_sample=False,  # Important for training stability
    prediction_type="epsilon",  # Predict noise
)
scheduler.set_timesteps(num_inference_timesteps)

# Training loop
def train_one_epoch(epoch):
    model.train()
    total_loss = 0
    num_batches = 0
    losses = []

    for i, (x, _) in enumerate(dataloader):
        # Encode images to latent space
        with torch.no_grad():
            posterior = autoencoder.encode(x.cuda())
            latents = posterior.latent_dist.sample()
            latents = latents * 0.18215  # Scale factor from latent diffusion

        # Sample noise/timesteps, and add noise to latents
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, num_train_timesteps, (x.shape[0],)).cuda()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        # Predict noise
        noise_pred = model(noisy_latents, timesteps).sample

        # Calculate loss
        loss = nn.functional.mse_loss(noise_pred, noise)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        dist_sync_grad(model)
        optimizer.step()

        total_loss += loss.item()
        losses.append(loss.item())
        num_batches += 1

        if (i + 1) % 100 == 0 and dist.rank() == 0:
            mean_loss = sum(losses[-100:]) / len(losses[-100:])
            print(f"Epoch {epoch} [{i}/{len(dataloader)}] Loss: {mean_loss:.4f}")
            losses = []

    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def generate_samples(num_samples=16):
    model.eval()
    # Start from fix noise
    generator = torch.Generator().manual_seed(0)
    latents = torch.randn(
        num_samples, LATENT_CHANNELS, latent_size, latent_size, generator=generator
    ).cuda()

    # Denoise
    for t in scheduler.timesteps:
        noise_pred = model(latents, t.repeat(num_samples).cuda()).sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # Decode latents to images
    latents = 1 / 0.18215 * latents
    images = autoencoder.decode(latents).sample
    model.train()
    return images.clamp(-1, 1)


# Main training loop
for epoch in trange(NUM_EPOCHS):
    avg_loss = train_one_epoch(epoch)
    if dist.rank() == 0:
        print("--------------------")
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
        print("--------------------")

        # Generate and save samples
        samples = generate_samples()
        tv.utils.save_image(
            samples.clamp(-1, 1).mul(0.5).add(0.5),
            f"./samples_epoch_{epoch}.png",
            nrow=4,
        )

        # Save checkpoint
        if epoch % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "ae": autoencoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                f"./latent_diffusion_checkpoint_{epoch}.pt",
            )
        if epoch == 100:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= 0.1
