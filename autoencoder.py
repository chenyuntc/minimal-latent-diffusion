from tqdm import tqdm
import torch
import torchvision as tv
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from diffusers import AutoencoderKL
from util import dist, dist_sync_grad, set_seed_all

dist.init()
torch.cuda.set_device(dist.local_rank())
device = torch.device("cuda")
set_seed_all(dist.rank())

# Hyperparameters
BATCH_SIZE = 32 // 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
IMAGE_SIZE = 256
LATENT_DIM = 4  # downsample factor
KL_WEIGHT = 1e-6  # very small KL weight as mentioned in the paper
# DATA_ROOT = "/x/data/celeba" # a folder that "{DATA_ROOT}/<subfolder>/*.png"
DATA_ROOT = "/home/yun/imgs/"
NUM_WORKERS = 2

# Data Loading
img_transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
dataset = ImageFolder(DATA_ROOT, transform=img_transform)
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

# Model
model = AutoencoderKL(
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
).to(device)
for param in list(model.parameters()) + list(
    model.buffers()
):  # sync all parameters and buffers
    dist.broadcast(param.data, src=0)

# Optimizer
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

# Main training loop
model.train()
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    total_rec_loss = 0
    total_kl_loss = 0
    total_psnr = 0
    for batch_idx, (x, _) in tqdm(enumerate(dataloader)):
        x = x.to(device)

        # Forward pass
        posterior = model.encode(x)
        latents = posterior.latent_dist.sample()
        reconstruction = model.decode(latents).sample

        # Calculate losses
        rec_loss = F.mse_loss(reconstruction, x)
        kl_loss = posterior.latent_dist.kl().mean()
        loss = rec_loss + KL_WEIGHT * kl_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        dist_sync_grad(model)
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_rec_loss += rec_loss.item()
        total_kl_loss += kl_loss.item()
        psnr = 10 * torch.log10(1 / rec_loss)
        total_psnr += psnr.item()

        if batch_idx % 100 == 0 and dist.rank() == 0:
            print(
                f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} "
                f"Rec: {rec_loss.item():.4f} "
                f"KL: {kl_loss.item():.4f}"
                f"Avg Loss: {total_loss / (batch_idx + 1):.4f}"
                f"Avg Rec: {total_rec_loss / (batch_idx + 1):.4f}"
                f"Avg KL: {total_kl_loss / (batch_idx + 1):.4f}"
                f"Avg PSNR: {total_psnr / (batch_idx + 1):.4f}"
            )

    avg_rec_loss = total_rec_loss / len(dataloader)
    avg_kl_loss = total_kl_loss / len(dataloader)
    avg_loss = total_loss / len(dataloader)
    print(
        f"Epoch {epoch} Average Loss: {avg_loss:.4f} Rec: {avg_rec_loss:.4f} KL: {avg_kl_loss:.4f}"
    )
    tv.utils.save_image(
        reconstruction.cpu().detach().clamp(-1, 1) * 0.5 + 0.5,
        f"./reconstruction_{epoch}.png",
        nrow=4,
    )
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        },
        f"autoencoder_checkpoint_epoch_{epoch}.pt",
    )
