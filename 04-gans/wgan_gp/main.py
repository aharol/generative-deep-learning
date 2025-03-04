import time
import argparse
import functools as ft
from pathlib import Path
from tqdm.auto import tqdm
from dataclasses import dataclass, field
import models
import dataset
import losses
import utils
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import matplotlib.pyplot as plt  # Visualization
from matplotlib.ticker import MaxNLocator


@dataclass
class Metrics:

    c_loss: list[float] = field(default_factory=list)
    g_loss: list[float] = field(default_factory=list)
    c_wass_loss: list[float] = field(default_factory=list)
    c_gp: list[float] = field(default_factory=list)
    thrp: list[float] = field(default_factory=list)

    def plot_and_save(self, save_dir):
        """Plot the metrics and save to a file in 'save_dir'"""
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15,9))
        ax[0, 0].plot(range(1, len(self.c_loss) + 1), self.c_loss, 'r')
        ax[0, 0].title.set_text("c_loss") #row=0, col=0
        ax[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))  # integer xaxis
        ax[0, 1].plot(range(1, len(self.g_loss) + 1), self.g_loss, 'b') #row=1, col=0
        ax[0, 1].title.set_text("g_loss") #row=0, col=0
        ax[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True)) # integer xaxis
        ax[1, 0].plot(range(1, len(self.c_wass_loss) + 1), self.c_wass_loss, 'g') #row=0, col=1
        ax[1, 0].title.set_text("c_wass_loss") #row=0, col=0
        ax[1, 0].xaxis.set_major_locator(MaxNLocator(integer=True)) # integer xaxis
        ax[1, 1].plot(range(1, len(self.c_gp) + 1), self.c_gp, 'w') #row=1, col=1
        ax[1, 1].title.set_text("c_gp") #row=0, col=0
        ax[1, 1].xaxis.set_major_locator(MaxNLocator(integer=True)) # integer xaxis
        img_path = utils.ensure_exists(Path(save_dir) / "metrics")
        fig.savefig(img_path / "training.png")


def train_epoch(c_model, c_loss_fn, c_optim,
                g_model, g_loss_fn, g_optim,
                critic_steps, gp_weight, z_dim,
                data, data_size, epoch):

    # local loss collector to be averaged for each epoch
    epoch_losses = {
        "c_loss": [],
        "g_loss": [],
        "c_wass_loss": [],
        "c_gp": [],
        "thrp": []
    }

    # Critic state to monitor
    c_state = [
        c_model.state, c_optim.state,
        g_model.state, mx.random.state
    ]

    @ft.partial(mx.compile, inputs=c_state, outputs=c_state)
    def train_critic(x):
        loss_and_grad_fn_c = nn.value_and_grad(c_model, c_loss_fn)
        (c_loss, c_wass_loss, c_gp), c_grads = loss_and_grad_fn_c(
                c_model, g_model, x, gp_weight, z_dim)
        c_optim.update(c_model, c_grads)
        return c_loss, c_wass_loss, c_gp

    # Discriminator state to monitor
    g_state = [
        c_model.state, c_optim.state,
        g_model.state, g_optim.state,
        mx.random.state
    ]

    @ft.partial(mx.compile, inputs=g_state, outputs=g_state)
    def train_generator(batch_size):
        loss_and_grad_fn_g = nn.value_and_grad(g_model, g_loss_fn)
        g_loss, g_grads = loss_and_grad_fn_g(
            c_model, g_model, batch_size, z_dim)
        g_optim.update(g_model, g_grads)
        return g_loss

    with tqdm(total=data_size, desc=f"Epoch: {epoch:02d}") as pbar:

        for batch_counter, batch in enumerate(data):

            tic = time.perf_counter()
            x = mx.array(batch["image"])
            c_loss, c_wass_loss, c_gp = train_critic(x)
            mx.eval(c_state)
            # Only update generator after running `num_critic_steps`
            if batch_counter % critic_steps == 0:
                batch_size = x.shape[0]
                g_loss = train_generator(batch_size)
                mx.eval(g_state)
            toc = time.perf_counter()
            thrp = x.shape[0] / (toc - tic)

            epoch_losses["c_loss"].append(c_loss)
            epoch_losses["g_loss"].append(g_loss)
            epoch_losses["c_wass_loss"].append(c_wass_loss)
            epoch_losses["c_gp"].append(c_gp)
            epoch_losses["thrp"].append(thrp)

            pbar.update(x.shape[0])

    # averaging out the epoch losses
    epoch_losses = {k: mx.array(v).mean() for k, v in epoch_losses.items()}

    return epoch_losses


def main(args):

    img_size = (64, 64, 3)

    data_buf = dataset.load_celeba(split="train")
    data_size = len(data_buf)

    data = (
        data_buf
        .shuffle()
        .to_stream()
        .image_resize("image", h=img_size[0], w=img_size[1])
        .key_transform("image", lambda x: (x.astype("float32") - 127.5) / 127.5)
        .batch(args.batch_size)
        .prefetch(prefetch_size=8, num_threads=8)
    )

    # models for the Discriminator, aka Critic, and the Generator
    c_model = models.Critic(input_dim=img_size[-1], output_dim=1)
    g_model = models.Generator(input_dim=args.z_dim, output_dim=img_size[-1])

    mx.eval(c_model.parameters())
    mx.eval(g_model.parameters())

    # optimizers
    c_optim = optim.AdamW(learning_rate=1e-4, betas=[0.9, 0.999], weight_decay=0.01)
    g_optim = optim.AdamW(learning_rate=1e-4, betas=[0.9, 0.999], weight_decay=0.01)

    # metrics
    metrics = Metrics()

    for epoch in range(args.epochs):

        data.reset()

        epoch_losses = train_epoch(
            c_model, losses.c_loss_fn, c_optim,
            g_model, losses.g_loss_fn, g_optim,
            args.critic_steps, args.gp_weight, args.z_dim,
            data, data_size, epoch)

        for k, v in epoch_losses.items():
            metrics.__dict__[k].append(v)

        print("-"*120)
        print(
            " | ".join([
                f"Epoch: {epoch:02d}",
                f"avg. Critic loss: {epoch_losses["c_loss"]:.3f}",
                f"avg. Generator loss: {epoch_losses["g_loss"]:.3f}",
                f"avg. Throughput: {epoch_losses["thrp"]:.2f} images/second",
                f"avg. Wasserstein loss: {epoch_losses["c_wass_loss"]:.3f}",
                f"avg. Gradient penalty: {epoch_losses["c_gp"]:.3f}",
            ])
        )
        print("-"*120)

        # plot some samples after save_interval epochs
        if epoch % args.save_every_epoch == 0:
            z = mx.random.normal(shape=(args.batch_size, args.z_dim))  # 128 random vectors
            fake_images = mx.array(g_model(z))
            img = utils.grid_image_from_batch(fake_images, num_rows=8)
            img_path = utils.ensure_exists(Path(args.save_dir) / "images")
            img.save(img_path / f"image_{epoch}.png")

    metrics.plot_and_save(args.save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true",
                        help="Use CPU instead of GPU acceleration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for training")
    parser.add_argument("--save-dir", type=str,
                        default=(Path(__file__).parent / "artifacts").as_posix(),
                        help="Path to save the model and reconstructed images.")
    parser.add_argument("--save-every-epoch", type=int, default=10,
                        help="Save data epoch frequency")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--z-dim", type=int,default=128,
                        help="Number of latent dimensions (positive integer)")
    parser.add_argument("--critic-steps", type=int, default=5,
                        help="Critic steps")
    parser.add_argument("--gp-weight", type=float, default=10.0,
                        help="Gradient penalty")
    parser.add_argument("--debug", action="store_true",
                        help="Run in debug mode (i.e mx.disable_comile)")
    args = parser.parse_args()

    if args.cpu:
        mx.set_default_device(mx.cpu)

    if args.debug:
        mx.disable_compile()

    mx.random.seed(args.seed)

    print('='*100)
    print("Options: ")
    print('='*100)
    print(f"  Device: {'GPU' if not args.cpu else 'CPU'}")
    print(f"  Seed: {args.seed}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Number of latent dimensions: {args.z_dim}")
    print(f"  Critic steps: {args.critic_steps}")
    print(f"  Gradient penalty weight: {args.gp_weight}")
    print(f"  Save every epoch: {args.save_every_epoch}")
    print(f"  Saving directory: {args.save_dir}")
    print(f"  Compilation enabled: {not args.debug}")
    print('='*100)

    main(args)
