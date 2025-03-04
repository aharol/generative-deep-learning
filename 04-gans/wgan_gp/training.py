import utils
import time
import functools as ft
import mlx.core as mx
import mlx.nn as nn


class WGAN_GP:

    def __init__(self,
                 c_model, g_model,
                 c_optim, g_optim,
                 z_dim=100,
                 gp_weight=10):

        self.c_model = c_model
        self.g_model = g_model
        self.c_optim = c_optim
        self.g_optim = g_optim
        self.z_dim = z_dim
        self.gp_weight = gp_weight

    @property
    def metrics(self):
        return {
            "c_loss": [],
            "g_loss": [],
            "c_wass_loss": [],
            "c_gp": [],
            "samples_per_sec": []
        }

    def gradient_penalty(self, real_data, fake_data):
        """Graidient penalty term"""

        batch_size = real_data.shape[0]

        # calculate interpolation
        alpha = mx.random.normal(shape=(batch_size, 1, 1, 1))
        interpo_data = (1 - alpha) * real_data + alpha * fake_data

        grads = mx.grad(lambda x: self.c_model(x).sum())(interpo_data)
        grad_norm = mx.linalg.norm(grads.reshape(batch_size, -1), axis=-1)

        c_gp = mx.mean((grad_norm - 1.0) ** 2)
        return c_gp

    def c_loss_fn(self, real_data):
        """Loss function for the discriminator, aka critic"""

        batch_size = real_data.shape[0]

        # generate fake data
        z = mx.random.normal(shape=(batch_size, self.z_dim))
        fake_data = self.g_model(z)

        # get discriminator predictions
        real_pred = self.c_model(real_data)
        fake_pred = self.c_model(fake_data)

        # Compute losses
        real_loss = -real_pred.mean()  # Gradient ascent for real loss
        fake_loss = fake_pred.mean()   # Gradient descent for fake loss
        c_wass_loss = fake_loss + real_loss  # Wasserstein loss
        c_gp = self.gradient_penalty(real_data, fake_data)
        c_loss = c_wass_loss + self.gp_weight * c_gp

        return c_loss

    def g_loss_fn(self, batch_size):
        """Loss function for generator"""
        # generate fake data
        z = mx.random.normal(shape=(batch_size, self.z_dim))
        fake_data = self.g_model(z)
        # classify fake data
        fake_preds = self.c_model(fake_data)
        # obtain loss
        g_loss = -fake_preds.mean()
        return g_loss

    def train_epoch(self, data, num_critic_steps):

        # Defining a state to be captured as input/output
        # IMPORTANT: append 'mx.random.state' to the list
        # of states as the 'self.c_model' is using 'Dropout'
        # see: https://ml-explore.github.io/mlx/build/html/usage/compile.html
        c_state = [self.c_model.state, self.c_optim.state, mx.random.state]

        @ft.partial(mx.compile, inputs=c_state, outputs=c_state)
        def train_critic(x):
            loss_and_grad_fn_c = nn.value_and_grad(self.c_model, self.c_loss_fn)
            c_loss, c_grads = loss_and_grad_fn_c(x)
            self.c_optim.update(self.c_model, c_grads)
            return c_loss

        g_state = [
            self.c_model.state, self.c_optim.state,
            self.g_model.state, self.g_optim.state, mx.random.state]

        @ft.partial(mx.compile, inputs=g_state, outputs=g_state)
        def train_generator(batch_size):
            loss_and_grad_fn_g = nn.value_and_grad(self.g_model, self.g_loss_fn)
            g_loss, g_grads = loss_and_grad_fn_g(batch_size)
            self.g_optim.update(self.g_model, g_grads)
            return g_loss

        for batch_counter, batch in enumerate(data):

            x = mx.array(batch["image"])
            c_loss = train_critic(x)
            mx.eval(c_state)
            # Only update generator after running `num_critic_steps`
            if batch_counter % num_critic_steps == 0:
                batch_size = x.shape[0]
                g_loss = train_generator(batch_size)
                mx.eval(g_state)

            self.metrics["c_loss"].append(c_loss.item())
            self.metrics["g_loss"].append(g_loss.item())

            if batch_counter % (10 * num_critic_steps) == 0:
                print(
                    " | ".join(
                        (
                            f"Discriminator Loss {c_loss:.3f}",
                            f"Generator loss {g_loss:.3f}",
                        )
                    )
                )

        mean_c_loss = mx.mean(mx.array(self.metrics["c_loss"]))
        mean_g_loss = mx.mean(mx.array(self.metrics["g_loss"]))

        return mean_c_loss, mean_g_loss

    def train(self, data,
              num_epochs, num_critic_steps,
              save_every_epoch, save_dir,
              num_samples_check=64):

        save_dir = utils.ensure_exists(save_dir)

        mx.eval(self.c_model.parameters())
        mx.eval(self.g_model.parameters())

        for epoch in range(num_epochs):

            # reset data iterator before each new epoch
            data.reset()

            c_loss, g_loss = self.train_epoch(data, num_critic_steps)

            self.metrics["c_loss"].append(c_loss)
            self.metrics["g_loss"].append(g_loss)
            # self.metrics["samples_per_sec"].append(throughput)
            # self.metrics["c_gp"].append()
            # self.metrics["c_wass_loss"].append()

            print("-"*100)
            print(
                " | ".join(
                    (
                        f"Epoch: {epoch:.02d}",
                        f"avg. Train loss discriminator {c_loss.item():.3f}",
                        f"avg. Train loss generator {g_loss.item():.3f}",
                        # f"Throughput: {throughput.item():.2f} images/sec",
                    )
                )
            )
            print("-"*100)

            # plot some samples after save_interval epochs
            if epoch % save_every_epoch == 0 or epoch == (num_epochs - 1):
                z = mx.random.normal(shape=(num_samples_check, self.z_dim))  # 128 random vectors
                fake_images = mx.array(self.g_model(z))
                img = utils.grid_image_from_batch(fake_images, num_rows=num_samples_check // 8)
                img.save(save_dir / f"image_{epoch}.png")

