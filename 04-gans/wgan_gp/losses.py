import mlx.core as mx


def c_loss_fn(c_model, g_model, real_data, gp_weight, z_dim):
    """Loss function for discriminator (i.e. critic)"""

    batch_size = real_data.shape[0]

    # generate latent variable
    z = mx.random.normal(shape=(batch_size, z_dim))
    fake_data = g_model(z)

    # get discriminator predictions for real/fake data
    real_pred = c_model(real_data)
    fake_pred = c_model(fake_data)

    # Compute losses
    real_loss = real_pred.mean()  # Gradient ascent for real loss
    fake_loss = fake_pred.mean()   # Gradient descent for fake loss
    c_wass_loss = fake_loss - real_loss  # Wasserstein loss
    c_gp = gradient_penalty(c_model, real_data, fake_data)
    c_loss = c_wass_loss + gp_weight * c_gp

    return c_loss, c_wass_loss, c_gp


def gradient_penalty(c_model, real_data, fake_data):
    """Gradient penalty term"""
    batch_size = real_data.shape[0]
    # interpolate data
    alpha = mx.random.normal(shape=(batch_size, 1, 1, 1))
    interpo_data = (1 - alpha) * real_data + alpha * fake_data
    grads = mx.grad(lambda x: c_model(x).sum())(interpo_data)
    grad_norm = mx.linalg.norm(grads.reshape(batch_size, -1), axis=-1)
    c_gp = mx.mean((grad_norm - 1.0) ** 2)
    return c_gp


def g_loss_fn(c_model, g_model, batch_size, z_dim):
    """Loss function for generator"""
    # generate fake data
    z = mx.random.normal(shape=(batch_size, z_dim))
    # generate and classify fake data
    fake_preds = c_model(g_model(z))
    # obtain loss
    g_loss = -fake_preds.mean()
    return g_loss
