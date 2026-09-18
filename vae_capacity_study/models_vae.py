"""
Architectural variants for the VAE Capacity & Latent Space Scaling Study.

Provides three distinct model capacity levels:
  1. 'baseline'  - Original 4 Conv layers (32->64->128->256, Dense 1024, Latent dz) ~13.7M params
  2. 'wide'      - 4 Conv layers with 2x filter channels (64->128->256->512, Dense 2048, Latent dz) ~54.8M params
  3. 'deep'      - 6 Conv layers with deeper hierarchical abstraction (32->64->64->128->256->512, Dense 2048, Latent dz) ~28.5M params

All variants feature:
  - Bounded log-sigma clipping to [-15.0, 1.0] preventing NaN explosion.
  - Pure TensorFlow implementation (no tensorflow_probability dependency).
  - Deterministic mean vector (mu) output when training=False for stable RL control.
  - Configurable latent dimensionality dz (16, 32, 64, 95, 128, 190, 256).
"""
import os
import tensorflow as tf

layers = tf.keras.layers


# ==============================================================================
# 1. BASELINE VAE (Original 4-Layer Architecture, ~13.7M Params)
# ==============================================================================
@tf.keras.utils.register_keras_serializable()
class BaselineEncoder(tf.keras.Model):
    def __init__(self, latent_dim=95, name='BASELINE_ENCODER', **kwargs):
        super().__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.conv1 = layers.Conv2D(32, (4, 4), strides=2, padding='same', activation='relu', name='conv1')
        self.conv2 = layers.Conv2D(64, (3, 3), strides=2, padding='same', activation='relu', name='conv2')
        self.bn1 = layers.BatchNormalization(name='bn1')
        self.conv3 = layers.Conv2D(128, (4, 4), strides=2, padding='same', activation='relu', name='conv3')
        self.conv4 = layers.Conv2D(256, (3, 3), strides=2, padding='same', activation='relu', name='conv4')
        self.flatten = layers.Flatten(name='flatten')
        self.dense1 = layers.Dense(1024, activation='relu', name='dense1')
        self.mu = layers.Dense(self.latent_dim, name='mu')
        self.sigma = layers.Dense(self.latent_dim, name='sigma')
        self.kl = 0.0

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.bn1(x, training=training)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.dense1(x)

        mu = self.mu(x)
        clipped_log_sigma = tf.clip_by_value(self.sigma(x), -15.0, 1.0)
        sigma = tf.exp(clipped_log_sigma)
        
        # Exact KL divergence against standard normal prior N(0, I)
        self.kl = 0.5 * tf.reduce_sum(tf.square(sigma) + tf.square(mu) - 2.0 * clipped_log_sigma - 1.0, axis=-1)

        if not training:
            return mu  # Deterministic inference for RL policy
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + sigma * eps


@tf.keras.utils.register_keras_serializable()
class BaselineDecoder(tf.keras.Model):
    def __init__(self, latent_dim=95, name='BASELINE_DECODER', **kwargs):
        super().__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.dense1 = layers.Dense(1024, activation='leaky_relu', name='dense1')
        self.dense2 = layers.Dense(10 * 5 * 256, activation='leaky_relu', name='dense2')
        self.unflatten = layers.Reshape((10, 5, 256), name='reshape')
        self.deconv1 = layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation='leaky_relu', name='deconv1')
        self.deconv2 = layers.Conv2DTranspose(64, (4, 4), strides=2, padding='same', activation='leaky_relu', name='deconv2')
        self.deconv3 = layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='leaky_relu', name='deconv3')
        self.deconv4 = layers.Conv2DTranspose(3, (4, 4), strides=2, padding='same', activation='sigmoid', name='deconv4')

    def call(self, z):
        x = self.dense1(z)
        x = self.dense2(x)
        x = self.unflatten(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return self.deconv4(x)


# ==============================================================================
# 2. WIDE VAE (2x Filter Channels & Dense Neurons, ~54.8M Params)
# ==============================================================================
@tf.keras.utils.register_keras_serializable()
class WideEncoder(tf.keras.Model):
    def __init__(self, latent_dim=95, name='WIDE_ENCODER', **kwargs):
        super().__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        # Double filter channels across all conv layers
        self.conv1 = layers.Conv2D(64, (4, 4), strides=2, padding='same', activation='relu', name='conv1')
        self.conv2 = layers.Conv2D(128, (3, 3), strides=2, padding='same', activation='relu', name='conv2')
        self.bn1 = layers.BatchNormalization(name='bn1')
        self.conv3 = layers.Conv2D(256, (4, 4), strides=2, padding='same', activation='relu', name='conv3')
        self.conv4 = layers.Conv2D(512, (3, 3), strides=2, padding='same', activation='relu', name='conv4')
        self.flatten = layers.Flatten(name='flatten')
        # Double dense bottleneck neurons
        self.dense1 = layers.Dense(2048, activation='relu', name='dense1')
        self.mu = layers.Dense(self.latent_dim, name='mu')
        self.sigma = layers.Dense(self.latent_dim, name='sigma')
        self.kl = 0.0

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.bn1(x, training=training)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.dense1(x)

        mu = self.mu(x)
        clipped_log_sigma = tf.clip_by_value(self.sigma(x), -15.0, 1.0)
        sigma = tf.exp(clipped_log_sigma)
        
        self.kl = 0.5 * tf.reduce_sum(tf.square(sigma) + tf.square(mu) - 2.0 * clipped_log_sigma - 1.0, axis=-1)

        if not training:
            return mu
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + sigma * eps


@tf.keras.utils.register_keras_serializable()
class WideDecoder(tf.keras.Model):
    def __init__(self, latent_dim=95, name='WIDE_DECODER', **kwargs):
        super().__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.dense1 = layers.Dense(2048, activation='leaky_relu', name='dense1')
        self.dense2 = layers.Dense(10 * 5 * 512, activation='leaky_relu', name='dense2')
        self.unflatten = layers.Reshape((10, 5, 512), name='reshape')
        self.deconv1 = layers.Conv2DTranspose(256, (3, 3), strides=2, padding='same', activation='leaky_relu', name='deconv1')
        self.deconv2 = layers.Conv2DTranspose(128, (4, 4), strides=2, padding='same', activation='leaky_relu', name='deconv2')
        self.deconv3 = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='leaky_relu', name='deconv3')
        self.deconv4 = layers.Conv2DTranspose(3, (4, 4), strides=2, padding='same', activation='sigmoid', name='deconv4')

    def call(self, z):
        x = self.dense1(z)
        x = self.dense2(x)
        x = self.unflatten(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return self.deconv4(x)


# ==============================================================================
# 3. DEEP VAE (6 Conv Layers + Multi-Dense Bottleneck, ~28.5M Params)
# ==============================================================================
@tf.keras.utils.register_keras_serializable()
class DeepEncoder(tf.keras.Model):
    def __init__(self, latent_dim=95, name='DEEP_ENCODER', **kwargs):
        super().__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        # 6 Conv layers with intermediate stride-1 feature extraction
        self.conv1 = layers.Conv2D(32, (4, 4), strides=2, padding='same', activation='relu', name='conv1')   # (80, 40, 32)
        self.conv2 = layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu', name='conv2')   # (80, 40, 64)
        self.bn1   = layers.BatchNormalization(name='bn1')
        self.conv3 = layers.Conv2D(64, (3, 3), strides=2, padding='same', activation='relu', name='conv3')   # (40, 20, 64)
        self.conv4 = layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu', name='conv4')  # (40, 20, 128)
        self.bn2   = layers.BatchNormalization(name='bn2')
        self.conv5 = layers.Conv2D(128, (4, 4), strides=2, padding='same', activation='relu', name='conv5')  # (20, 10, 128)
        self.conv6 = layers.Conv2D(256, (3, 3), strides=2, padding='same', activation='relu', name='conv6')  # (10, 5, 256)
        
        self.flatten = layers.Flatten(name='flatten')
        self.dense1 = layers.Dense(2048, activation='relu', name='dense1')
        self.dense2 = layers.Dense(1024, activation='relu', name='dense2')
        self.mu = layers.Dense(self.latent_dim, name='mu')
        self.sigma = layers.Dense(self.latent_dim, name='sigma')
        self.kl = 0.0

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.bn1(x, training=training)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn2(x, training=training)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        mu = self.mu(x)
        clipped_log_sigma = tf.clip_by_value(self.sigma(x), -15.0, 1.0)
        sigma = tf.exp(clipped_log_sigma)
        
        self.kl = 0.5 * tf.reduce_sum(tf.square(sigma) + tf.square(mu) - 2.0 * clipped_log_sigma - 1.0, axis=-1)

        if not training:
            return mu
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + sigma * eps


@tf.keras.utils.register_keras_serializable()
class DeepDecoder(tf.keras.Model):
    def __init__(self, latent_dim=95, name='DEEP_DECODER', **kwargs):
        super().__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.dense1 = layers.Dense(1024, activation='leaky_relu', name='dense1')
        self.dense2 = layers.Dense(2048, activation='leaky_relu', name='dense2')
        self.dense3 = layers.Dense(10 * 5 * 256, activation='leaky_relu', name='dense3')
        self.unflatten = layers.Reshape((10, 5, 256), name='reshape')
        
        self.deconv1 = layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation='leaky_relu', name='deconv1')
        self.deconv2 = layers.Conv2DTranspose(128, (4, 4), strides=2, padding='same', activation='leaky_relu', name='deconv2')
        self.deconv3 = layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='leaky_relu', name='deconv3')
        self.deconv4 = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='leaky_relu', name='deconv4')
        self.deconv5 = layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='leaky_relu', name='deconv5')
        self.deconv6 = layers.Conv2DTranspose(3, (4, 4), strides=2, padding='same', activation='sigmoid', name='deconv6')

    def call(self, z):
        x = self.dense1(z)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.unflatten(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        return self.deconv6(x)


# ==============================================================================
# UNIFIED FULL VAE MODEL WRAPPER
# ==============================================================================
@tf.keras.utils.register_keras_serializable()
class FullVariationalAutoencoder(tf.keras.Model):
    def __init__(self, encoder, decoder, beta=1.0, name='FULL_VAE', **kwargs):
        super().__init__(name=name, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta

    def call(self, x, training=False):
        z = self.encoder(x, training=training)
        return self.decoder(z)

    def train_step(self, data):
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        with tf.GradientTape() as tape:
            reconstructed = self(x, training=True)
            # Reconstruction MSE loss across spatial pixel dimensions
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - reconstructed), axis=[1, 2, 3]))
            # Mean KL divergence across batch
            kl_loss = tf.reduce_mean(self.encoder.kl)
            total_loss = recon_loss + self.beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }


def build_vae(arch='baseline', latent_dim=95, beta=1.0):
    """
    Factory builder for VAE models.
    arch: 'baseline', 'wide', or 'deep'
    latent_dim: integer (e.g. 16, 32, 64, 95, 128, 190, 256)
    """
    arch = arch.lower().strip()
    if arch == 'baseline':
        encoder = BaselineEncoder(latent_dim=latent_dim)
        decoder = BaselineDecoder(latent_dim=latent_dim)
    elif arch == 'wide':
        encoder = WideEncoder(latent_dim=latent_dim)
        decoder = WideDecoder(latent_dim=latent_dim)
    elif arch == 'deep':
        encoder = DeepEncoder(latent_dim=latent_dim)
        decoder = DeepDecoder(latent_dim=latent_dim)
    else:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from: 'baseline', 'wide', 'deep'.")

    vae = FullVariationalAutoencoder(encoder, decoder, beta=beta)
    return vae
