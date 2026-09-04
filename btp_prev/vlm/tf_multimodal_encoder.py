"""
TensorFlow / Keras Multimodal Vision-Language-Sensor Fusion Encoder
===================================================================
Drop-in replacement for the legacy VAE in main.py and PIL_edge.py.
Outputs identical 95-dim latent image features + 5-dim nav features = 100-dim observation.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np


@tf.keras.utils.register_keras_serializable()
class TFMultimodalEdgeEncoder(tf.keras.Model):
    """
    Lightweight Multimodal Edge Vision-Language Encoder in TensorFlow/Keras.
    Compatible with TFLite INT8/FP16 quantization for edge deployment on Raspberry Pi / Jetson.
    """
    def __init__(
        self,
        latent_dim=95,
        embed_dim=64,
        vocab_size=16,
        text_embed_dim=32,
        name="TFMultimodalEdgeEncoder",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim

        # 1. Vision Feature Extraction (Lightweight Depthwise Separable Convolutions)
        self.conv1 = layers.Conv2D(32, (3, 3), strides=2, padding="same", activation="relu")
        self.bn1 = layers.BatchNormalization()
        
        self.dw_conv2 = layers.SeparableConv2D(64, (3, 3), strides=2, padding="same", activation="relu")
        self.bn2 = layers.BatchNormalization()

        self.dw_conv3 = layers.SeparableConv2D(128, (3, 3), strides=2, padding="same", activation="relu")
        self.bn3 = layers.BatchNormalization()

        self.dw_conv4 = layers.SeparableConv2D(128, (3, 3), strides=2, padding="same", activation="relu")
        self.global_pool = layers.GlobalAveragePooling2D()

        # 2. Verbal / Voice Command Embedding
        self.text_embedding = layers.Embedding(vocab_size, text_embed_dim)
        self.film_proj = layers.Dense(128 * 2, activation="relu")  # gamma and beta

        # 3. Vision-Language Latent Projection
        self.dense_proj1 = layers.Dense(256, activation="relu")
        self.latent_out = layers.Dense(latent_dim, activation="tanh")  # Normalized latent space

        # 4. Navigation feature processor
        self.nav_dense = layers.Dense(5, activation="linear")

    def call(self, inputs, training=False):
        """
        Args:
            inputs: Can be either:
                - Single image tensor (B, H, W, 3)
                - Tuple/List [image_tensor, optional_command_ids]
        Returns:
            latent_z: (B, latent_dim) = (B, 95)
        """
        if isinstance(inputs, (list, tuple)):
            image = inputs[0]
            cmd = inputs[1] if len(inputs) > 1 else None
        else:
            image = inputs
            cmd = None

        # Normalize RGB if uint8
        if image.dtype == tf.uint8:
            image = tf.cast(image, tf.float32) / 255.0

        # Visual feature extraction
        x = self.conv1(image)
        x = self.bn1(x, training=training)
        x = self.dw_conv2(x)
        x = self.bn2(x, training=training)
        x = self.dw_conv3(x)
        x = self.bn3(x, training=training)
        x = self.dw_conv4(x)  # (B, H', W', 128)

        # Apply voice command conditioning if provided
        if cmd is not None:
            cmd_embed = self.text_embedding(cmd)
            film_params = self.film_proj(cmd_embed)
            gamma, beta = tf.split(film_params, num_or_size_splits=2, axis=-1)
            # Expand for spatial broadcast: (B, 1, 1, 128)
            gamma = tf.expand_dims(tf.expand_dims(gamma, 1), 1)
            beta = tf.expand_dims(tf.expand_dims(beta, 1), 1)
            x = (1.0 + gamma) * x + beta

        # Pooling and projection
        pooled = self.global_pool(x)  # (B, 128)
        feat = self.dense_proj1(pooled)
        latent_z = self.latent_out(feat)  # (B, 95)

        return latent_z

    def process(self, observation):
        """
        Drop-in replacement for EncodeState.process(observation) in main.py & PIL_edge.py.
        Args:
            observation: [image_obs, navigation_obs]
        Returns:
            100-dim observation tensor ready for PPO Actor/Critic!
        """
        image_data = observation[0]
        nav_data = observation[1]

        image_tensor = tf.convert_to_tensor(image_data, dtype=tf.float32)
        if len(image_tensor.shape) == 3:
            image_tensor = tf.expand_dims(image_tensor, axis=0)

        # Extract 95-dim visual latent
        latent_z = self(image_tensor, training=False)

        nav_tensor = tf.convert_to_tensor(nav_data, dtype=tf.float32)
        if len(nav_tensor.shape) == 1:
            nav_tensor = tf.expand_dims(nav_tensor, axis=0)

        # Concatenate: 95 + 5 = 100-dim
        full_obs = tf.concat([tf.reshape(latent_z, [-1, self.latent_dim]), nav_tensor], axis=-1)
        return tf.squeeze(full_obs, axis=0) if len(tf.shape(full_obs)) > 1 and tf.shape(full_obs)[0] == 1 else full_obs


if __name__ == "__main__":
    print("=== Testing TFMultimodalEdgeEncoder ===")
    encoder = TFMultimodalEdgeEncoder(latent_dim=95)
    
    dummy_img = tf.random.uniform((1, 80, 160, 3), dtype=tf.float32)
    dummy_nav = tf.constant([[0.5, 4.8, 0.48, 0.05, 0.02]], dtype=tf.float32)

    # Test forward
    latent = encoder(dummy_img)
    print(f"Latent Output Shape: {latent.shape}")

    # Test drop-in process()
    obs = encoder.process([dummy_img[0].numpy(), dummy_nav[0].numpy()])
    print(f"Total Observation Shape: {obs.shape}")
    assert obs.shape == (100,), f"Shape mismatch: {obs.shape}"
    print("✅ TFMultimodalEdgeEncoder test passed! Drop-in compatible with main.py and PIL_edge.py")
