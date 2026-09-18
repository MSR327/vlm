"""
Trainer script for VAE Capacity and Latent Space Scaling Study.

Trains Baseline, Wide, or Deep VAEs on Cityscapes semantic frames with:
  - Configurable capacity: --arch {baseline, wide, deep}
  - Configurable latent dimension: --latent {16, 32, 64, 95, 128, 190, 256}
  - Exact SavedModel signature export compatible with vae_loader.py:
      input_1:  (None, 160, 80, 3)
      output_1: (None, LATENT_DIM)
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
import tensorflow as tf

from models_vae import build_vae


def find_dataset(data_arg):
    candidates = [
        data_arg,
        os.path.join(os.path.dirname(__file__), data_arg),
        os.path.join(os.path.dirname(__file__), '../btp_prev/VAE/dataset'),
        'btp_prev/VAE/dataset',
        'VAE/dataset',
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, 'train')):
            return os.path.abspath(c)
    raise FileNotFoundError(f"Could not locate VAE dataset in any candidate path: {candidates}")


def build_data_pipelines(data_dir, batch_size=32, target_size=(160, 80)):
    train_dir = os.path.join(data_dir, 'train')
    test_dir  = os.path.join(data_dir, 'test')

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        horizontal_flip=True,
    )
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0
    )

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='input',
        shuffle=True
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='input',
        shuffle=False
    )
    return train_gen, test_gen


def export_saved_model(encoder, export_path, latent_dim):
    """
    Exports the trained encoder as a standard TensorFlow SavedModel
    matching the legacy serving signature (input_1 -> output_1).
    """
    os.makedirs(export_path, exist_ok=True)

    class ExportModule(tf.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc

        @tf.function(input_signature=[tf.TensorSpec(shape=(None, 160, 80, 3), dtype=tf.float32, name='input_1')])
        def serve(self, input_1):
            # Deterministic inference for policy stability
            out = self.enc(input_1, training=False)
            return {'output_1': out}

    module = ExportModule(encoder)
    tf.saved_model.save(module, export_path, signatures={'serving_default': module.serve})
    print(f"[export] SavedModel exported successfully to: {export_path}")


def main():
    parser = argparse.ArgumentParser(description="Train VAE Capacity & Latent Scaling Variants")
    parser.add_argument('--arch', type=str, default='baseline', choices=['baseline', 'wide', 'deep'],
                        help="VAE architecture: baseline (13.7M), wide (54.6M), or deep (29.2M)")
    parser.add_argument('--latent', type=int, default=95,
                        help="Latent space dimensionality dz (e.g. 16, 32, 64, 95, 128, 190, 256)")
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Adam learning rate")
    parser.add_argument('--beta', type=float, default=1.0,
                        help="Beta weighting on KL divergence loss")
    parser.add_argument('--data', type=str, default='dataset',
                        help="Path to VAE dataset")
    parser.add_argument('--out', type=str, default=None,
                        help="Output directory to save the trained model")
    args = parser.parse_args()

    # Configure GPU memory growth if available
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    data_dir = find_dataset(args.data)
    print(f"\n=======================================================")
    print(f" TRAINING VAE CAPACITY ABLATION")
    print(f" Architecture:      {args.arch.upper()}")
    print(f" Latent Dimension:  {args.latent}")
    print(f" Dataset Path:      {data_dir}")
    print(f" Epochs:            {args.epochs}, Batch: {args.batch_size}, LR: {args.lr:g}")
    print(f"=======================================================\n")

    train_gen, test_gen = build_data_pipelines(data_dir, batch_size=args.batch_size)

    # Instantiate model
    vae = build_vae(arch=args.arch, latent_dim=args.latent, beta=args.beta)
    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
    vae.compile(optimizer=opt)

    # Train steps
    train_steps = train_gen.samples // args.batch_size
    val_steps   = test_gen.samples // args.batch_size

    history = {'epoch': [], 'loss': [], 'recon_loss': [], 'kl_loss': [], 'val_recon_loss': []}

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        losses, recons, kls = [], [], []

        for step in range(train_steps):
            x_batch, _ = next(train_gen)
            m = vae.train_step(x_batch)
            losses.append(float(m['loss'].numpy()))
            recons.append(float(m['recon_loss'].numpy()))
            kls.append(float(m['kl_loss'].numpy()))

        # Validation evaluation
        val_recons = []
        for v_step in range(min(val_steps, 20)):
            x_val, _ = next(test_gen)
            rec_val = vae(x_val, training=False)
            val_mse = float(tf.reduce_mean(tf.reduce_sum(tf.square(x_val - rec_val), axis=[1, 2, 3])))
            val_recons.append(val_mse)

        mean_loss = float(np.mean(losses))
        mean_recon = float(np.mean(recons))
        mean_kl = float(np.mean(kls))
        mean_val_recon = float(np.mean(val_recons))

        history['epoch'].append(epoch)
        history['loss'].append(mean_loss)
        history['recon_loss'].append(mean_recon)
        history['kl_loss'].append(mean_kl)
        history['val_recon_loss'].append(mean_val_recon)

        elapsed = time.time() - t_ep
        print(f"Epoch {epoch:2d}/{args.epochs:2d} ({elapsed:.1f}s) | "
              f"Train Loss: {mean_loss:.4f} (Recon: {mean_recon:.4f}, KL: {mean_kl:.2f}) | "
              f"Val Recon MSE: {mean_val_recon:.4f}")

    total_time = time.time() - t_start
    print(f"\n[training] Finished {args.epochs} epochs in {total_time:.1f} seconds.")

    # Determine export path
    tag = f"vae_{args.arch}_{args.latent}"
    out_dir = args.out or os.path.join(os.path.dirname(__file__), 'models', tag, 'var_auto_encoder_model')
    export_saved_model(vae.encoder, out_dir, args.latent)

    # Save training metrics JSON
    metrics_path = os.path.join(os.path.dirname(out_dir), 'training_metrics.json')
    summary = {
        'arch': args.arch,
        'latent_dim': args.latent,
        'epochs': args.epochs,
        'final_train_recon_mse': history['recon_loss'][-1],
        'final_val_recon_mse': history['val_recon_loss'][-1],
        'final_kl_div': history['kl_loss'][-1],
        'training_wall_time_s': total_time,
        'history': history
    }
    with open(metrics_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[metrics] Saved training metrics to: {metrics_path}")


if __name__ == '__main__':
    main()
