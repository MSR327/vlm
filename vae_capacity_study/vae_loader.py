"""
TensorFlow-version-agnostic loader for the pretrained VAE encoder.

The saved encoder is a legacy TF SavedModel directory (March 2025).
How you load it depends on the TF/Keras version:

  TF <= 2.15 (Keras 2) : tf.keras.models.load_model() works.
  TF >= 2.16 (Keras 3) : it raises
        "File format not supported ... legacy SavedModel format is not supported"
     and you must go through tf.saved_model.load() instead.

We try both so the same code runs on the CARLA machine and on a dev box
without anyone having to check versions first. Same weights, same 95-dim
output either way.
"""
import os


def load_vae_encoder(model_dir):
    """Return f(batch_hwc_float32) -> (B, LATENT_DIM) tensor."""
    import tensorflow as tf

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"VAE encoder not found at '{model_dir}'. "
            f"Expected a SavedModel directory containing saved_model.pb."
        )

    # --- path A: Keras 2 / TF <= 2.15 -------------------------------------
    try:
        model = tf.keras.models.load_model(model_dir, compile=False)
        model.trainable = False
        for layer in getattr(model, 'layers', []):
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.training = False
        print(f"[vae_loader] loaded via tf.keras.models.load_model  ({model_dir})")
        return lambda x: model(x, training=False)
    except Exception as keras_err:
        keras_msg = f"{type(keras_err).__name__}: {str(keras_err)[:120]}"

    # --- path B: Keras 3 / TF >= 2.16 -------------------------------------
    try:
        sm = tf.saved_model.load(model_dir)
        fn = sm.signatures['serving_default']
        in_key  = list(fn.structured_input_signature[1].keys())[0]
        out_key = list(fn.structured_outputs.keys())[0]

        def _call(x):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            return fn(**{in_key: x})[out_key]

        print(f"[vae_loader] loaded via tf.saved_model.load  ({model_dir})")
        return _call
    except Exception as sm_err:
        raise RuntimeError(
            f"Could not load VAE encoder from '{model_dir}'.\n"
            f"  keras path : {keras_msg}\n"
            f"  savedmodel : {type(sm_err).__name__}: {str(sm_err)[:160]}"
        )
