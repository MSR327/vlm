# Quarantined — do not import

These files target an architecture that no longer exists. Brief section 9 rule 7
says delete them. They were **moved rather than deleted** because
`MTP_TESTING/.git` is corrupt (`packfile does not match index`) and the remote
`git@github.com:0skyq/MTP_TESTING.git` is unreachable from this machine, so a
deletion could not be undone.

**Delete this directory once the git object store is restored.**

| File | Why it is dead |
|---|---|
| `main_vlm.py` | References undefined `LATENT_DIM` and constructor kwargs `MultimodalEdgeEncoder` no longer accepts. Cannot import. |
| `test_vlm_carla.py` | Imports `PyTorchActorCritic` from `main_vlm.py`. |
| `PIL_edge_vlm.py` | Its 13-channel wire format does not match `PIL_simulation.py`'s sender. Protocol mismatch, not a cosmetic difference. |
| `convert.py` | TF-Lite export for the old TensorFlow VAE. The stack is PyTorch. |
| `tf_multimodal_encoder.py` | TensorFlow port of a superseded encoder. |
