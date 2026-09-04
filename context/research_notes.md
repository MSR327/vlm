# 📚 BTP Resource Guide: LLM + CARLA + RL + Chatbot Fusion

Everything you need to read/watch/clone before we start building. Organized by topic, prioritized within each section.

---

## Suggested Reading Order

> [!TIP]
> Don't try to read everything at once. Follow this path:
> 1. **Week 1**: Skim the survey (§1), then deep-read SimLingo + CaRL (§2) — these are your two closest reference architectures
> 2. **Week 1–2**: Get CARLA running (§8), clone the VAE+PPO baseline repo (§7), reproduce a basic driving agent
> 3. **Week 2**: Read the PPO paper + SpinningUp (§3), understand DRQN for partial observability
> 4. **Week 2–3**: Study InternVL2 architecture (§4) and the fusion techniques (§6) — this is where your novelty lives

---

## §1. Survey Papers (Start Here for Big Picture)

| Paper | Link | Why Read It |
|---|---|---|
| **LLM4AD: LLMs for Autonomous Driving** | [arXiv:2410.15281](https://arxiv.org/abs/2410.15281) | The most comprehensive survey connecting LLMs to AD. Covers the full taxonomy — perception, planning, control, evaluation. Read §2-4 to understand where your project sits in the landscape. |
| **Vision Language Models in Autonomous Driving: A Survey** (IEEE TIV 2024) | [arXiv:2310.14414](https://arxiv.org/abs/2310.14414) | Specifically covers VLMs (not just text LLMs) in driving — perception, navigation, planning. Directly relevant since you're replacing VAE with a VLM. |
| **A Survey of Reasoning in AD Systems** | [arXiv:2603.11093](https://arxiv.org/abs/2603.11093) | Introduces a "Cognitive Hierarchy" for driving tasks. Useful for framing your uMDP formulation — how does reasoning under uncertainty fit into the hierarchy? |

---

## §2. Core Architecture Papers (Your Closest References)

### 🏆 SimLingo — **READ THIS FIRST**
| | |
|---|---|
| **Paper** | [arXiv:2406.10165](https://arxiv.org/abs/2406.10165) (CVPR 2025) |
| **GitHub** | [github.com/OpenDriveLab/SimLingo](https://github.com/OpenDriveLab/SimLingo) |
| **Why** | Winner of the CARLA Challenge 2024. Uses **InternVL2-1B** (the same model your mentor suggested), end-to-end vision-language-action alignment, instruction following. This is the closest existing system to what you're building. Study its architecture diagram, the LoRA fine-tuning setup, and the "Action Dreaming" data generation method. |
| **Key takeaways** | Vision-only (no LiDAR), disentangled waypoint outputs, language-action alignment without unsafe execution |

### DriveMLM
| | |
|---|---|
| **Paper** | [arXiv:2312.09245](https://arxiv.org/abs/2312.09245) |
| **GitHub** | [github.com/OpenGVLab/DriveMLM](https://github.com/OpenGVLab/DriveMLM) |
| **Why** | Shows how to plug an MLLM into an existing driving stack (Apollo/Autopilot). Unlike SimLingo (end-to-end), DriveMLM is **modular** — the LLM makes high-level decisions, and a separate planner executes them. Useful if you want to compare architectures. |
| **Key takeaways** | Standardized "decision states" that bridge language→control, explainable decisions, data engine for annotation |

### CaRL — Scaling PPO in CARLA
| | |
|---|---|
| **Paper** | [arXiv:2306.09345](https://arxiv.org/abs/2306.09345) (CoRL 2025) |
| **Why** | Demonstrates that **simple reward functions** (just route completion) work better than complex multi-term rewards when scaling PPO. This directly informs how you should design your reward function. Shows PPO scaling to 16K batch sizes. |
| **Key takeaways** | Simplify reward → better scalability. Route completion + infraction termination is all you need. |

---

## §3. Reinforcement Learning Foundations

### PPO (your primary algorithm)

| Resource | Link | Type | Why |
|---|---|---|---|
| **Original Paper**: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017) | [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) | Paper | The foundational paper. Understand the clipped surrogate objective — it's the core of everything. |
| **OpenAI Spinning Up — PPO** | [spinningup.openai.com/…/ppo](https://spinningup.openai.com/en/latest/algorithms/ppo.html) | Tutorial | Best intuitive explanation of PPO. Read this *before* the paper. Covers GAE, clipping, and why PPO beats TRPO. |
| **Stable-Baselines3 — PPO Docs** | [stable-baselines3.readthedocs.io/…/ppo](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) | API Docs | The library you'll actually use. Read the hyperparameter descriptions and examples. |

### DDPG (your ablation/comparison)

| Resource | Link | Type |
|---|---|---|
| **Original Paper**: "Continuous control with deep RL" (Lillicrap et al., 2015) | [arXiv:1509.02971](https://arxiv.org/abs/1509.02971) | Paper |
| **Spinning Up — DDPG** | [spinningup.openai.com/…/ddpg](https://spinningup.openai.com/en/latest/algorithms/ddpg.html) | Tutorial |

### Partial Observability / uMDP (critical for your formulation)

| Resource | Link | Why |
|---|---|---|
| **DRQN**: "Deep Recurrent Q-Learning for Partially Observable MDPs" (Hausknecht & Stone) | [arXiv:1507.06527](https://arxiv.org/abs/1507.06527) | Seminal paper on adding LSTM to DQN for POMDPs. Your uMDP approach with LSTM belief states is directly descended from this. |
| **Recurrent Policy Gradients for POMDPs** (Wierstra et al.) | [link.springer.com](https://link.springer.com/chapter/10.1007/978-3-540-74958-5_78) | Earlier work combining policy gradients + RNNs for memory-based policies. |
| **SB3-Contrib RecurrentPPO** | [sb3-contrib.readthedocs.io/…/ppo_recurrent](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html) | The actual implementation you'll use — PPO with LSTM layers for handling observation histories. |
| Arthur Juliani's **"Simple RL with TF — Part 6: POMDPs"** | [medium.com/@awjuliani](https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-6-partial-observability-and-deep-recurrent-q-68463e9aeefc) | Beginner-friendly walkthrough of partial observability + DRQN |

---

## §4. Multimodal LLM / Vision-Language Model Backbone

### InternVL2 (recommended backbone)

| Resource | Link | Why |
|---|---|---|
| **InternVL2 Technical Report** | [arXiv:2404.16821](https://arxiv.org/abs/2404.16821) | Understand the ViT-MLP-LLM architecture. Your perception encoder will use the frozen ViT from this model. |
| **GitHub Repo** | [github.com/OpenGVLab/InternVL](https://github.com/OpenGVLab/InternVL) | Code, tutorials, pre-trained checkpoints, LoRA fine-tuning guides |
| **HuggingFace Collection** | [huggingface.co/OpenGVLab](https://huggingface.co/collections/OpenGVLab/internvl-20-667d3961ab5eb12c7ed1361b) | Download InternVL2-1B, InternVL2-2B, etc. Start with the 1B model. |

### Alternatives (if InternVL2 doesn't work for your setup)

| Model | Link | Notes |
|---|---|---|
| **LLaVA-1.5** | [github.com/haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA) | 7B/13B. Better zero-shot but heavier. Good for prototyping with quantization. |
| **Phi-3-Vision** | [huggingface.co/microsoft/Phi-3-vision-128k-instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) | 4.2B. Microsoft's efficient VLM. Good middle ground. |

---

## §5. Chatbot & Natural Language Grounding

| Resource | Link | Why |
|---|---|---|
| **Talk2Car Dataset** | [talk2car.github.io](https://talk2car.github.io/) | The canonical dataset for natural language commands → autonomous driving actions. Built on nuScenes. Study the command categories and grounding task. |
| **Talk2Car-Trajectory** | [github.com/talk2car](https://github.com/talk2car/Talk2Car-Trajectory) | Extension that maps NL commands → physical trajectories. Closer to what your chatbot needs to do. |
| **Sentence-Transformers (SBERT)** | [sbert.net](https://www.sbert.net/) | Library for encoding text into dense vectors. `all-MiniLM-L6-v2` (384-dim, fast) is the go-to for your text encoder. |
| **HuggingFace: all-MiniLM-L6-v2** | [huggingface.co/sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | The specific model to start with for encoding passenger commands. |

---

## §6. Multimodal Fusion Techniques

| Resource | Link | Why |
|---|---|---|
| **Attention Is All You Need** (Vaswani et al., 2017) | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) | Foundation for the cross-attention mechanism you'll use in fusion. If you haven't read this, read it. |
| **Gated Multimodal Units (GMU)** (Arevalo et al., 2017) | [arXiv:1702.01992](https://arxiv.org/abs/1702.01992) | Introduces the gated fusion concept — learn to weight modalities dynamically. This is the simpler fusion approach for your baseline. |
| **VisualBERT / ViLBERT** | [arXiv:1908.03557](https://arxiv.org/abs/1908.03557) / [arXiv:1908.02265](https://arxiv.org/abs/1908.02265) | Early vision-language fusion architectures. Useful for understanding how cross-attention between image regions and text tokens works. |
| **CLIP** (Radford et al., 2021) | [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) | Contrastive pre-training to align visual and text embeddings. You might use CLIP-style alignment to pre-align your visual + text features before fusion. |

---

## §7. GitHub Repos with Working Code (Clone These)

### VAE + PPO Baselines (to beat)

| Repo | Link | Notes |
|---|---|---|
| **CARLA-SB3-RL-Training-Environment** | [github.com/alberto-mate/CARLA-SB3-RL-Training-Environment](https://github.com/alberto-mate/CARLA-SB3-RL-Training-Environment) | ⭐ Best starting point. Has VAE pre-trained models, SB3 integration, modular reward functions. Start here. |
| **Carla-ppo** (bitsauce) | [github.com/bitsauce/Carla-ppo](https://github.com/bitsauce/Carla-ppo) | Clean VAE training code (`train_vae.py`), latent space inspection tools. |
| **E2E-CARLA-RL-PPO** | [github.com/4lberto/E2E-CARLA-ReinforcementLearning-PPO](https://github.com/4lberto/E2E-CARLA-ReinforcementLearning-PPO) | Minimal end-to-end PPO in CARLA. Good for understanding the simplest pipeline. |

### CARLA Gym Wrappers

| Repo | Link | Notes |
|---|---|---|
| **CARLA-GymDrive** | [github.com/angelomorgado/CARLA-GymDrive](https://github.com/angelomorgado/CARLA-GymDrive) | Gymnasium-compatible CARLA wrapper with PPO and DQN agents. Well-documented. |

### LLM + Driving

| Repo | Link | Notes |
|---|---|---|
| **SimLingo** | [github.com/OpenDriveLab/SimLingo](https://github.com/OpenDriveLab/SimLingo) | Full code + checkpoints for the CARLA Challenge 2024 winner. |
| **DriveMLM** | [github.com/OpenGVLab/DriveMLM](https://github.com/OpenGVLab/DriveMLM) | Modular LLM-driving integration code. |

---

## §8. CARLA Simulator — Setup & Documentation

| Resource | Link | Why |
|---|---|---|
| **CARLA Releases (download 0.9.15)** | [github.com/carla-simulator/carla/releases](https://github.com/carla-simulator/carla/releases) | Download the pre-built package. ~20GB. |
| **Official Documentation** | [carla.readthedocs.io](https://carla.readthedocs.io/en/0.9.15/) | API reference, sensor guide, map descriptions, synchronous mode setup. |
| **Python API Quick Start** | [carla.readthedocs.io/…/quickstart](https://carla.readthedocs.io/en/latest/start_quickstart/) | Get hello-world running in 10 minutes. |
| **Sensor Reference** | [carla.readthedocs.io/…/ref_sensors](https://carla.readthedocs.io/en/latest/ref_sensors/) | Critical — understand RGB camera, GPS, IMU, speedometer specs. You need to know your observation space. |
| **Bench2Drive Benchmark** | [github.com/Thinklab-SJTU/Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) | 220 short routes, 44 scenarios. Use this for evaluation if targeting Leaderboard 2.0. |
| **Stable-Baselines3 Docs** | [stable-baselines3.readthedocs.io](https://stable-baselines3.readthedocs.io/) | Your RL library. Read Custom Environments and Callbacks sections. |
| **SB3-Contrib (RecurrentPPO)** | [sb3-contrib.readthedocs.io](https://sb3-contrib.readthedocs.io/) | LSTM-based PPO for your uMDP formulation. |

---

## Quick Install Cheatsheet

```bash
# Python environment
conda create -n carla-btp python=3.10 -y
conda activate carla-btp

# CARLA client
pip install carla==0.9.15

# RL
pip install stable-baselines3 sb3-contrib

# Vision-Language Model
pip install transformers accelerate

# Text Encoder
pip install sentence-transformers

# Utilities
pip install gymnasium pygame numpy opencv-python tensorboard wandb
```

---

## 🎯 Priority Matrix

| Resource | Urgency | Effort | Impact on Your BTP |
|---|---|---|---|
| SimLingo paper + code | 🔴 Now | ~4 hrs | ⭐⭐⭐ Closest reference arch |
| CARLA setup + hello-world | 🔴 Now | ~2 hrs | ⭐⭐⭐ Can't do anything without it |
| VAE+PPO baseline repo | 🔴 Now | ~3 hrs | ⭐⭐⭐ Your performance floor |
| PPO SpinningUp tutorial | 🔴 Now | ~2 hrs | ⭐⭐⭐ Must understand your algorithm |
| LLM4AD survey | 🟡 This week | ~3 hrs | ⭐⭐ Landscape + related work section |
| CaRL paper | 🟡 This week | ~2 hrs | ⭐⭐ Reward design insights |
| InternVL2 report + HF | 🟡 This week | ~2 hrs | ⭐⭐ Your perception backbone |
| DRQN paper | 🟡 Week 2 | ~2 hrs | ⭐⭐ uMDP belief state foundation |
| Talk2Car dataset | 🟢 Week 2–3 | ~1 hr | ⭐ Understand NL grounding in driving |
| Fusion papers (GMU/CLIP) | 🟢 Week 3 | ~3 hrs | ⭐⭐ Your novel contribution area |
| DriveMLM paper | 🟢 Week 3 | ~2 hrs | ⭐ Alternative architecture comparison |
