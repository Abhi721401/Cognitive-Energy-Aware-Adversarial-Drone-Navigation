# Cognitive-Energy-Aware-Adversarial-Drone-Navigation

## 📌 Overview

CEAADN addresses a core challenge in autonomous drone navigation: how to fly from point A to point B while **simultaneously minimizing energy consumption, avoiding radar detection, handling wind resistance, and navigating terrain hazards** — all in real time.

Rather than relying on a single algorithm, CEAADN stacks three layers of intelligence:

| Layer | Method | Role |
|-------|--------|------|
| 🗺️ Global Planning | A\* on multi-objective cost map | Fast, reliable baseline path |
| 🧠 Risk Embedding | Graph Neural Network (GCN) | Learns terrain risk beyond raw cost |
| 🤖 Local Adaptation | PPO Reinforcement Learning | Re-routes around high-radar segments |

## 🏗️ Architecture

```
Synthetic Terrain (DEM)
        │
        ▼
Environment Simulation
  ├── Elevation Map
  ├── Slope & River Mask
  ├── Wind Field
  └── Radar Detection Map (Adversarial Layer)
        │
        ▼
Multi-Objective Cost Function
  C(x) = α·E(x) + β·D(x) + γ·W(x) + δ·R(x)
        │
        ▼
   ┌────┴────┐
   │         │
A* Planner  GNN (GCNConv × 3)
(baseline)  (learns risk embeddings)
   │         │
   └────┬────┘
        │
        ▼
PPO RL Agent
(local segment re-router)
        │
        ▼
Final Stitched Path + Plotly Visualizations
```

## ✨ Key Features

- **Synthetic terrain generation** with Gaussian-smoothed mountain peaks, river masks, and wind fields
- **Adversarial radar modeling** — multiple radar stations with line-of-sight and distance decay
- **Multi-objective cost function** with tunable weights (energy, detection, wind, terrain risk)
- **A\* baseline planner** with 8-directional movement on a grid graph
- **3-layer GCN** trained to predict terrain risk embeddings from node features
- **PPO RL agent** trained with imitation reward shaping from A\* path guidance
- **Two-stage curriculum training** — guided (80k steps) → independent fine-tuning (60k steps)
- **Greedy fallback + A\* stitch** safety net for guaranteed path completion
- **6 interactive Plotly visualizations** including 3D cinematic flight, 2D heatmaps, altitude profiles

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.0+ | GNN training |
| `torch-geometric` | latest | GCNConv layers |
| `stable-baselines3` | latest | PPO agent |
| `gymnasium` | latest | RL environment |
| `numpy` / `scipy` | — | Terrain simulation |
| `plotly` | latest | All visualizations |


## ⚙️ Configuration

All key parameters are in **Cell 2** of the notebook:

```python
GRID  = 64          # Grid resolution (use 32 for fast testing)
START = (2, 2)      # Drone start position
GOAL  = (61, 32)    # Drone goal position (offset — not diagonal)

# Cost function weights
ALPHA = 0.4   # Energy (slope)
BETA  = 0.3   # Radar detection
GAMMA = 0.2   # Wind penalty
DELTA = 0.1   # Terrain/river risk
```
## 🗺️ Visualizations

The notebook generates 6 interactive Plotly figures:

| # | Figure | Description |
|---|--------|-------------|
| 1 | 🚁 3D Flight | Cinematic 3D mountain terrain with all 3 flight paths, radar towers, shadows, direction cones |
| 2 | 📊 4-Panel Heatmaps | Elevation, radar detection, GNN risk, cost map side by side |
| 3 | 🗺️ 2D Top-Down | All paths overlaid on cost map with radar markers |
| 4 | 📈 Altitude Profile | How each drone climbs and descends across the mountain |
| 5 | 📊 Bar Chart | Energy / exposure / steps comparison across all methods |
| 6 | 🧠 GNN Loss Curve | Training convergence of the terrain risk predictor |

## 🧠 Method Details

### Multi-Objective Cost Function
```
C(x) = α·E(x) + β·D(x) + γ·W(x) + δ·R(x)
```
- **E(x)** — Energy cost (terrain slope)
- **D(x)** — Radar detection probability (line-of-sight + distance decay)
- **W(x)** — Wind opposition penalty (dot product of wind vs. movement direction)
- **R(x)** — Terrain risk (river zones, low elevation)

### GNN Architecture
- 3-layer GCNConv with BatchNorm and ReLU
- Input: 6 node features per cell `[elevation, slope, wind_x, wind_y, detection, river]`
- Output: learned risk score in `[0, 1]`
- Trained with MSE loss to predict the cost map

### PPO RL Agent
- Operates on **local 15×15 patches** around high-radar A\* segments
- Reward: `8·progress − 0.5·detection − 0.2·cost + guide_bonus`
- Two-stage curriculum: A\*-guided (80k steps) → independent (60k steps)
- Greedy fallback + A\* stitch as safety net

## 🔭 Future Work

- [ ] Real DEM data (SRTM / NASA elevation datasets)
- [ ] Dynamic radar (moving adversaries)
- [ ] 3D altitude control (not just 2D grid)
- [ ] Multi-drone coordination
- [ ] Transformer-based path planner replacing GNN
- [ ] Real-time replanning on partial map updates

## 🙏 Acknowledgements

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) — GNN framework
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) — PPO implementation
- [Plotly](https://plotly.com/python/) — Interactive visualizations

---

<p align="center">Made with ❤️ | If you found this useful, give it a ⭐</p>
