# Orthographic Map Generation via 3D Reconstruction and Intelligent Point Cloud Filtering

> A research‑grade pipeline for **Orthographic Map Generation** from casual photo/video inputs, combining **3D Gaussian Splatting (3DGS)**, **Supersplat visualization**, and **two‑stage intelligent point cloud filtering**.

---

## Abstract

This project presents an end‑to‑end pipeline for generating high‑quality orthographic (bird’s‑eye) maps from multi‑view imagery. We leverage **3D Gaussian Splatting (3DGS)** for colored 3D reconstruction, apply a **two‑stage intelligent filtering process** to refine the resulting point cloud (main region preservation and cloud/ceiling removal), and produce the final **orthographic map** via **Spherical Harmonics (SH) decoding** and **soft splatting**. Additional visualization capabilities are supported through **Supersplat**, ensuring interactive exploration of intermediate results. We evaluate our approach on the **Tanks and Temples** and **ETH3D** benchmark datasets.

---

## Methods & Key Techniques

1. **3D Gaussian Splatting (3DGS)**

   * Input: Sparse camera poses and undistorted images from Structure‑from‑Motion.
   * Representation: Learnable anisotropic Gaussians (position, covariance, orientation) with **Spherical Harmonics (SH)** for view‑dependent color.
   * Training: Progressive densification with scheduled learning rates; online quality inspection via integrated Viewer.
   * Original repository: [https://github.com/graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

2. **Supersplat Visualization**

   * Interactive visualization and manual verification of dense point clouds and filtering effects.
   * Repository: [https://github.com/supersplat/supersplat](https://github.com/supersplat/supersplat)

3. **Two‑Stage Intelligent Point Cloud Filtering**
   **Pass‑1 (Main Region Preservation)**:

   * Establishes a ground‑aligned coordinate system (U, V, h) and extracts features (height, radial distance, local density, scale statistics, **Median Absolute Deviation (MAD)**, opacity).
   * Trains a lightweight MLP with weak statistical labels; keeps connected components consistent in height, removes distant/sparse/noisy points.

   **Pass‑2 (Cloud Detection + Ceiling Peeling)**:

   * Adds a second head for detecting high, low‑density clusters (clouds).
   * **Ceiling peeling** via grid‑based height quantile statistics removes floating layers above the main structure.

4. **Orthographic Projection (SH‑aware Soft Splatting)**

   * Sets up an orthographic camera aligned with the ground plane.
   * Decodes SH colors per Gaussian for a top‑down view and blends via depth/opacity‑weighted soft splatting.
   * Employs robust quantile cropping, supersampling, and light inpainting to produce sharp, artifact‑free maps.

---

## Training & Visualization

### Training (Example)

```bash
python train.py -s baseline --disable_viewer --resolution 1024 --sh_degree 2 --iterations 10000 --model_path ./output/baseline --checkpoint_iterations 4000 8000 --save_iterations 10000
```

### Point Cloud Visualization (3DGS Viewer)

```bash
Q:\\Master_Project\\Scene2Map\\gaussian-splatting\\viewers\\bin\\SIBR_gaussianViewer_app.exe -m Q:\\Master_Project\\Scene2Map\\gaussian-splatting\\output\\baseline
```

**Note**: The Viewer reads live training outputs for free camera inspection, density/color debugging, and early artifact detection.

### Supersplat

* Used for additional visualization, cleanup verification, and interactive editing of reconstructed point clouds.

---

## Two‑Stage Point Cloud Filtering

### Pass‑1: Main Region Preservation

```bash
python ml_point_filter.py --ply ./output/baseline/point_cloud/iteration_10000/point_cloud.ply --out_ply ./output/baseline/point_cloud/iteration_10000/point_cloud_ml.ply --epochs 12 --thr 0.55 --device cuda
```

**Highlights**:

* Ground alignment (U, V, h), geometric/statistical/appearance features;
* Lightweight MLP with weak labels;
* Thresholded probability + connected component filtering to suppress background.

### Pass‑2: Cloud Detection + Ceiling Peeling

```bash
python ml_point_filter_v4.py --ply ./output/baseline/point_cloud/iteration_10000/point_cloud_ml.ply --out_ply ./output/Lighthouse/point_cloud/iteration_10000/point_cloud_ml_clean.ply --epochs 12 --device cuda --keep_thr 0.55 --cloud_thr 0.60 --ceil_on 1 --ceil_cell 0.22 --ceil_p 0.35 --ceil_tolq 0.25
```

**Highlights**:

* Dual‑head classification (main/cloud);
* Grid‑quantile ceiling peeling to remove floating layers;
* Outputs `point_cloud_ml_clean.ply` for orthographic rendering.

---

## Orthographic Map Rendering

```bash
python gs_ortho_auto.py --ply ./output/baseline_fresh/point_cloud/iteration_10000/point_cloud_ml_clean.ply --pix 0.010 --ss 3 --beta 4.5 --q_low 0.10 --q_high 0.90 --pad 0.003 --shift_v -0.15 --frame_scale 1.15 --out_img ortho_down.png
```

**Core Parameters**:

* `--pix`: Pixel resolution (smaller = higher resolution).
* `--ss`: Supersampling factor.
* `--beta`: Depth/opacity blending bias to suppress thin floating layers.
* `--q_low/--q_high`: Robust quantile cropping for stable framing.
* `--frame_scale/--shift_v`: Global scaling and directional shift for composition.

---

## Evaluation & Visualization Criteria

* **Density Consistency**: Continuous density/scale distribution in the main region.
* **Color Fidelity**: SH‑decoded colors match input imagery, free of haze.
* **Edge Sharpness**: Roads/buildings appear crisp and artifact‑free.
* **Artifact Suppression**: Distant scenery, sky, and floating layers effectively removed.
* **Reproducibility**: Fixed random seeds and recorded parameter sets.

---

## Datasets

* **Tanks and Temples**: [https://www.tanksandtemples.org/](https://www.tanksandtemples.org/)
* **ETH3D**: [https://www.eth3d.net/datasets](https://www.eth3d.net/datasets)

---

## Troubleshooting

* **Dark/gray maps**: Increase `--beta`, adjust opacity thresholds, or verify filtering.
* **Tight/loose framing**: Adjust quantiles (`--q_low/--q_high`) and frame scale.
* **Holes/gaps**: Increase supersampling (`--ss`) or relax filtering thresholds.
* **Viewer issues**: Ensure viewer is enabled and port is available; training continues even if viewer closes.

---

## Citation

* Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). *3D Gaussian Splatting for Real-Time Radiance Field Rendering.* ACM Transactions on Graphics (SIGGRAPH). [https://github.com/graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
* **Supersplat**: [https://github.com/playcanvas/supersplat]/[https://superspl.at/editor/](https://github.com/supersplat/supersplat)(https://superspl.at/editor/)
* **Tanks and Temples Dataset**: [https://www.tanksandtemples.org/](https://www.tanksandtemples.org/)
* **ETH3D Dataset**: [https://www.eth3d.net/datasets](https://www.eth3d.net/datasets)

---

## License

Research and educational use only. 3DGS and visualization components follow their original licenses. Filtering and orthographic modules may be used in non‑commercial research; commercial use requires separate agreement.
