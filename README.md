# pyradplan-pb-baseline

Pencil-beam baseline dose computation for the DoseRad2026 challenge.
Reproduces per-control-point (photons) and per-beamlet (protons) dose distributions using [pyRadPlan](https://github.com/e0404/pyRadPlan) and compares them against the Monte-Carlo ground truth supplied with the dataset.

---

## Requirements

- Python 3.10+
- pyRadPlan 0.3.5 (see [Setup](#setup) below)
- NumPy, SimpleITK, pydantic, Matplotlib (pulled in by pyRadPlan)

---

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/DoseRAD2026/pyradplan-pb-baseline
cd pyradplan-pb-baseline
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Point the scripts at your dataset**

Both scripts hardcode the dataset root and output directory in their `__main__` block. Edit before running:

```python
# baseline_photons.py  /  baseline_protons.py
baseline_dir = "home/user/doserad2026"   # root of the challenge dataset
write_dir    = "home/user/doserad2026_out"   # where PB dose outputs are written
```

---

## Usage

Select the patient, modality, split, and beam/control-point or beam/ray/beamlet indices directly in the `__main__` block of each script, then run:

```bash
# Photon baseline — one control point
python baseline_photons.py

# Proton baseline — one beamlet
python baseline_protons.py
```

Key parameters in `__main__`:

| Parameter | Description |
|-----------|-------------|
| `patient_id` | Patient identifier (e.g. `"1ABB169"`, `"1THB008"`) |
| `modality` | `"photon"` or `"proton"` |
| `split` | `"train"` or `"test"` |
| `beam_idx` | 0-based beam index into the plan JSON |
| `cp_idx` *(photons)* | 0-based control-point index within the beam |
| `ray_idx` *(protons)* | 0-based ray index within the beam |
| `beamlet_idx` *(protons)* | 0-based beamlet index within the ray |
| `show_plots` | Display a three-panel comparison figure |
| `plot_view_slice` | Viewing plane: `"axial"`, `"sagittal"`, or `"coronal"` |

---

## Outputs

When `output_dir` is set, the computed PB dose is written to:

| Modality | Filename |
|----------|---------|
| Photon | `<id>_photon_<split>_PB_Dose_B{b}_CP{cp:03d}.mha` |
| Proton | `<id>_proton_<split>_PB_Dose_B{b}_R{r}_L{l}.mha` |

When `show_plots=True`, a three-panel figure is displayed:

| Panel | Content |
|-------|---------|
| Left | Pencil-beam dose overlaid on CT |
| Centre | MC reference dose overlaid on CT |
| Right | Difference (PB − MC), masked to the dose region |

---

## Algorithms

### Photons — SVDPB

`baseline_photons.py` operates on individual VMAT **control points**.
For each control point the MLC leaf positions and jaw settings are reconstructed from the plan JSON, a field-based STF is built via pyRadPlan's `photonSingleBixel` BLD generator, beamlet weights are normalised to the MC fluence scale, and the SVDPB forward engine computes the dose.

| Setting | Value |
|---------|-------|
| Machine | Generic photon, 6 MeV |
| MLC leaf width | 5 mm |
| Dose-grid resolution | CT native grid |
| Penumbra (forced) | 1.8 mm |

### Protons — Hong pencil-beam

`baseline_protons.py` operates on individual **beamlets**.
For each beamlet a minimal single-ray STF is constructed from the plan JSON geometry, the requested energy is snapped to the nearest energy available in the Generic proton machine model, and pyRadPlan's Hong pencil-beam forward engine computes the dose.

| Setting | Value |
|---------|-------|
| Machine | Generic proton |
| SAD | from machine model (plan JSON SAD used only as a reference) |
| Lateral cutoff | 25 mm |
| Dose-grid resolution | CT native grid |
| Air-offset correction | enabled |


### Authors
Niklas Wahl (DKFZ Heidelberg)
Samir Schulz (DKFZ Heidelberg)
Lina Bucher (DKFZ Heidelberg)