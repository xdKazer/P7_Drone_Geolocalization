<p align="center">
  <h1 align="center"><ins>FVL-SAR & TVL-SAR</ins> <br>Towards GNSS-Denied Geo-Positioning using Search Area Refinement</h1>
  <p align="center">
    <a href="https://www.linkedin.com/in/signe-moller-skuldbol/">Signe Møller-Skuldbøl</a>
    -
    <a href="https://www.linkedin.com/in/kasperkjaersgaardlauritsen/">Kasper Lauritsen</a>
    -
    <a href="https://www.linkedin.com/in/maria-daliana-moga-b35b51262/">Maria Moga</a>
    -
    <a href="https://www.linkedin.com/in/omkarkorg/">Omkar Korgaonkar</a>
    -
    <a href="https://www.linkedin.com/in/peterguldleth/">Peter Leth</a>
    -
    <a href="https://www.linkedin.com/in/andreasmoegelmose/">Andreas Møgelmose</a>
  </p>
</p>

# Geopositioning with FVL-SAR and TVL-SAR
This repository contains the implementation of FVL-SAR (Filter aided Visual Localisation with Search Area Refinement) & TVL-SAR (Transformer-based Visual Localisation with Search Area Refinement)

Both implementations rely on a functional LightGlue installation: [LightGlue](https://github.com/cvg/LightGlue)

FVL-SAR and TVL-SAR are entirely independent from one another. Therefore, the usage of both algorithms differs.
An example of how to run FVL-SAR and TVL-SAR will be provided after the setup guide.

# Setup Guide
The following guide entails the basics of downloading this repository and the corresponding dependencies.

First and foremost, start by cloning this repository and LightGlue:

```bash
git clone https://github.com/xdKazer/P7_Drone_Geolocalization.git
cd P7_Drone_Geolocalization
rmdir LightGlue
git clone https://github.com/cvg/LightGlue.git
cd LightGlue
python -m pip install -e .
```

With this, you should now have FVL-SAR, TVL-SAR and LightGlue installed in a directory.

If you wish to work with the same datasets as the ones used for the development of both methods, please download the datasets from here:
- [UAV-VisLoc](https://github.com/IntelliSensing/UAV-VisLoc)
- [VPAIR](https://github.com/AerVisLoc/vpair)

Once these are downloaded, please ensure that the following files are placed in the following folders: (Follow the guide for the model you wish to use, or both for comparison)

FVL-SAR:


TVL-SAR:

- UAV-VisLoc:

Each individual satellite.tif file from UAV-VisLoc -> geolocalization_dinov3\dataset_data\satellite_images

All UAV images from UAV-VisLoc in individual folders named 01, 02, ..., 11 -> geolocalization_dinov3\dataset_data\drone_images (Rename folder "drone" to corresponding dataset)

All satellite images from vpair\reference_views -> geolocalization_dinov3\VPAIR_TVL\tiles

- VPAIR:

All UAV images from vpair\queries -> geolocalization_dinov3\VPAIR_TVL\drone

The file "poses.csv" must be renamed to "poses_lat_long.csv" and moved to -> geolocalization_dinov3\VPAIR_TVL

The file "camera_calibration.yaml" must be moved to -> geolocalization_dinov3\VPAIR_TVL

# Using FVL-SAR
...

# Using TVL-SAR
...

# BibTeX citation
If you use either TVL-SAR or FVL-SAR, please consider citing our GitHub:
```txt
@software{FVLSAR_TVLSAR,
  author = {M{\o}ller-Skuldb{\o}l, Signe and Lauritsen, Kasper and Moga, Maria and Korgaonkar, Omkar and Leth, Peter and M{\o}gelmose, Andreas},
  month = {3},
  title = {Towards GNSS-Denied Geo-Positioning using Search Area Refinement},
  url = {https://github.com/xdKazer/P7_Drone_Geolocalization},
  version = {1.0.0},
  year = {2026}
}
```
