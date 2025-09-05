# Vessel Centerline Analysis Results

This folder contains the analysis results of vessel centerlines.

## Folder Structure:
- `centerlines/`: Centerline data in NumPy format
- `centerlines_STL/`: Centerline STL models
- `figure1_with_centerline/`: 3D vessel with centerline
- `figure2_xy_projection/`: XY projection view
- `figure3_xz_projection/`: XZ projection view
- `figure4_yz_projection/`: YZ projection view
- `interactive_3D_html/`: Interactive 3D visualization
- `STL_cropped/`: Processed STL files

## Processing Details:
- Algorithm: 3D Medial Axis Extraction
- Smoothing: Cubic Spline Interpolation
- Optimization: Douglas-Peucker Simplification
