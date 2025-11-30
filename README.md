# EI_SmartTrafficManagementSystem
Edge Impulse Studio Smart Traffic Management System




# Routing Random Forest with Custom Learning Block

The block.json contains a Custom Learning Block for Edge Impulse that trains a Random Forest classifier (route) and regressor (total weight) from per-intersection snapshots.

The following files were uploaded into Edge Impulse Studio
- block.json — metadata and input/output spec for Edge Impulse
- training.py — training entry point (produces artifacts/)
- inference.py — inference entry point (predict(features, artifacts_path))

Instructions:
1. Zip the files (block.json, training.py, inference.py)
2. In Edge Impulse Studio -> Create Impulse -> Add Learning Block -> Custom -> Upload Block ZIP.
3. Provide training dataset (JSON list of records) to the project and start training the custom block.
4. The block will save artifacts/ which the inference function will load at runtime.
