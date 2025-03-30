import yaml
import kagglehub

with open('./data.yml') as f:
    path = yaml.safe_load(f)['path']

# Download latest version
if path is None:
    path = kagglehub.dataset_download("kaito117/opportunity-spline-interpolated-sensor-data-only")

print("Path to dataset files:", path)
