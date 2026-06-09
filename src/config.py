import os

# Central Configuration for Road Hazard Intelligence System

# --- Kaggle Dataset Authentication & Targets ---
KAGGLE_USERNAME = "kaggleuser"  # Fallback dummy username (we will authenticate using token string)
KAGGLE_KEY = "KGAT_b04865aa0947b72aef5e424add470836"  # Provided Kaggle API token

# Default directory for datasets on the D:/ drive
DATASET_DIR = os.path.normpath("d:/road_hazard_datasets")
os.makedirs(DATASET_DIR, exist_ok=True)

# --- Customizable Camera Calibration parameters ---
# (Used in physical depth and distance approximations. Can be customized per bike mounting.)
CAMERA_HEIGHT_M = 1.0          # Height of the camera mount from the ground (meters)
CAMERA_TILT_DEG = -15.0        # Pitch angle looking down (negative is down, in degrees)
CAMERA_FOCAL_LENGTH_PX = 800.0 # Approximate focal length in pixels (fx and fy)
IMAGE_WIDTH = 1280             # Input image width
IMAGE_HEIGHT = 720             # Input image height
PRINCIPAL_POINT_X = 640.0      # Camera optical center X coordinate (px)
PRINCIPAL_POINT_Y = 360.0      # Camera optical center Y coordinate (px)

# --- Model Architectures and Parameters ---
# SegFormer-B5 Parameters
SEGFORMER_BACKBONE = "nvidia/mit-b5"
NUM_CLASSES = 5
CLASS_MAP = {
    0: "background",
    1: "pothole",
    2: "crack",
    3: "water_pothole",
    4: "manhole"
}
CLASS_COLORS = {
    0: (0, 0, 0),       # Background - Black
    1: (0, 0, 255),     # Pothole - Red
    2: (0, 255, 255),   # Crack - Yellow
    3: (255, 0, 255),   # Water Pothole - Magenta
    4: (0, 255, 0)      # Manhole - Green
}

# Depth Anything V2 Checkpoint / Hugging Face model
# We can use the HuggingFace transformers compatible repository for easy inference
DEPTH_MODEL_HF = "depth-anything/Depth-Anything-V2-Small-hf"

# Local checkpoints folders
CHECKPOINT_DIR = os.path.normpath("d:/road_hazard_checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
SEGMENTATION_WEIGHTS = os.path.join(CHECKPOINT_DIR, "segformer_b3_road.pth")
ROAD_QUALITY_WEIGHTS = os.path.join(CHECKPOINT_DIR, "road_quality_regressor.pth")

# --- Risk Assessment Engine Coefficients ---
DEFAULT_BIKE_SPEED_KMH = 15.0  # Default bicycle speed in km/h if no speedometer input is available
RISK_THRESHOLD = 75.0          # Threshold above which alerts are triggered (out of 100)
WEIGHT_AREA = 0.4              # Weight for hazard area in severity calculation
WEIGHT_DEPTH = 0.6             # Weight for hazard depth in severity calculation
TTC_TAU = 2.0                  # Time-to-collision normalization time-constant (seconds)

# Max normalization limits (used to clip physical estimates to [0, 1] range for models)
MAX_HAZARD_AREA_M2 = 2.0       # Max area of a single pothole/crack to consider (m^2)
MAX_HAZARD_DEPTH_CM = 25.0     # Max depth of a pothole to consider (cm)
