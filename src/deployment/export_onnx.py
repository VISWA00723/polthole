import os
import sys
import torch

# Add src to python path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_dir)

import config
from segmentation.model import get_segformer_model, MultiTaskRoadHazardModel

def main():
    print("--- Starting ONNX Export for Multi-Task Road Hazard model ---")
    
    # Setup device (CPU is preferred for ONNX export trace)
    device = torch.device("cpu")
    
    # 1. Initialize model
    print(f"Loading SegFormer backbone: {config.SEGFORMER_BACKBONE}...")
    base_model = get_segformer_model(pretrained=False)  # False is faster and sufficient for export structure
    model = MultiTaskRoadHazardModel(base_model, hidden_dim=512)
    model.to(device)
    model.eval()
    
    # Load weights if they exist
    if os.path.exists(config.SEGMENTATION_WEIGHTS):
        print(f"Loading custom weights from {config.SEGMENTATION_WEIGHTS}...")
        model.load_state_dict(torch.load(config.SEGMENTATION_WEIGHTS, map_location=device))
    else:
        print("Using random weights for structure export.")
        
    # 2. Create dummy inputs
    dummy_pixel_values = torch.zeros(1, 3, 512, 512, dtype=torch.float32, device=device)
    dummy_physical_inputs = torch.zeros(1, 4, dtype=torch.float32, device=device)
    
    # 3. Export path
    output_onnx = os.path.join(config.CHECKPOINT_DIR, "road_hazard_multitask.onnx")
    print(f"Export destination: {output_onnx}")
    
    # 4. Perform export
    try:
        torch.onnx.export(
            model=model,
            args=(dummy_pixel_values, dummy_physical_inputs),
            f=output_onnx,
            input_names=["pixel_values", "physical_inputs"],
            output_names=["segmentation_logits", "severity_logits", "road_score", "risk_score"],
            dynamic_axes={
                "pixel_values": {0: "batch_size", 2: "height", 3: "width"},
                "physical_inputs": {0: "batch_size"},
                "segmentation_logits": {0: "batch_size", 2: "height", 3: "width"},
                "severity_logits": {0: "batch_size"},
                "road_score": {0: "batch_size"},
                "risk_score": {0: "batch_size"}
            },
            opset_version=14,
            do_constant_folding=True
        )
        print("ONNX model successfully exported!")
    except Exception as e:
        print(f"Error exporting model to ONNX: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
