import os
import sys
import subprocess

# Add src to python path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_dir)

import config

def optimize_using_trtexec(onnx_path, engine_path):
    """
    Attempts to optimize the ONNX model using the standard NVIDIA TensorRT CLI tool 'trtexec'.
    This is the most reliable way to serialize engines across Jetson/Orin versions.
    """
    print(f"--- Attempting TensorRT compilation using 'trtexec' CLI ---")
    
    # trtexec command syntax:
    # trtexec --onnx=model.onnx --saveEngine=model.engine --fp16 --workspace=2048
    command = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",              # Enable FP16 precision
        "--workspace=2048",    # Allocate workspace in MB
        "--verbose"            # Verbose logging
    ]
    
    print(f"Running command: {' '.join(command)}")
    try:
        # Run process
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("TensorRT CLI compilation finished successfully!")
        print(result.stdout[-1000:])  # Print last 1000 chars of compilation logs
        return True
    except FileNotFoundError:
        print("WARNING: 'trtexec' CLI tool was not found on PATH. TensorRT optimizations require NVIDIA JetPack/CUDA SDK.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"ERROR: 'trtexec' compilation failed: {e.stderr}")
        return False

def optimize_using_python_api(onnx_path, engine_path):
    """
    Attempts to serialize TensorRT engine using the python API.
    """
    print(f"--- Attempting TensorRT compilation using Python API ---")
    try:
        import tensorrt as trt
    except ImportError:
        print("WARNING: 'tensorrt' python package is not installed. Run 'pip install tensorrt'.")
        return False
        
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # 1. Create builder, network, and parser
    builder = trt.Builder(TRT_LOGGER)
    config_trt = builder.create_builder_config()
    
    # Set memory pool limit (equivalent to workspace)
    # 2GB in bytes
    config_trt.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)
    
    # Enable FP16 if supported
    if builder.platform_has_fast_fp16:
        print("Enabling fast FP16 mode...")
        config_trt.set_flag(trt.BuilderFlag.FP16)
        
    # Create network definition
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX file
    print(f"Parsing ONNX model: {onnx_path}...")
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX file {onnx_path} does not exist. Run export_onnx.py first.")
        return False
        
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
            
    # Build engine
    print("Building TensorRT serialized engine (this can take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config_trt)
    
    if serialized_engine is None:
        print("ERROR: Engine building failed.")
        return False
        
    # Write engine to file
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
        
    print(f"Successfully saved TensorRT engine to: {engine_path}")
    return True

def main():
    onnx_path = os.path.join(config.CHECKPOINT_DIR, "road_hazard_multitask.onnx")
    engine_path = os.path.join(config.CHECKPOINT_DIR, "road_hazard_multitask.engine")
    
    if not os.path.exists(onnx_path):
        print("ONNX model not found. Automatically invoking ONNX export script first...")
        import export_onnx
        export_onnx.main()
        
    # Attempt python API first
    success = optimize_using_python_api(onnx_path, engine_path)
    
    # Fallback to trtexec CLI if python API is unavailable
    if not success:
        success = optimize_using_trtexec(onnx_path, engine_path)
        
    if success:
        print("\n=== TensorRT Optimization finished successfully! ===")
    else:
        print("\n=== TensorRT Optimization was skipped or failed. ===")
        print("For Jetson Orin edge deployments, ensure you run this script inside the JetPack L4T container:")
        print("1. Build/run container: nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3")
        print(f"2. Run: trtexec --onnx={onnx_path} --saveEngine={engine_path} --fp16")

if __name__ == '__main__':
    main()
