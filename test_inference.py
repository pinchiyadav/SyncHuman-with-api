import sys
import os
import time

os.chdir('/workspace/SyncHuman')
sys.path.insert(0, '/workspace/SyncHuman')

print("=" * 60)
print("Testing SyncHuman Stage 1 Inference")
print("=" * 60)

# Test basic imports
print("\n1. Testing imports...")
try:
    from SyncHuman.pipelines.SyncHumanOneStagePipeline import SyncHumanOneStagePipeline
    print("   ✓ SyncHumanOneStagePipeline imported")
except ImportError as e:
    print(f"   ✗ Error importing: {e}")
    sys.exit(1)

# Load pipeline
print("\n2. Loading pipeline (this may take a moment)...")
start_time = time.time()
try:
    pipeline = SyncHumanOneStagePipeline.from_pretrained(
        './ckpts/OneStage',
    )
    load_time = time.time() - start_time
    print(f"   ✓ Pipeline loaded in {load_time:.1f}s")
except Exception as e:
    print(f"   ✗ Error loading pipeline: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Enable memory efficient attention
print("\n3. Enabling memory efficient attention...")
try:
    pipeline.SyncHuman_2D3DCrossSpaceDiffusion.enable_xformers_memory_efficient_attention()
    print("   ✓ xformers attention enabled")
except Exception as e:
    print(f"   ⚠ Warning: Could not enable xformers: {e}")
    print("   Falling back to standard attention")

# Prepare input image
print("\n4. Preparing input image...")
input_path = './test_image1.png'
if not os.path.exists(input_path):
    print(f"   ✗ Input image not found: {input_path}")
    sys.exit(1)
print(f"   ✓ Using: {input_path}")

# Run inference
print("\n5. Running Stage 1 inference...")
print("   This will take 2-5 minutes depending on GPU...")
inference_start = time.time()
try:
    pipeline.run(
        image_path=input_path,
        save_path='./outputs/OneStage',
    )
    inference_time = time.time() - inference_start
    print(f"   ✓ Inference completed in {inference_time/60:.1f} minutes")
except Exception as e:
    print(f"   ✗ Error during inference: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check outputs
print("\n6. Verifying outputs...")
output_dir = './outputs/OneStage'
if os.path.exists(output_dir):
    files = os.listdir(output_dir)
    print(f"   ✓ Output directory exists with {len(files)} files:")
    for f in files[:10]:
        print(f"     - {f}")
else:
    print(f"   ✗ Output directory not found: {output_dir}")

print("\n" + "=" * 60)
print("✓ Stage 1 inference test completed successfully!")
print("=" * 60)
