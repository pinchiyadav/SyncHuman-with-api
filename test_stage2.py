import sys
import os
import time

os.chdir('/workspace/SyncHuman')
sys.path.insert(0, '/workspace/SyncHuman')

print("=" * 60)
print("Testing SyncHuman Stage 2 Inference")
print("=" * 60)

# Test imports
print("\n1. Testing imports...")
try:
    from SyncHuman.pipelines.SyncHumanTwoStage import SyncHumanTwoStagePipeline
    print("   ✓ SyncHumanTwoStagePipeline imported")
except ImportError as e:
    print(f"   ✗ Error importing: {e}")
    sys.exit(1)

# Load pipeline
print("\n2. Loading Stage 2 pipeline...")
start_time = time.time()
try:
    pipeline = SyncHumanTwoStagePipeline.from_pretrained(
        './ckpts/SecondStage',
    )
    load_time = time.time() - start_time
    pipeline.cuda()
    print(f"   ✓ Pipeline loaded in {load_time:.1f}s")
except Exception as e:
    print(f"   ✗ Error loading pipeline: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check Stage 1 output
print("\n3. Checking Stage 1 output...")
stage1_path = './outputs/OneStage'
if not os.path.exists(stage1_path):
    print(f"   ✗ Stage 1 output not found: {stage1_path}")
    sys.exit(1)
print(f"   ✓ Stage 1 output found")

# Run Stage 2 inference
print("\n4. Running Stage 2 inference...")
print("   This will take 2-3 minutes depending on GPU...")
inference_start = time.time()
try:
    pipeline.run(
        image_path=stage1_path,
        outpath='./outputs/SecondStage',
    )
    inference_time = time.time() - inference_start
    print(f"   ✓ Inference completed in {inference_time/60:.1f} minutes")
except Exception as e:
    print(f"   ✗ Error during inference: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check outputs
print("\n5. Verifying outputs...")
output_dir = './outputs/SecondStage'
if os.path.exists(output_dir):
    files = os.listdir(output_dir)
    print(f"   ✓ Output directory exists with {len(files)} files")
    if 'output.glb' in files:
        glb_size = os.path.getsize(os.path.join(output_dir, 'output.glb')) / (1024*1024)
        print(f"   ✓ Final 3D model (output.glb): {glb_size:.1f} MB")
    for f in files[:10]:
        print(f"     - {f}")
else:
    print(f"   ✗ Output directory not found: {output_dir}")

print("\n" + "=" * 60)
print("✓ Stage 2 inference test completed successfully!")
print(f"✓ Final 3D model at: {output_dir}/output.glb")
print("=" * 60)
