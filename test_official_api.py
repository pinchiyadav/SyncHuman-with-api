"""
Test script for Official SyncHuman API

Tests the full two-stage pipeline with maximum quality
"""
import requests
import json
import time
from pathlib import Path

def test_api_health():
    """Test health endpoint"""
    print("\n" + "="*70)
    print("TEST: API Health Check")
    print("="*70)

    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API is healthy")
            print(f"  Service: {data.get('service')}")
            print(f"  Stage 1: {data.get('stage1')}")
            print(f"  Stage 2: {data.get('stage2')}")
            print(f"  Approach: {data.get('approach')}")
            return True
        else:
            print(f"✗ API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ API connection failed: {str(e)}")
        return False

def test_api_info():
    """Test info endpoint"""
    print("\n" + "="*70)
    print("TEST: API Info")
    print("="*70)

    try:
        response = requests.get('http://localhost:8000/info', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API info retrieved")
            print(f"  Version: {data.get('api_version')}")
            print(f"  Approach: {data.get('approach')}")
            print(f"  Total time estimate: {data.get('total_time_estimate')}")

            gpu = data.get('gpu_info', {})
            print(f"  GPU: {gpu.get('device')}")
            print(f"  GPU Memory: {gpu.get('memory_mb', 0) // 1024}GB")
            return True
        else:
            print(f"✗ API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Failed to get API info: {str(e)}")
        return False

def test_generate_3d(image_path: str):
    """Test 3D generation endpoint with real image"""
    print("\n" + "="*70)
    print(f"TEST: 3D Generation with {Path(image_path).name}")
    print("="*70)

    if not Path(image_path).exists():
        print(f"✗ Image not found: {image_path}")
        return False

    try:
        print(f"📤 Uploading image: {image_path}")
        with open(image_path, 'rb') as f:
            files = {'image': (Path(image_path).name, f, 'image/png')}

            start_time = time.time()
            response = requests.post(
                'http://localhost:8000/generate',
                files=files,
                timeout=600  # 10 minutes
            )
            elapsed = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            print(f"✓ Generation completed in {elapsed:.1f} seconds")
            print(f"  Job ID: {result.get('job_id')}")
            print(f"  Status: {result.get('status')}")
            print(f"  Approach: {result.get('approach')}")
            print(f"  Alpha coverage: {result.get('alpha_coverage', 'N/A')}")

            # Stage 1 info
            s1 = result.get('stage1', {})
            print(f"\n  Stage 1:")
            print(f"    Status: {s1.get('status')}")
            print(f"    Output: {s1.get('output_dir')}")
            if s1.get('files'):
                print(f"    Files: {len(s1['files'])} outputs")
                for key, val in list(s1['files'].items())[:3]:
                    print(f"      - {key}: {val}")

            # Stage 2 info
            s2 = result.get('stage2', {})
            print(f"\n  Stage 2:")
            print(f"    Status: {s2.get('status')}")
            print(f"    Output: {s2.get('output_dir')}")
            if s2.get('files'):
                print(f"    Files: {len(s2['files'])} outputs")
                for key, val in s2['files'].items():
                    print(f"      - {key}: {val}")

                # Show GLB path
                glb_path = Path(s2.get('output_dir', '')) / 'output.glb'
                print(f"\n    Final 3D Model: {glb_path}")

            return True
        else:
            print(f"✗ API returned status {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except requests.exceptions.Timeout:
        print(f"✗ Request timed out (>10 minutes)")
        return False
    except Exception as e:
        print(f"✗ Generation failed: {str(e)}")
        return False

def main():
    print("\n" + "="*70)
    print("SyncHuman Official API Test Suite")
    print("="*70)
    print("Testing: api_server_official.py")
    print("Approach: Official two-stage pipeline for maximum quality")
    print("="*70)

    # Test 1: Health check
    if not test_api_health():
        print("\n✗ API is not running. Start it with:")
        print("  source /opt/conda/bin/activate SyncHuman")
        print("  python api_server_official.py")
        return

    # Test 2: Info endpoint
    test_api_info()

    # Test 3: Generate 3D with Dussehra image
    dussehra_image = Path("test_dussehra.png")
    if dussehra_image.exists():
        print(f"\n📥 Found test image: {dussehra_image}")
        test_generate_3d(str(dussehra_image))
    else:
        print(f"\n⚠ Test image not found: {dussehra_image}")
        print("  Download it with:")
        print('  curl -L "https://www.pngfind.com/pngs/b/41-416466_dussehra-png.png" -o test_dussehra.png')

    print("\n" + "="*70)
    print("Test suite completed")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
