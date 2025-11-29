import requests
import json
import time

# Start the inference
print("Sending inference request to API...")
print("This will take ~2 minutes...")

with open('test_image2.png', 'rb') as f:
    files = {
        'image': ('test_image2.png', f, 'image/png')
    }
    data = {
        'stage1_steps': '50'
    }
    
    response = requests.post(
        'http://localhost:8000/generate',
        files=files,
        data=data,
        timeout=300
    )

print(f"\nResponse Status: {response.status_code}")
print(f"Response Content-Type: {response.headers.get('content-type', 'unknown')}")

if response.status_code == 200:
    try:
        result = response.json()
        print("\n✓ API Response (formatted):")
        print(json.dumps(result, indent=2))
    except:
        print("Response (raw):")
        print(response.text[:500])
else:
    print(f"\n✗ Error: {response.status_code}")
    print(response.text[:500])
