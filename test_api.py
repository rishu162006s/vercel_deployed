import requests
import json

# Your Vercel URL
VERCEL_URL = "https://your-project.vercel.app"

def test_api():
    """Test the deployed API"""
    
    # Test data
    test_payload = {
        "pdf_url": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        "questions": [
            "What is this document about?",
            "What is the purpose of this PDF?"
        ],
        "api_key": "your-secret-key-here"  # Change this
    }
    
    print(f"Testing API at: {VERCEL_URL}/aibattle/ask")
    
    try:
        response = requests.post(
            f"{VERCEL_URL}/aibattle/ask",
            json=test_payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API Test Successful!")
            print(f"Model: {result['model_used']}")
            print(f"Processing Time: {result['total_processing_time']}s")
            
            for answer in result['answers']:
                print(f"\nQ: {answer['question']}")
                print(f"A: {answer['answer']}")
                print(f"Confidence: {answer['confidence']}")
        else:
            print(f"❌ API Test Failed: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get(f"{VERCEL_URL}/aibattle/health", timeout=10)
        print(f"Health Check: {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")

if __name__ == "__main__":
    test_health()
    test_api()
