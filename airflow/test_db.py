import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test imports
from utils.db_helper import save_results_to_db, supabase

print("Testing Supabase connection...")
print(f"SUPABASE_URL: {os.getenv('SUPABASE_URL')}")

# Test connection
try:
    result = supabase.table("analysis_results").select("*").limit(1).execute()
    print("✅ Database connection successful!")
except Exception as e:
    print(f"❌ Connection failed: {e}")

# Test insert with mock data
mock_results = {
    "video_id": "test_123",
    "timestamp": "2025-12-06T10:00:00Z",
    "status": "success",
    "results": {
        "overall_score": 75,
        "performance_level": "intermediate"
    }
}

try:
    save_results_to_db("test_123", mock_results)
    print("✅ Test data inserted successfully!")
except Exception as e:
    print(f"❌ Insert failed: {e}")
