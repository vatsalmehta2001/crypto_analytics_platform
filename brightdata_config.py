"""
Configuration helper for Brightdata Twitter Scraper integration.
This file helps manage your Brightdata API credentials securely.
"""
import os
import json
from pathlib import Path

CONFIG_FILE = "brightdata_config.json"

def setup_brightdata_config():
    """Interactive setup for Brightdata API configuration"""
    print("\n=== Brightdata API Configuration ===\n")
    
    api_key = input("Enter your Brightdata API Key: ").strip()
    
    # Verify the API key format
    if not api_key or len(api_key) < 10:
        print("Warning: The API key seems too short. Please check it.")
        confirm = input("Continue anyway? (y/n): ").lower()
        if confirm != 'y':
            print("Configuration aborted.")
            return False
    
    # Create config data
    config_data = {
        "api_key": api_key,
        "rate_limit": 60,  # requests per minute (adjust based on your plan)
        "max_retries": 3
    }
    
    # Save to config file
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"\nConfiguration saved to {CONFIG_FILE}")
        print("IMPORTANT: Keep this file secure and do not share it or commit to version control.")
        
        # Set file permissions to be readable only by the owner
        config_path = Path(CONFIG_FILE)
        config_path.chmod(0o600)  # Read/write for owner only
        
        return True
    except Exception as e:
        print(f"Error saving configuration: {e}")
        return False

def load_brightdata_config():
    """Load Brightdata API configuration from file or environment variables"""
    # First try to get from environment variable
    api_key = os.environ.get('BRIGHTDATA_API_KEY')
    if api_key:
        return {
            "api_key": api_key,
            "rate_limit": int(os.environ.get('BRIGHTDATA_RATE_LIMIT', 60)),
            "max_retries": int(os.environ.get('BRIGHTDATA_MAX_RETRIES', 3))
        }
    
    # Then try to load from config file
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        else:
            print(f"Configuration file {CONFIG_FILE} not found.")
            print("Run setup_brightdata_config() to create it, or set BRIGHTDATA_API_KEY environment variable.")
            return None
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None

def get_api_key():
    """Simple helper to get just the API key"""
    config = load_brightdata_config()
    if config:
        return config.get("api_key")
    return None

if __name__ == "__main__":
    # Run the configuration setup when executed directly
    setup_result = setup_brightdata_config()
    if setup_result:
        print("\nTesting configuration...")
        config = load_brightdata_config()
        if config:
            print("Configuration loaded successfully!")
            print(f"API Key: {config['api_key'][:5]}{'*' * (len(config['api_key']) - 8)}{config['api_key'][-3:]}")
            print(f"Rate Limit: {config['rate_limit']} requests per minute")
            print(f"Max Retries: {config['max_retries']}")
        else:
            print("Failed to load configuration after setup.")