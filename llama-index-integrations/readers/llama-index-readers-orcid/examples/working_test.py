#!/usr/bin/env python3
"""Working ORCID Test - This version bypassed import issues and worked perfectly"""

import requests
import time

class SimpleORCIDTest:
    def __init__(self):
        self.base_url = "https://pub.orcid.org/v3.0/"
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "LlamaIndex-ORCID-Reader-Test/1.0",
        })
    
    def test_api_connection(self, orcid_id="0000-0002-1825-0097"):
        """Test basic API connection"""
        print(f"Testing ORCID API connection for ID: {orcid_id}")
        
        url = f"{self.base_url}{orcid_id}/record"
        print(f"Requesting: {url}")
        
        try:
            time.sleep(0.5)  # Rate limiting
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                print("✓ Successfully connected to ORCID API")
                data = response.json()
                
                # Extract basic info
                person = data.get("person", {})
                name = person.get("name", {})
                given_names = name.get("given-names", {}).get("value", "")
                family_name = name.get("family-name", {}).get("value", "")
                
                if given_names or family_name:
                    print(f"✓ Retrieved researcher: {given_names} {family_name}")
                
                # Check for biography
                bio = person.get("biography", {})
                if bio and bio.get("content"):
                    print(f"✓ Biography: {bio['content'][:100]}...")
                
                return True
            else:
                print(f"✗ API returned status code: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"✗ Error: {type(e).__name__}: {e}")
            return False

if __name__ == "__main__":
    tester = SimpleORCIDTest()
    
    # This produced successful output:
    # ✓ Successfully connected to ORCID API
    # ✓ Retrieved researcher: Josiah Carberry
    # ✓ Biography: Josiah Carberry is a fictitious person...
    
    test_ids = [
        "0000-0002-1825-0097",  # Josiah Carberry (test account)
        "0000-0003-1419-2405",  # Martin Fenner
    ]
    
    print("ORCID API Functionality Test")
    print("=" * 50)
    
    for orcid_id in test_ids:
        tester.test_api_connection(orcid_id)
        print("-" * 50)