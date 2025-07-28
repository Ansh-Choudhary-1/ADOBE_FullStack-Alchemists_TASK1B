#!/usr/bin/env python3
"""
Enhanced Setup for Challenge 1B
Handles any domain automatically
"""

import subprocess
import sys
import os
import json

def install_requirements():
    """Install required packages"""
    print("📦 Installing enhanced requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def download_nltk_data():
    """Download required NLTK data"""
    print("📚 Downloading NLTK data...")
    try:
        import nltk
        
        datasets = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']
        for dataset in datasets:
            try:
                nltk.download(dataset, quiet=True)
            except:
                pass  # Continue if download fails
        
        print("✅ NLTK data downloaded successfully")
        return True
    except Exception as e:
        print(f"⚠️ Warning: Could not download NLTK data: {e}")
        return False

def detect_domain():
    """Detect document domain from input"""
    try:
        if os.path.exists("challenge1b_input.json"):
            with open("challenge1b_input.json", 'r') as f:
                data = json.load(f)
            
            persona = data.get('persona', {}).get('role', '').lower()
            task = data.get('job_to_be_done', {}).get('task', '').lower()
            combined = f"{persona} {task}"
            
            # Detect domain
            domain_indicators = {
                'research': ['research', 'academic', 'study', 'analysis', 'literature'],
                'business': ['business', 'financial', 'revenue', 'market', 'strategy'],
                'medical': ['medical', 'clinical', 'patient', 'treatment', 'diagnosis'],
                'legal': ['legal', 'law', 'regulation', 'compliance', 'contract'],
                'education': ['student', 'course', 'study', 'learning', 'education'],
                'travel': ['travel', 'trip', 'vacation', 'tourism', 'planner']
            }
            
            detected_domains = []
            for domain, indicators in domain_indicators.items():
                if any(indicator in combined for indicator in indicators):
                    detected_domains.append(domain)
            
            if detected_domains:
                print(f"🎯 Detected domain(s): {', '.join(detected_domains)}")
            else:
                print("🎯 Domain: Generic (will auto-adapt)")
                
    except Exception as e:
        print(f"⚠️ Could not detect domain: {e}")

def setup_enhanced_environment():
    """Complete enhanced environment setup"""
    print("🚀 Setting up Enhanced Challenge 1B Environment")
    print("=" * 60)
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Download NLTK data
    download_nltk_data()
    
    # Detect domain
    detect_domain()
    
    # Check required files
    required_files = ["challenge1b_main.py", "working_pdf_extractor.py"]
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️ Missing files: {missing_files}")
        if "working_pdf_extractor.py" in missing_files:
            print("Please copy working_pdf_extractor.py from task 1 directory")
    else:
        print("✅ All required files present")
    
    print("\n🎉 Enhanced setup complete!")
    print("Ready to process ANY domain!")
    return True

if __name__ == "__main__":
    success = setup_enhanced_environment()
    exit(0 if success else 1)