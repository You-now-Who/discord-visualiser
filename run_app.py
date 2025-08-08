#!/usr/bin/env python3
"""
Discord Messages Visualizer Runner
Run this script to start the Streamlit app
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit app"""
    try:
        # Change to the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # Run streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "main.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)

if __name__ == "__main__":
    main()
