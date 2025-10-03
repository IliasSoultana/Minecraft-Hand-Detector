"""
Test if keyboard input is working by sending a simple key press to Minecraft
"""

import time
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from pynput import keyboard

def test_keyboard():
    print("Testing keyboard input to Minecraft...")
    print("This will send the 'w' key (move forward) in 3 seconds")
    print("Make sure Minecraft is the active window!")
    
    # Wait for user to focus on Minecraft
    for i in range(3, 0, -1):
        print(f"Sending 'w' in {i}...")
        time.sleep(1)
    
    # Create keyboard controller
    kb = keyboard.Controller()
    
    # Send 'w' key
    print("Sending 'w' key now!")
    kb.press('w')
    time.sleep(0.5)  # Hold for half second
    kb.release('w')
    
    print("Key sent! Did your character move forward in Minecraft?")

if __name__ == "__main__":
    test_keyboard()