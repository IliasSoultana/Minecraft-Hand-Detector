import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))
from pynput import keyboard
def test_keyboard():
    print("Testing keyboard input to Minecraft...")
    print("This will send the 'w' key (move forward) in 3 seconds")
    print("Make sure Minecraft is the active window!")
    for i in range(3, 0, -1):
        print(f"Sending 'w' in {i}...")
        time.sleep(1)
    kb = keyboard.Controller()
    print("Sending 'w' key now!")
    kb.press('w')
    time.sleep(0.5)
    kb.release('w')
    print("Key sent! Did your character move forward in Minecraft?")
if __name__ == "__main__":
    test_keyboard()