#!/usr/bin/env python3
"""Install app on device using subprocess with timeout handling."""
import subprocess
import sys

app_path = "/Users/yw68/Library/Developer/Xcode/DerivedData/ANETestLoader-edfthwvfzqgwakebcjhgxdiuthke/Build/Products/Debug-iphoneos/ANETestLoader.app"
device_id = "00008120-0011554C3C90C01E"

cmd = ["xcrun", "devicectl", "device", "install", "app", 
       "--device", device_id, "--timeout", "300", app_path]

print(f"Running: {' '.join(cmd)}", flush=True)
try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    print(f"stdout: {result.stdout}", flush=True)
    print(f"stderr: {result.stderr}", flush=True)
    print(f"returncode: {result.returncode}", flush=True)
except subprocess.TimeoutExpired:
    print("TIMEOUT expired after 120s", flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)
