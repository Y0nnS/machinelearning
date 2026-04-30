"""
Convenience startup script — starts backend from the repo root
"""
import subprocess
import sys
import os

os.chdir("backend")
subprocess.run(
    [sys.executable, "-m", "uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
    check=True,
)
