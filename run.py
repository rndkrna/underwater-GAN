import os
import sys
import subprocess
import time
import threading

def stream_reader(pipe, prefix):
    try:
        for line in iter(pipe.readline, ""):
            if line:
                print(f"{prefix} {line.strip()}")
    except Exception:
        pass
    finally:
        try:
            pipe.close()
        except Exception:
            pass

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check virtual environment python
    venv_python = os.path.join(base_dir, ".venv", "Scripts", "python.exe")
    if not os.path.exists(venv_python):
        venv_python = sys.executable  # fallback to current python
        
    print("=" * 60)
    print("  DeepBlue Restoration GAN - Unified Runner")
    print("=" * 60)
    
    # Copy and clean environment variables to prevent global Python conflicts (e.g. LibreOffice, Anaconda)
    clean_env = os.environ.copy()
    clean_env.pop("PYTHONHOME", None)
    clean_env.pop("PYTHONPATH", None)
    
    # 1. Start FastAPI Backend
    print("[RUNNER] Starting FastAPI Backend on http://127.0.0.1:8080...")
    backend_cmd = [
        venv_python, 
        "-m", 
        "uvicorn", 
        "api_server:app", 
        "--host", "127.0.0.1", 
        "--port", "8080"
    ]
    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=base_dir,
        env=clean_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # 2. Start React Vite Frontend
    print("[RUNNER] Starting React Vite Frontend on http://localhost:5173...")
    frontend_dir = os.path.join(base_dir, "frontend")
    # On Windows, running npm dev server via cmd requires shell=True
    npm_cmd = "npm run dev"
    frontend_proc = subprocess.Popen(
        npm_cmd,
        cwd=frontend_dir,
        shell=True,
        env=clean_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # 3. Start reading outputs asynchronously
    threading.Thread(target=stream_reader, args=(backend_proc.stdout, "[BACKEND]"), daemon=True).start()
    threading.Thread(target=stream_reader, args=(frontend_proc.stdout, "[FRONTEND]"), daemon=True).start()
    
    # 4. Monitor loop
    print("[RUNNER] System is running. Press Ctrl+C to terminate both servers.")
    try:
        while True:
            # Check if processes have died
            if backend_proc.poll() is not None:
                print(f"[RUNNER] Backend process exited unexpectedly with code {backend_proc.returncode}")
                break
            if frontend_proc.poll() is not None:
                print(f"[RUNNER] Frontend process exited unexpectedly with code {frontend_proc.returncode}")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[RUNNER] KeyboardInterrupt detected. Shutting down servers...")
    finally:
        # Terminate processes cleanly
        for proc, name in [(backend_proc, "Backend"), (frontend_proc, "Frontend")]:
            if proc.poll() is None:
                print(f"[RUNNER] Terminating {name}...")
                if sys.platform == 'win32':
                    # On Windows, use taskkill to kill the whole process tree (important for npm/shell child)
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    proc.terminate()
        print("[RUNNER] Services terminated.")

if __name__ == "__main__":
    main()
