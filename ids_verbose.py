"""
Improved Launcher Script for IDS System
Shows IDS output and errors in real-time

Author: Tan Yi FEng
"""

import subprocess
import sys
import os
import time
import threading
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required = ['streamlit', 'pandas', 'plotly', 'numpy', 'sklearn', 'yaml']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing dependencies:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\n📦 Install with:")
        print("   pip install -r requirements_dashboard.txt")
        return False
    return True

def check_files():
    """Check if required files exist"""
    required_files = [
        'dashboard.py',
        'ultrafast_ids.py'
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
    
    if missing:
        print("❌ Missing required files:")
        for file in missing:
            print(f"   - {file}")
        return False
    return True

def stream_output(pipe, prefix, color_code=""):
    """Stream output from subprocess in real-time"""
    reset_code = "\033[0m" if color_code else ""
    try:
        for line in iter(pipe.readline, ''):
            if line:
                print(f"{color_code}{prefix}{line.rstrip()}{reset_code}")
    except:
        pass

def start_ids_verbose():
    """Start the IDS simulation with visible output"""
    print("🚀 Starting IDS simulation...")
    print("   " + "="*56)
    
   
    ids_process = subprocess.Popen(
        [sys.executable, 'ultrafast_ids.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    

    stdout_thread = threading.Thread(
        target=stream_output,
        args=(ids_process.stdout, "   [IDS] ", "\033[32m"),  # Green
        daemon=True
    )
    stderr_thread = threading.Thread(
        target=stream_output,
        args=(ids_process.stderr, "   [ERR] ", "\033[31m"),  # Red
        daemon=True
    )
    
    stdout_thread.start()
    stderr_thread.start()
    
    return ids_process, stdout_thread, stderr_thread

def start_dashboard():
    """Start the Streamlit dashboard"""
    print("\n🖥️  Starting dashboard...")
    print("   " + "="*56)
    
    dashboard_process = subprocess.Popen(
        ['streamlit', 'run', 'dashboard.py', '--server.headless', 'true'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
   
    stdout_thread = threading.Thread(
        target=stream_output,
        args=(dashboard_process.stdout, "   [DASH] ", "\033[34m"),  # Blue
        daemon=True
    )
    stderr_thread = threading.Thread(
        target=stream_output,
        args=(dashboard_process.stderr, "   [WARN] ", "\033[33m"),  # Yellow
        daemon=True
    )
    
    stdout_thread.start()
    stderr_thread.start()
    
    return dashboard_process

def main():
    print("="*60)
    print("🛡️  IDS ALERT SYSTEM LAUNCHER (VERBOSE MODE)")
    print("="*60)
    
   
    print("\n1️⃣  Checking dependencies...")
    if not check_dependencies():
        return
    print("   ✅ All dependencies installed")
    
   
    print("\n2️⃣  Checking required files...")
    if not check_files():
        return
    print("   ✅ All required files present")
    
   
    print("\n3️⃣  Preparing environment...")
    if Path('alerts.jsonl').exists():
        backup_name = f"alerts_backup_{int(time.time())}.jsonl"
        Path('alerts.jsonl').rename(backup_name)
        print(f"   📦 Backed up old alerts to {backup_name}")
    print("   ✅ Environment ready")
    
    print("\n" + "="*60)
    print("STARTING SYSTEM WITH LIVE OUTPUT...")
    print("="*60)
    print("\n💡 Watch for errors below. IDS should show:")
    print("   - 'Loading IDS models...'")
    print("   - '✅ Models loaded successfully'")
    print("   - '[Batch 1/X] Processing...'")
    print("\n" + "="*60 + "\n")
    
    try:
       
        ids_proc, ids_stdout, ids_stderr = start_ids_verbose()
        
       
        time.sleep(5)
        
       
        if ids_proc.poll() is not None:
            print("\n" + "="*60)
            print("❌ IDS FAILED TO START")
            print("="*60)
            print(f"   Return code: {ids_proc.poll()}")
            print("\n💡 Check the error messages above")
            print("   Common issues:")
            print("   1. Missing dataset file")
            print("   2. Missing model files")
            print("   3. Wrong file paths in ultrafast_ids_jsonl.py")
            print("   4. Insufficient memory")
            print("\n📝 Run diagnostic: python diagnose_ids.py")
            return
        
        print("\n" + "="*60)
        print("✅ IDS STARTED SUCCESSFULLY")
        print("="*60)
        print(f"   PID: {ids_proc.pid}")
        
      
        dashboard_proc = start_dashboard()
        
        time.sleep(2)
        
        if dashboard_proc.poll() is not None:
            print("\n❌ Dashboard failed to start")
            print(f"   Return code: {dashboard_proc.poll()}")
            ids_proc.terminate()
            return
        
        print("\n" + "="*60)
        print("✅ SYSTEM RUNNING")
        print("="*60)
        print(f"\n🛡️  IDS PID: {ids_proc.pid}")
        print(f"📊 Dashboard PID: {dashboard_proc.pid}")
        print(f"🌐 URL: http://localhost:8501")
        print("\n💡 Tips:")
        print("   - IDS output shown with [IDS] prefix")
        print("   - Dashboard output shown with [DASH] prefix")
        print("   - Errors shown with [ERR] or [WARN] prefix")
        print("   - Press Ctrl+C to stop both processes")
        print("\n⏸️  Monitoring processes...")
        print("="*60 + "\n")
        
       
        while True:
            time.sleep(1)
            
            
            if ids_proc.poll() is not None:
                print("\n" + "="*60)
                print("⚠️  IDS PROCESS STOPPED")
                print("="*60)
                print(f"   Return code: {ids_proc.poll()}")
                print("   This might be normal if simulation completed")
                print("   Check output above for details")
                
               
                print("\n   Dashboard is still running at http://localhost:8501")
                print("   Press Ctrl+C to stop dashboard...")
                
              
                while True:
                    time.sleep(1)
                    if dashboard_proc.poll() is not None:
                        print("\n⚠️  Dashboard stopped")
                        break
                break
            
           
            if dashboard_proc.poll() is not None:
                print("\n⚠️  Dashboard stopped unexpectedly")
                break
    
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("🛑 STOPPING SYSTEM...")
        print("="*60)
        
      
        try:
            print("   Stopping IDS...")
            ids_proc.terminate()
            ids_proc.wait(timeout=5)
            print("   ✅ IDS stopped")
        except:
            print("   ⚠️  IDS already stopped")
        
       
        try:
            print("   Stopping Dashboard...")
            dashboard_proc.terminate()
            dashboard_proc.wait(timeout=5)
            print("   ✅ Dashboard stopped")
        except:
            print("   ⚠️  Dashboard already stopped")
        
        print("\n✅ System stopped cleanly")
        print(f"📁 Alerts saved to: {Path('alerts.jsonl').absolute()}")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Attempting to stop processes...")
        try:
            ids_proc.terminate()
            dashboard_proc.terminate()
        except:
            pass

if __name__ == "__main__":
    main()