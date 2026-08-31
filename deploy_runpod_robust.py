import os
import runpod
import time
import sys

sys.stdout.reconfigure(encoding='utf-8')
runpod.api_key = "rpa_3VYTZ7GMB0Z9YDB0X57VYEYV6L1BQ78YY07K4IJ8euetii"

GPUs = ["NVIDIA RTX A4000", "NVIDIA GeForce RTX 3090", "NVIDIA RTX A5000"]

print("Attempting to deploy Notebook 2 Pod on Community Cloud...")

for gpu in GPUs:
    print(f"Trying to find capacity for: {gpu}")
    try:
        pod = runpod.create_pod(
            name="Fact-Checking-NB2",
            image_name="runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04",
            gpu_type_id=gpu,
            cloud_type="COMMUNITY",
            gpu_count=1,
            volume_in_gb=50,
            container_disk_in_gb=30,
            ports="8888/http,22/tcp",
            volume_mount_path="/workspace",
            env={
                "JUPYTER_PASSWORD": "runpod",
            }
        )
        pod_id = pod.get("id")
        print(f"✅ SUCCESS! Pod Requested! ID: {pod_id} on {gpu}")
        
        # Poll until running
        print("Waiting for pod status to be 'RUNNING'...")
        is_running = False
        for _ in range(30):
            info = runpod.get_pod(pod_id)
            if info.get("desiredStatus") == "RUNNING" and info.get("machineId") is not None:
                is_running = True
                break
            time.sleep(5)
            
        if is_running:
            print("\n🚀 POD IS UP AND RUNNING!")
            print(f"🔗 JupyterLab Access: https://{pod_id}-8888.proxy.runpod.net/")
            print("🔑 Jupyter Password: runpod")
        else:
            print("\nPod is booting up. Check your Runpod Console (runpod.io/console/pods) to track status!")
        sys.exit(0)
    except Exception as e:
        print(f" => No capacity or error: {e}")

print("❌ Failed to find any capacity for the targeted GPUs. Try again later or use Secure Cloud.")
