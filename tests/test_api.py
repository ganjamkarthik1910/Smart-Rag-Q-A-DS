import requests, subprocess, time, json, os, signal

def _launch_server():
    return subprocess.Popen(
        ["python", "rag/serve.py", "--ckpt", "outputs/llama3-ds-qlora"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

def test_api_smoke():
    proc = _launch_server()
    time.sleep(10)  # wait for startup
    try:
        r = requests.post("http://localhost:8000/query",
                          json={"question": "What is bias‑variance trade‑off?", "top_k": 2})
        assert r.status_code == 200
        payload = r.json()
        assert "answer" in payload and "sources" in payload
    finally:
        os.kill(proc.pid, signal.SIGTERM)
