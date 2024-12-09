import subprocess
import sys


def start_websocket_server():
    subprocess.run([sys.executable, "listen.py"])


def start_http_server():
    subprocess.run([sys.executable, "serve.py"])


if __name__ == "__main__":
    # Start WebSocket server in a separate process
    websocket_process = subprocess.Popen([sys.executable, "listen.py"])

    # Start HTTP server in the main process
    start_http_server()

    # Wait for WebSocket server process to finish
    websocket_process.wait()
