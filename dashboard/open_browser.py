"""Open the default browser once the Streamlit server accepts connections.  Usage: python open_browser.py [port]"""
import socket
import sys
import time
import webbrowser


def wait_for_port(port, timeout=120, host="127.0.0.1"):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8501
    if wait_for_port(port):
        webbrowser.open(f"http://localhost:{port}")
