#!/usr/bin/env python3
import http.server
import os
import socketserver
import sys
import time
from pathlib import Path
from typing import Tuple

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
LOG_DIR = Path(os.getenv("LOG_DIR", "/log"))
EXPORTER_PORT = int(os.getenv("EXPORTER_PORT", "8000"))

def scan_directory(path: Path) -> Tuple[int, int, int]:
    total_bytes = 0
    total_files = 0
    try:
        if not path.exists():
            raise FileNotFoundError(path)
        for root, _, files in os.walk(path, followlinks=False):
            for name in files:
                file_path = Path(root) / name
                try:
                    if not file_path.is_symlink():
                        total_bytes += file_path.stat().st_size
                    total_files += 1
                except OSError:
                    # Count the failure once by surfacing an error for the entire path.
                    raise
        return total_bytes, total_files, 0
    except OSError:
        return 0, 0, 1

def render_metrics() -> bytes:
    data_bytes, data_files, data_error = scan_directory(DATA_DIR)
    log_bytes, log_files, log_error = scan_directory(LOG_DIR)
    now = time.time()

    body = f"""# HELP trex_mqtt_data_dir_bytes Size of the EMQX data directory in bytes.
# TYPE trex_mqtt_data_dir_bytes gauge
trex_mqtt_data_dir_bytes {data_bytes}
# HELP trex_mqtt_data_dir_files Number of files in the EMQX data directory.
# TYPE trex_mqtt_data_dir_files gauge
trex_mqtt_data_dir_files {data_files}
# HELP trex_mqtt_data_dir_stat_error 1 if the exporter could not stat the EMQX data directory.
# TYPE trex_mqtt_data_dir_stat_error gauge
trex_mqtt_data_dir_stat_error {data_error}
# HELP trex_mqtt_log_dir_bytes Size of the EMQX log directory in bytes.
# TYPE trex_mqtt_log_dir_bytes gauge
trex_mqtt_log_dir_bytes {log_bytes}
# HELP trex_mqtt_log_dir_files Number of files in the EMQX log directory.
# TYPE trex_mqtt_log_dir_files gauge
trex_mqtt_log_dir_files {log_files}
# HELP trex_mqtt_log_dir_stat_error 1 if the exporter could not stat the EMQX log directory.
# TYPE trex_mqtt_log_dir_stat_error gauge
trex_mqtt_log_dir_stat_error {log_error}
# HELP trex_mqtt_volume_metrics_scrape_timestamp_seconds Unix time of the last successful scrape generation.
# TYPE trex_mqtt_volume_metrics_scrape_timestamp_seconds gauge
trex_mqtt_volume_metrics_scrape_timestamp_seconds {now:.3f}
"""
    return body.encode("utf-8")

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path not in ("/metrics", "/metrics/"):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"not found")
            return

        payload = render_metrics()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args) -> None:
        return

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True

if __name__ == "__main__":
    with ThreadedTCPServer(("0.0.0.0", EXPORTER_PORT), Handler) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            sys.exit(0)
