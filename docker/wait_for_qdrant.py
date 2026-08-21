"""
docker/wait_for_qdrant.py

Blocks until Qdrant answers its readiness endpoint, or exits non-zero
after a timeout. Pure stdlib (urllib) so it needs nothing beyond the
base Python image -- no curl/wget/nc dependency on either this image
or the qdrant image.

Used by docker/entrypoint-backend.sh and docker/entrypoint-ingestion.sh
so neither the API nor the ingestion job races Qdrant's startup.
"""

import os
import sys
import time
import urllib.request

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333").rstrip("/")
READY_URL = f"{QDRANT_URL}/readyz"
TIMEOUT_SECONDS = int(os.environ.get("QDRANT_WAIT_TIMEOUT", "60"))
POLL_INTERVAL_SECONDS = 1

deadline = time.time() + TIMEOUT_SECONDS

while time.time() < deadline:
    try:
        with urllib.request.urlopen(READY_URL, timeout=3) as resp:
            if resp.status == 200:
                print(f"[wait_for_qdrant] Qdrant ready at {QDRANT_URL}")
                sys.exit(0)
    except Exception as exc:
        print(f"[wait_for_qdrant] not ready yet ({exc}); retrying...")
    time.sleep(POLL_INTERVAL_SECONDS)

print(f"[wait_for_qdrant] Qdrant NOT ready after {TIMEOUT_SECONDS}s at {QDRANT_URL}", file=sys.stderr)
sys.exit(1)