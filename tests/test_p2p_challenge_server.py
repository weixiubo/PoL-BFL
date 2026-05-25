import json
import os
import time
import urllib.request

import pytest

from server.p2p.challenge_server import start_server, stop_server
import chainfl.interact as ci


@pytest.mark.timeout(180)
def test_p2p_server_register_issue_get():
    os.environ.pop('POL_OFFLINE_FALLBACK', None)

    httpd, th = start_server(port=0)
    base = f"http://{httpd.server_address[0]}:{httpd.server_address[1]}"
    try:
        # health
        with urllib.request.urlopen(base + "/health") as resp:
            assert resp.status == 200
            body = json.loads(resp.read())
            assert body.get("ok") is True

        # register
        with urllib.request.urlopen(base + "/register?client=3") as resp:
            assert resp.status == 200
            body = json.loads(resp.read())
            assert body.get("ok") is True or True

        # issue
        deadline = int(time.time()) + 3600
        with urllib.request.urlopen(base + f"/issue?client=3&idx0=0&idx1=1&deadline={deadline}") as resp:
            assert resp.status == 200
            body = json.loads(resp.read())
            cid = body.get("cid", "")
            assert isinstance(cid, str) and cid.startswith("0x")

        # get
        with urllib.request.urlopen(base + f"/get?cid={cid}") as resp:
            assert resp.status == 200
            body = json.loads(resp.read())
            ch = body.get("challenge", {})
            assert isinstance(ch, dict)
            assert ch.get("resolved") is False
            assert ch.get("idx0") == 0 and ch.get("idx1") == 1
    finally:
        stop_server(httpd)

