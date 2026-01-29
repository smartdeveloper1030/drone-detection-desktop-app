"""
ISAPI-based zoom control for Hikvision PTZ cameras.
Uses PUT /ISAPI/PTZCtrl/channels/1/continuous with XML body (zoom in/out/stop).
"""
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import requests
    from requests.auth import HTTPDigestAuth
except ImportError:
    requests = None
    HTTPDigestAuth = None


class ISAPIZoom:
    """Hikvision ISAPI PTZ zoom client. Connect via HTTP Digest, then zoom in/out/stop."""

    def __init__(self):
        self._base_url: Optional[str] = None
        self._auth: Optional[HTTPDigestAuth] = None
        self._channel = 1
        self._timeout = 5.0

    def connect(self, ip: str, username: str, password: str, port: int = 80) -> Tuple[bool, str]:
        """
        Test ISAPI connection (deviceInfo). Sets base_url and auth on success.
        Returns (success, message).
        """
        if not requests or not HTTPDigestAuth:
            return False, "requests library required. Install: pip install requests"

        ip = (ip or "").strip()
        if not ip:
            return False, "IP address required"

        base = f"http://{ip}:{port}"
        auth = HTTPDigestAuth(username or "admin", password or "")

        try:
            r = requests.get(
                f"{base}/ISAPI/System/deviceInfo",
                auth=auth,
                timeout=self._timeout,
            )
            if r.status_code == 401:
                return False, "Authentication failed (401). Check username/password."
            if r.status_code != 200:
                return False, f"Connection failed (HTTP {r.status_code})"
        except requests.exceptions.Timeout:
            return False, "Connection timeout. Check IP and network."
        except requests.exceptions.RequestException as e:
            return False, f"Connection error: {e}"

        self._base_url = base
        self._auth = auth
        logger.info("ISAPI zoom connected: %s", ip)
        return True, "Connected"

    def disconnect(self) -> None:
        self._base_url = None
        self._auth = None
        logger.info("ISAPI zoom disconnected")

    @property
    def is_connected(self) -> bool:
        return self._base_url is not None and self._auth is not None

    def _put_continuous(self, pan: int, tilt: int, zoom: int) -> Tuple[bool, str]:
        """Send PUT .../continuous with PTZData XML. zoom: 1=in, -1=out, 0=stop."""
        if not self.is_connected:
            return False, "Not connected"
        url = f"{self._base_url}/ISAPI/PTZCtrl/channels/{self._channel}/continuous"
        body = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<PTZData>\n"
            f"  <pan>{pan}</pan>\n"
            f"  <tilt>{tilt}</tilt>\n"
            f"  <zoom>{zoom}</zoom>\n"
            "</PTZData>"
        )
        headers = {"Content-Type": "application/xml"}
        try:
            r = requests.put(
                url,
                data=body,
                headers=headers,
                auth=self._auth,
                timeout=self._timeout,
            )
            if r.status_code == 200:
                return True, "OK"
            return False, f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.exceptions.RequestException as e:
            return False, str(e)

    def zoom_in(self, speed: int = 50) -> Tuple[bool, str]:
        """Start continuous zoom in. speed 1–100 (may not affect all models)."""
        ok, msg = self._put_continuous(0, 0, 1)
        if ok:
            logger.debug("Zoom IN sent (speed=%s)", speed)
        return ok, msg

    def zoom_out(self, speed: int = 50) -> Tuple[bool, str]:
        """Start continuous zoom out."""
        ok, msg = self._put_continuous(0, 0, -1)
        if ok:
            logger.debug("Zoom OUT sent (speed=%s)", speed)
        return ok, msg

    def zoom_stop(self) -> Tuple[bool, str]:
        """Stop zoom (and pan/tilt)."""
        ok, msg = self._put_continuous(0, 0, 0)
        if ok:
            logger.debug("Zoom STOP sent")
        return ok, msg
