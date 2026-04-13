#!/usr/bin/env python3
"""
View an MJPEG stream from any camera server.

Usage: python server/view_stream.py <ip>
Or open http://<ip>:8080 in a browser.  q=quit
"""

import sys
import time
import threading

import cv2

ip = sys.argv[1] if len(sys.argv) > 1 else (print("Usage: view_stream.py <ip>") or sys.exit(1))

_frame = None
_flock = threading.Lock()


def _reader():
    global _frame
    while True:
        cap = cv2.VideoCapture(f"http://{ip}:8080")
        while True:
            ret, f = cap.read()
            if not ret:
                break
            with _flock:
                _frame = f
        cap.release()
        time.sleep(1)


threading.Thread(target=_reader, daemon=True).start()

while True:
    with _flock:
        frame = _frame
    if frame is not None:
        cv2.imshow("Stream  [q] quit", frame)
    if cv2.waitKey(15) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
