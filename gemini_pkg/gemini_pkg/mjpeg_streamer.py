import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2


class MjpegStreamer:
    # Serves the latest frame as an MJPEG stream: open http://<car-ip>:<port>/ in a browser
    def __init__(self, port=8080, width=424, height=240, quality=60):
        self.size = (width, height)
        self.quality = quality
        self.jpeg = None
        self.condition = threading.Condition()

        streamer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                # Low latency: no Nagle delay, small send buffer so old frames don't queue up on slow Wi-Fi
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 32 * 1024)
                try:
                    while True:
                        # Wait for a new frame
                        with streamer.condition:
                            streamer.condition.wait(timeout=1.0)
                            jpeg = streamer.jpeg
                        if jpeg is None:
                            continue
                        # One write per frame
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n'
                                         b'Content-Length: ' + str(len(jpeg)).encode() + b'\r\n\r\n'
                                         + jpeg + b'\r\n')
                except (BrokenPipeError, ConnectionResetError):
                    pass  # Browser closed

            def log_message(self, format, *args):
                pass  # No log line for every request

        self.server = ThreadingHTTPServer(('0.0.0.0', port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def update(self, bgr_image):
        # Downscale + JPEG encode the frame, then wake up the waiting clients
        small = cv2.resize(bgr_image, self.size, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        with self.condition:
            self.jpeg = buf.tobytes()
            self.condition.notify_all()

    def shutdown(self):
        self.server.shutdown()
        self.server.server_close()
