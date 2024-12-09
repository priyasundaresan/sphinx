import os
from http.server import HTTPServer, SimpleHTTPRequestHandler


# Define a class that extends SimpleHTTPRequestHandler to add CORS headers
class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()


def http_server():
    dir_to_search = os.path.abspath(os.path.dirname(__file__))
    os.chdir(dir_to_search)
    server_address = ("", 8080)
    httpd = HTTPServer(server_address, CORSRequestHandler)
    print(f"Starting HTTP server on port 8080")
    httpd.serve_forever()


if __name__ == "__main__":
    http_server()
