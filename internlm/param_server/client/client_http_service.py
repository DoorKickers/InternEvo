from internlm.param_server.master.manager import StateManager
from internlm.param_server.client.send_recv import get_history_status_list, get_current_status
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import urllib.parse

class ClientServiceHandler(BaseHTTPRequestHandler):
    client = None
    def do_GET(self):
        # 解析 URL 和参数
        parsed_path = urllib.parse.urlparse(self.path)
        query_params = urllib.parse.parse_qs(parsed_path.query)

        # 只处理 /number 路径
        if parsed_path.path == '/query_group_status':
            # 获取 value 参数，默认是10
            result = dict(current_group_status=get_current_status(), group_status_history_list=get_history_status_list())

            # 构造响应
            response = result
            response_bytes = json.dumps(response).encode('utf-8')

            # 返回 HTTP 200 OK
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(response_bytes)))
            self.end_headers()
            self.wfile.write(response_bytes)
        else:
            # 其他路径返回 404
            self.send_error(404, "Not Found")

if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', 8080), ClientServiceHandler)
    print('Starting server on port 8080...')
    server.serve_forever()