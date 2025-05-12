from internlm.param_server.master.manager import StateManager
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import urllib.parse

class MasterServiceHandler(BaseHTTPRequestHandler):
    state_manager = None
    def do_GET(self):
        # 解析 URL 和参数
        parsed_path = urllib.parse.urlparse(self.path)
        query_params = urllib.parse.parse_qs(parsed_path.query)

        # 只处理 /number 路径
        if parsed_path.path == '/query_global_status':
            # 获取 value 参数，默认是10
            last_of_n = int(query_params.get('last_of_n', [10])[0])
            print("llllllllllllllllllll ", last_of_n)
            result = dict(current_global_status=self.state_manager.get_current_global_status(), global_status_history_list=self.state_manager.get_global_status_history_list(last_of_n))

            # 构造响应
            response = result
            response_bytes = json.dumps(response).encode('utf-8')

            # 返回 HTTP 200 OK
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(response_bytes)))
            self.end_headers()
            self.wfile.write(response_bytes)
        elif parsed_path.path == '/query_alive_ps':
            result = dict(alive_ps=self.state_manager.get_alive_ps())
            response = result
            response_bytes = json.dumps(response).encode('utf-8') 
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(response_bytes)))
            self.end_headers()
            self.wfile.write(response_bytes)
        elif parsed_path.path == '/query_alive_group':
            result = dict(alive_ps=self.state_manager.get_alive_group())
            response = result
            response_bytes = json.dumps(response).encode('utf-8') 
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(response_bytes)))
            self.end_headers()
            self.wfile.write(response_bytes)
        elif parsed_path.path == '/query_group_status':
            # 获取 value 参数，默认是10
            group_id = int(query_params.get('group_id', [0])[0])
            last_of_n = int(query_params.get('last_of_n', [10])[0])
            result = dict(self.state_manager.get_group_status(group_id, last_of_n))

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
    server = HTTPServer(('0.0.0.0', 8080), MasterServiceHandler)
    print('Starting server on port 8080...')
    server.serve_forever()