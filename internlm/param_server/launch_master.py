import sys

import grpc

sys.path.append(".")


from concurrent import futures

from loguru import logger

import internlm.param_server.proto.master_pb2_grpc as master_pb2_grpc
import threading
from internlm.param_server.common import config
from internlm.param_server.master.master_service import MasterControlFlowServicer
from internlm.param_server.master.master_http_service import MasterServiceHandler
from http.server import HTTPServer

def http_main(state_manager):
    logger.info(f"http server has started")
    MasterServiceHandler.state_manager = state_manager
    http_server = HTTPServer(('0.0.0.0', 8080), MasterServiceHandler)
    http_server.serve_forever()
    
def main():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service = MasterControlFlowServicer()
    master_pb2_grpc.add_ControlFlowServiceServicer_to_server(service, server)
    server.add_insecure_port(f"[::]:{config.MASTER_PORT}")
    logger.info(f"master is running on [::]:{config.MASTER_PORT}")
    http_server_thread = threading.Thread(target=http_main, args=(service.state_manager,), daemon=True)
    http_server_thread.start()
    server.start()
    server.wait_for_termination()





if __name__ == "__main__":
    main()
