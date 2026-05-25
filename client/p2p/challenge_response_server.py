"""
客户端P2P挑战响应服务器
用于接收服务器的挑战并返回响应
"""

import json
import threading
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from typing import Dict, Optional, Callable

logger = logging.getLogger(__name__)


class ChallengeResponseHandler(BaseHTTPRequestHandler):
    """
    处理挑战请求的HTTP处理器
    """

    # 类变量，由服务器设置
    challenge_handler_func: Optional[Callable] = None

    @classmethod
    def set_handler(cls, handler_func: Callable):
        """设置处理函数"""
        cls.challenge_handler_func = handler_func
    
    def _send_json(self, code: int, payload: Dict):
        """发送JSON响应"""
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    
    def log_message(self, format, *args):
        """抑制默认的日志输出"""
        pass
    
    def do_GET(self):
        """处理GET请求"""
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query)
            
            if path == "/health":
                return self._send_json(200, {"ok": True})
            
            return self._send_json(404, {"ok": False, "error": "not found"})
        
        except Exception as e:
            logger.error(f"Error in GET handler: {e}")
            return self._send_json(500, {"ok": False, "error": str(e)})
    
    def do_POST(self):
        """处理POST请求"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode('utf-8'))
            
            parsed = urlparse(self.path)
            path = parsed.path
            
            if path == "/challenge":
                # 处理挑战请求
                client_id = payload.get('client_id')
                challenge_data = payload.get('challenge')
                
                if not client_id or not challenge_data:
                    return self._send_json(400, {
                        "ok": False,
                        "error": "missing client_id or challenge"
                    })
                
                # 调用处理函数
                if self.challenge_handler_func is None:
                    return self._send_json(500, {
                        "ok": False,
                        "error": "challenge handler not configured"
                    })
                
                try:
                    # 调用处理函数
                    handler = ChallengeResponseHandler.challenge_handler_func
                    if callable(handler):
                        response = handler(client_id, challenge_data)
                    else:
                        response = None
                    
                    if response is None:
                        return self._send_json(500, {
                            "ok": False,
                            "error": "failed to generate response"
                        })
                    
                    return self._send_json(200, {
                        "ok": True,
                        "response": response
                    })
                
                except Exception as e:
                    logger.error(f"Error handling challenge: {e}")
                    return self._send_json(500, {
                        "ok": False,
                        "error": str(e)
                    })
            
            return self._send_json(404, {"ok": False, "error": "not found"})
        
        except Exception as e:
            logger.error(f"Error in POST handler: {e}")
            return self._send_json(500, {"ok": False, "error": str(e)})


class ChallengeResponseServer:
    """
    客户端P2P挑战响应服务器
    
    功能:
    1. 接收服务器的挑战请求
    2. 调用客户端的响应处理函数
    3. 返回响应数据
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        """
        初始化服务器
        
        Args:
            host: 绑定的主机地址
            port: 绑定的端口（0表示自动分配）
        """
        self.host = host
        self.port = port
        self.httpd: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.is_running = False
        
        logger.info(f"ChallengeResponseServer initialized")
        logger.info(f"  Host: {host}")
        logger.info(f"  Port: {port}")
    
    def set_challenge_handler(self, handler_func: Callable):
        """
        设置挑战处理函数

        Args:
            handler_func: 处理函数，签名为 (client_id, challenge_data) -> response_data
        """
        ChallengeResponseHandler.set_handler(handler_func)
        logger.info("Challenge handler set")
    
    def start(self) -> bool:
        """
        启动服务器
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.httpd = HTTPServer((self.host, self.port), ChallengeResponseHandler)
            
            # 获取实际绑定的端口
            actual_port = self.httpd.server_port
            self.port = actual_port
            
            # 在后台线程中运行服务器
            self.thread = threading.Thread(
                target=self.httpd.serve_forever,
                daemon=True
            )
            self.thread.start()
            self.is_running = True
            
            logger.info(f"ChallengeResponseServer started on {self.host}:{self.port}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False
    
    def stop(self):
        """停止服务器"""
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.is_running = False
            logger.info("ChallengeResponseServer stopped")
    
    def get_url(self) -> str:
        """获取服务器URL"""
        return f"http://{self.host}:{self.port}"
    
    def is_healthy(self) -> bool:
        """检查服务器是否健康"""
        return self.is_running and self.httpd is not None


def start_challenge_response_server(
    host: str = "127.0.0.1",
    port: int = 0,
    challenge_handler: Optional[Callable] = None
) -> Optional[ChallengeResponseServer]:
    """
    启动挑战响应服务器的便利函数
    
    Args:
        host: 绑定的主机地址
        port: 绑定的端口
        challenge_handler: 挑战处理函数
        
    Returns:
        服务器实例，如果启动失败返回None
    """
    server = ChallengeResponseServer(host, port)
    
    if challenge_handler:
        server.set_challenge_handler(challenge_handler)
    
    if server.start():
        return server
    else:
        return None

