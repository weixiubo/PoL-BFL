"""
P2P Challenge Client
用于服务器向客户端发送挑战并接收响应
"""

import requests
import json
import logging
import time
from typing import Dict, Optional, List
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class ChallengeClient:
    """
    P2P挑战客户端
    
    功能:
    1. 向客户端发送挑战
    2. 接收客户端的响应
    3. 支持重试机制
    4. 支持超时控制
    """
    
    def __init__(self, base_url: str, timeout: int = 30, max_retries: int = 3):
        """
        初始化ChallengeClient
        
        Args:
            base_url: 客户端服务器的基础URL（如 http://localhost:8000）
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        
        logger.info(f"ChallengeClient initialized")
        logger.info(f"  Base URL: {self.base_url}")
        logger.info(f"  Timeout: {timeout}s")
        logger.info(f"  Max retries: {max_retries}")
    
    def _make_request(self, endpoint: str, params: Dict = None, 
                     method: str = 'GET', data: Dict = None) -> Optional[Dict]:
        """
        发送HTTP请求，支持重试
        
        Args:
            endpoint: API端点（如 /challenge）
            params: 查询参数
            method: HTTP方法（GET或POST）
            data: 请求体数据
            
        Returns:
            响应JSON数据，如果失败返回None
        """
        url = urljoin(self.base_url, endpoint)
        
        for attempt in range(self.max_retries):
            try:
                if method == 'GET':
                    response = requests.get(
                        url,
                        params=params,
                        timeout=self.timeout
                    )
                elif method == 'POST':
                    response = requests.post(
                        url,
                        json=data,
                        timeout=self.timeout
                    )
                else:
                    logger.error(f"Unsupported HTTP method: {method}")
                    return None
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"Request failed with status {response.status_code}: {response.text}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries})")
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error (attempt {attempt + 1}/{self.max_retries})")
            except Exception as e:
                logger.warning(f"Request error: {e} (attempt {attempt + 1}/{self.max_retries})")
            
            # 如果不是最后一次尝试，等待后重试
            if attempt < self.max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避
                logger.debug(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        logger.error(f"Failed to get response from {url} after {self.max_retries} attempts")
        return None
    
    def health_check(self) -> bool:
        """
        检查客户端服务器是否在线
        
        Returns:
            True if server is healthy, False otherwise
        """
        response = self._make_request('/health')
        return response is not None and response.get('ok', False)
    
    def send_challenge(self, client_id: str, challenge_data: Dict) -> Optional[Dict]:
        """
        向客户端发送挑战
        
        Args:
            client_id: 客户端ID
            challenge_data: 挑战数据，包含:
                - checkpoint_indices: 要验证的checkpoint索引列表
                - data_indices: 数据索引列表
                - deadline: 响应截止时间
                
        Returns:
            响应数据，如果失败返回None
        """
        payload = {
            'client_id': client_id,
            'challenge': challenge_data
        }
        
        response = self._make_request(
            '/challenge',
            method='POST',
            data=payload
        )
        
        if response and response.get('ok'):
            logger.info(f"Challenge sent to client {client_id}")
            return response.get('response')
        else:
            logger.error(f"Failed to send challenge to client {client_id}")
            return None
    
    def get_response(self, response_id: str) -> Optional[Dict]:
        """
        获取挑战响应
        
        Args:
            response_id: 响应ID
            
        Returns:
            响应数据，如果不存在返回None
        """
        response = self._make_request(
            '/response',
            params={'response_id': response_id}
        )
        
        if response and response.get('ok'):
            return response.get('response_data')
        else:
            logger.warning(f"Failed to get response {response_id}")
            return None
    
    def batch_send_challenges(self, challenges: List[Dict]) -> Dict[str, Optional[Dict]]:
        """
        批量发送挑战
        
        Args:
            challenges: 挑战列表，每个元素包含:
                - client_id: 客户端ID
                - challenge_data: 挑战数据
                
        Returns:
            {client_id: response_data} 的字典
        """
        results = {}
        
        for challenge in challenges:
            client_id = challenge.get('client_id')
            challenge_data = challenge.get('challenge_data')
            
            if not client_id or not challenge_data:
                logger.warning(f"Invalid challenge format: {challenge}")
                results[client_id] = None
                continue
            
            response = self.send_challenge(client_id, challenge_data)
            results[client_id] = response
        
        return results
    
    def wait_for_response(self, response_id: str, max_wait: int = 60) -> Optional[Dict]:
        """
        等待挑战响应
        
        Args:
            response_id: 响应ID
            max_wait: 最大等待时间（秒）
            
        Returns:
            响应数据，如果超时返回None
        """
        start_time = time.time()
        poll_interval = 1  # 初始轮询间隔（秒）
        
        while time.time() - start_time < max_wait:
            response = self.get_response(response_id)
            
            if response is not None:
                logger.info(f"Received response {response_id}")
                return response
            
            # 指数退避轮询
            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, 5)  # 最大5秒
        
        logger.error(f"Timeout waiting for response {response_id}")
        return None
    
    def register_client(self, client_id: str) -> bool:
        """
        注册客户端
        
        Args:
            client_id: 客户端ID
            
        Returns:
            True if successful, False otherwise
        """
        response = self._make_request(
            '/register',
            params={'client_id': client_id}
        )
        
        if response and response.get('ok'):
            logger.info(f"Client {client_id} registered")
            return True
        else:
            logger.error(f"Failed to register client {client_id}")
            return False

