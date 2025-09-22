import json
import time
import base64
import hashlib
import hmac
from aiohttp import web
from core.utils.util import get_local_ip
from core.api.base_handler import BaseHandler

TAG = __name__


class OTAHandler(BaseHandler):
    def __init__(self, config: dict):
        super().__init__(config)
        
    def generate_password_signature(self, content: str, secret_key: str) -> str:
        """生成MQTT密码签名
        
        Args:
            content: 签名内容 (clientId + '|' + username)
            secret_key: 密钥
            
        Returns:
            str: Base64编码的HMAC-SHA256签名
        """
        try:
            hmac_obj = hmac.new(secret_key.encode('utf-8'), content.encode('utf-8'), hashlib.sha256)
            signature = hmac_obj.digest()
            return base64.b64encode(signature).decode('utf-8')
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"生成MQTT密码签名失败: {e}")
            return ""

    def _get_websocket_url(self, local_ip: str, port: int) -> str:
        """获取websocket地址

        Args:
            local_ip: 本地IP地址
            port: 端口号

        Returns:
            str: websocket地址
        """
        server_config = self.config["server"]
        websocket_config = server_config.get("websocket", "")

        if "你的" not in websocket_config:
            return websocket_config
        else:
            return f"ws://{local_ip}:{port}/xiaozhi/v1/"

    async def handle_post(self, request):
        """处理 OTA POST 请求"""
        try:
            data = await request.text()
            self.logger.bind(tag=TAG).debug(f"OTA请求方法: {request.method}")
            self.logger.bind(tag=TAG).debug(f"OTA请求头: {request.headers}")
            self.logger.bind(tag=TAG).debug(f"OTA请求数据: {data}")

            device_id = request.headers.get("device-id", "")
            if device_id:
                self.logger.bind(tag=TAG).info(f"OTA请求设备ID: {device_id}")
            else:
                raise Exception("OTA请求设备ID为空")

            data_json = json.loads(data)

            server_config = self.config["server"]
            port = int(server_config.get("port", 8000))
            local_ip = get_local_ip()

            return_json = {
                "server_time": {
                    "timestamp": int(round(time.time() * 1000)),
                    "timezone_offset": server_config.get("timezone_offset", 8) * 60,
                },
                "firmware": {
                    "version": data_json["application"].get("version", "1.0.0"),
                    "url": "",
                },
            }
            
            mqtt_gateway_config = server_config.get("mqtt_gateway", {})
            mqtt_gateway_host = mqtt_gateway_config.get("host", "")
            mqtt_gateway_port = mqtt_gateway_config.get("port", "")
            
            
            if mqtt_gateway_host:  # 配置了mqtt_gateway，使用MQTT和UDP协议传输
                # 从设备型号获取group_id，与manager-api保持一致
                # 客户端ID格式：groupId@@@macAddress@@@macAddress
                # 尝试从请求数据中获取设备型号
                device_model = "default"
                try:
                    # 假设设备型号在request数据的某个字段中
                    if "device" in data_json and isinstance(data_json["device"], dict):
                        device_model = data_json["device"].get("model", "default")
                    elif "model" in data_json:
                        device_model = data_json["model"]
                    # 为了保证格式一致性，进行与manager-api相同的处理
                    group_id = f"GID_{device_model}".replace(":", "_").replace(" ", "_")
                except Exception as e:
                    self.logger.bind(tag=TAG).error(f"获取设备型号失败: {e}")
                    # 如果获取失败，使用配置文件中的默认值
                    group_id = mqtt_gateway_config.get("group_id", "GID_default").replace(":", "_")
                
                mac_address_safe = device_id.replace(":", "_")
                mqtt_client_id = f"{group_id}@@@{mac_address_safe}@@@{mac_address_safe}"

                # 构建用户数据（包含IP等信息）
                user_data = {
                    "ip": "unknown"
                }
                
                # 将用户数据编码为Base64 JSON
                try:
                    user_data_json = json.dumps(user_data)
                    username = base64.b64encode(user_data_json.encode('utf-8')).decode('utf-8')
                except Exception as e:
                    self.logger.bind(tag=TAG).error(f"生成用户名失败: {e}")
                    username = ""

                # 获取MQTT签名密钥
                password = ""
                signature_key = server_config.get("mqtt_signature_key", "")
                if signature_key:
                    # 使用签名密钥生成密码
                    password = self.generate_password_signature(mqtt_client_id + "|" + username, signature_key)
                    if not password:
                        # 如果签名生成失败，使用配置文件中的密码
                        password = mqtt_gateway_config.get("password", "")
                else:
                    # 如果没有签名密钥，使用配置文件中的密码
                    password = mqtt_gateway_config.get("password", "")
                    self.logger.bind(tag=TAG).warning("缺少MQTT签名密钥，使用配置文件中的密码")

                # 构建MQTT配置
                endpoint = f"{mqtt_gateway_host}:{mqtt_gateway_port}"
                return_json["mqtt_gateway"] = {
                    "endpoint": endpoint,
                    "client_id": mqtt_client_id,
                    "username": username,
                    "password": password,
                    "publish_topic": "device-server",
                    "subscribe_topic": f"devices/p2p/{mac_address_safe}"
                }
                self.logger.bind(tag=TAG).info(f"为设备 {device_id} 下发MQTT网关配置")
                
                # 添加UDP网关配置
                if "udp_gateway" in server_config:
                    udp_config = server_config["udp_gateway"]
                    udp_host = udp_config.get("host", "")
                    udp_port = udp_config.get("port", "")
                    if udp_host:
                        return_json["udp_gateway"] = {
                            "host": udp_host,
                            "port": udp_port
                        }
                        self.logger.bind(tag=TAG).info(f"为设备 {device_id} 下发UDP网关配置")
                
            else:  # 未配置mqtt_gateway，使用WebSocket协议传输
                return_json["websocket"] = {
                    "url": self._get_websocket_url(local_ip, port),
                }
                self.logger.bind(tag=TAG).info(f"未配置MQTT网关，为设备 {device_id} 下发WebSocket配置")
            
            response = web.Response(
                text=json.dumps(return_json, separators=(",", ":")),
                content_type="application/json",
            )
        except Exception as e:
            return_json = {"success": False, "message": "request error."}
            response = web.Response(
                text=json.dumps(return_json, separators=(",", ":")),
                content_type="application/json",
            )
        finally:
            self._add_cors_headers(response)
            return response

    async def handle_get(self, request):
        """处理 OTA GET 请求"""
        try:
            server_config = self.config["server"]
            local_ip = get_local_ip()
            port = int(server_config.get("port", 8000))
            websocket_url = self._get_websocket_url(local_ip, port)
            message = f"OTA接口运行正常，向设备发送的websocket地址是：{websocket_url}"
            response = web.Response(text=message, content_type="text/plain")
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"OTA GET请求异常: {e}")
            response = web.Response(text="OTA接口异常", content_type="text/plain")
        finally:
            self._add_cors_headers(response)
            return response
