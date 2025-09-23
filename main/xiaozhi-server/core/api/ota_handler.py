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

    async def handle_post(self, request):
        """处理 OTA POST 请求 - 仅下发 MQTT 配置"""
        try:
            data = await request.text()
            self.logger.bind(tag=TAG).debug(f"OTA请求方法: {request.method}")
            self.logger.bind(tag=TAG).debug(f"OTA请求头: {request.headers}")
            self.logger.bind(tag=TAG).debug(f"OTA请求数据: {data}")

            device_id = request.headers.get("device-id", "")
            if not device_id:
                raise Exception("OTA请求设备ID为空")

            data_json = json.loads(data)

            server_config = self.config["server"]
            mqtt_endpoint = server_config.get("mqtt_gateway", "")
            signature_key = server_config.get("mqtt_signature_key", "")

            if not mqtt_endpoint:
                raise Exception("未配置 mqtt_gateway")

            # 从设备型号获取 group_id
            device_model = "default"
            try:
                if "device" in data_json and isinstance(data_json["device"], dict):
                    device_model = data_json["device"].get("model", "default")
                elif "model" in data_json:
                    device_model = data_json["model"]
                group_id = f"GID_{device_model}".replace(":", "_").replace(" ", "_")
            except Exception as e:
                self.logger.bind(tag=TAG).error(f"获取设备型号失败: {e}")
                group_id = "GID_default"

            mac_address_safe = device_id.replace(":", "_")
            mqtt_client_id = f"{group_id}@@@{mac_address_safe}@@@{mac_address_safe}"

            # 构建用户名（Base64编码的JSON）
            user_data = {"ip": "unknown"}  # 可根据需要扩展
            try:
                user_data_json = json.dumps(user_data)
                username = base64.b64encode(user_data_json.encode('utf-8')).decode('utf-8')
            except Exception as e:
                self.logger.bind(tag=TAG).error(f"生成用户名失败: {e}")
                username = ""

            # 生成密码
            password = ""
            if signature_key:
                password = self.generate_password_signature(mqtt_client_id + "|" + username, signature_key)
                if not password:
                    raise Exception("MQTT密码签名生成失败")
            else:
                raise Exception("未配置 mqtt_signature_key")

            # 构建返回的 MQTT 配置
            return_json = {
                "server_time": {
                    "timestamp": int(round(time.time() * 1000)),
                    "timezone_offset": server_config.get("timezone_offset", 8) * 60,
                },
                "firmware": {
                    "version": data_json["application"].get("version", "1.0.0"),
                    "url": "",
                },
                "mqtt": {
                    "endpoint": mqtt_endpoint,
                    "client_id": mqtt_client_id,
                    "username": username,
                    "password": password,
                    "publish_topic": "device-server",
                    "subscribe_topic": f"devices/p2p/{mac_address_safe}"
                }
            }

            self.logger.bind(tag=TAG).info(f"为设备 {device_id} 下发MQTT配置")

            response = web.Response(
                text=json.dumps(return_json, separators=(",", ":")),
                content_type="application/json",
            )
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"处理OTA请求失败: {e}")
            return_json = {"success": False, "message": str(e)}
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
            mqtt_endpoint = server_config.get("mqtt_gateway", "未配置")
            message = f"OTA接口运行正常，MQTT网关地址：{mqtt_endpoint}"
            response = web.Response(text=message, content_type="text/plain")
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"OTA GET请求异常: {e}")
            response = web.Response(text="OTA接口异常", content_type="text/plain")
        finally:
            self._add_cors_headers(response)
            return response