from core.providers.vllm.base import VLLMProviderBase


class VLLMProvider(VLLMProviderBase):
    def __init__(self, config):
        # No configuration needed for the fake provider.
        pass

    def response(self, question, base64_image):
        return "拍照完成"
