import asyncio
import logging
import os
import statistics
import time
from typing import Dict, Optional
import yaml
import aiohttp
from tabulate import tabulate
from core.utils.llm import create_instance as create_llm_instance

# 设置全局日志级别为 WARNING，抑制 INFO 级别日志
logging.basicConfig(level=logging.WARNING)

description = "大语言模型性能测试"
class LLMPerformanceTester:
    def __init__(self):
        self.config = self._load_config()
        self.test_sentences = self.config.get("module_test", {}).get(
            "test_sentences",
            [
                "你好，请介绍一下你自己",
                "What's the weather like today?",
                "请用100字概括量子计算的基本原理和应用前景",
            ],
        )
        self.results = {}

    def _load_config(self) -> Dict:
        """从 data/.config.yaml 加载配置"""
        config = {}
        config_file_path = os.path.join(os.getcwd(), "data", ".config.yaml")
        print(f"[DEBUG] 加载配置文件: {config_file_path}")

        if not os.path.exists(config_file_path):
            print(f"[DEBUG] 配置文件 {config_file_path} 不存在")
            return config

        try:
            with open(config_file_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                print(f"[DEBUG] 从 {config_file_path} 加载配置成功")
        except Exception as e:
            print(f"加载配置文件 {config_file_path} 失败: {str(e)}")
        return config

    async def _check_ollama_service(self, base_url: str, model_name: str) -> bool:
        """异步检查 Ollama 服务状态"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{base_url}/api/version") as response:
                    if response.status != 200:
                        print(f"Ollama 服务未启动或无法访问: {base_url}")
                        return False
                async with session.get(f"{base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get("models", [])
                        if not any(model["name"] == model_name for model in models):
                            print(
                                f"Ollama 模型 {model_name} 未找到，请先使用 `ollama pull {model_name}` 下载"
                            )
                            return False
                    else:
                        print("无法获取 Ollama 模型列表")
                        return False
                return True
            except Exception as e:
                print(f"无法连接到 Ollama 服务: {str(e)}")
                return False

    async def _test_single_sentence(
        self, llm_name: str, llm, sentence: str
    ) -> Optional[Dict]:
        """测试单个句子的性能"""
        try:
            print(f"{llm_name} 开始测试: {sentence[:20]}...")
            sentence_start = time.time()
            first_token_received = False
            first_token_time = None

            async def process_response():
                nonlocal first_token_received, first_token_time
                for chunk in llm.response(
                    "perf_test", [{"role": "user", "content": sentence}]
                ):
                    if not first_token_received and chunk.strip() != "":
                        first_token_time = time.time() - sentence_start
                        first_token_received = True
                        print(f"{llm_name} 首个 Token: {first_token_time:.3f}s")
                    yield chunk

            response_chunks = []
            async for chunk in process_response():
                response_chunks.append(chunk)

            response_time = time.time() - sentence_start
            print(f"{llm_name} 完成响应: {response_time:.3f}s")

            return {
                "name": llm_name,
                "type": "llm",
                "first_token_time": first_token_time,
                "response_time": response_time,
            }
        except Exception as e:
            print(f"{llm_name} 句子测试失败: {str(e)}")
            return None

    async def _test_llm(self, llm_name: str, config: Dict) -> Dict:
        """异步测试单个 LLM 性能"""
        try:
            # 对于 Ollama，跳过 api_key 检查并进行特殊处理
            if llm_name == "Ollama":
                base_url = config.get("base_url", "http://localhost:11434")
                model_name = config.get("model_name")
                if not model_name:
                    print("Ollama 未配置 model_name")
                    return {"name": llm_name, "type": "llm", "errors": 1}

                if not await self._check_ollama_service(base_url, model_name):
                    return {"name": llm_name, "type": "llm", "errors": 1}
            else:
                if "api_key" in config and any(
                    x in config["api_key"] for x in ["你的", "placeholder", "sk-xxx"]
                ):
                    print(f"跳过未配置的 LLM: {llm_name}")
                    return {"name": llm_name, "type": "llm", "errors": 1}

            # 获取实际类型（兼容旧配置）
            module_type = config.get("type", llm_name)
            llm = create_llm_instance(module_type, config)

            # 统一使用 UTF-8 编码
            test_sentences = [
                s.encode("utf-8").decode("utf-8") for s in self.test_sentences
            ]

            # 创建所有句子的测试任务
            sentence_tasks = []
            for sentence in test_sentences:
                sentence_tasks.append(
                    self._test_single_sentence(llm_name, llm, sentence)
                )

            # 并发执行所有句子测试
            sentence_results = await asyncio.gather(*sentence_tasks)

            # 处理结果
            valid_results = [r for r in sentence_results if r is not None]
            if not valid_results:
                print(f"{llm_name} 无有效数据，可能配置错误")
                return {"name": llm_name, "type": "llm", "errors": 1}

            first_token_times = [r["first_token_time"] for r in valid_results]
            response_times = [r["response_time"] for r in valid_results]

            # 过滤异常数据
            mean = statistics.mean(response_times)
            stdev = statistics.stdev(response_times) if len(response_times) > 1 else 0
            filtered_times = [t for t in response_times if t <= mean + 3 * stdev]

            if len(filtered_times) < len(test_sentences) * 0.5:
                print(f"{llm_name} 有效数据不足，可能网络不稳定")
                return {"name": llm_name, "type": "llm", "errors": 1}

            return {
                "name": llm_name,
                "type": "llm",
                "avg_response": sum(response_times) / len(response_times),
                "avg_first_token": sum(first_token_times) / len(first_token_times),
                "errors": 0,
            }
        except Exception as e:
            print(f"LLM {llm_name} 测试失败: {str(e)}")
            return {"name": llm_name, "type": "llm", "errors": 1}

    def _print_results(self):
        """打印测试结果"""
        llm_table = []
        for name, data in self.results.items():
            if data["errors"] == 0:
                llm_table.append(
                    [
                        name,
                        f"{data['avg_first_token']:.3f}秒",
                        f"{data['avg_response']:.3f}秒",
                    ]
                )

        if llm_table:
            print("\nLLM 性能排行:\n")
            print(
                tabulate(
                    llm_table,
                    headers=["模型名称", "首字耗时", "总耗时"],
                    tablefmt="github",
                    colalign=("left", "right", "right"),
                    disable_numparse=True,
                )
            )
        else:
            print("\n没有可用的 LLM 模块进行测试。")

    async def run(self):
        """执行全量异步测试"""
        print("开始筛选可用 LLM 模块...")

        # 创建所有测试任务
        all_tasks = []

        # LLM 测试任务
        if self.config.get("LLM") is not None:
            for llm_name, config in self.config.get("LLM", {}).items():
                # 检查配置有效性
                if llm_name == "CozeLLM":
                    if any(x in config.get("bot_id", "") for x in ["你的"]) or any(
                        x in config.get("user_id", "") for x in ["你的"]
                    ):
                        print(f"LLM {llm_name} 未配置 bot_id/user_id，已跳过")
                        continue
                elif "api_key" in config and any(
                    x in config["api_key"] for x in ["你的", "placeholder", "sk-xxx"]
                ):
                    print(f"LLM {llm_name} 未配置 api_key，已跳过")
                    continue

                # 对于 Ollama，先检查服务状态
                if llm_name == "Ollama":
                    base_url = config.get("base_url", "http://localhost:11434")
                    model_name = config.get("model_name")
                    if not model_name:
                        print("Ollama 未配置 model_name")
                        continue

                    if not await self._check_ollama_service(base_url, model_name):
                        continue

                print(f"添加 LLM 测试任务: {llm_name}")
                all_tasks.append(self._test_llm(llm_name, config))

        print(f"\n找到 {len(all_tasks)} 个可用 LLM 模块")
        print("\n开始并发测试所有模块...\n")

        # 并发执行所有测试任务
        all_results = await asyncio.gather(*all_tasks, return_exceptions=True)

        # 处理结果
        for result in all_results:
            if isinstance(result, dict) and result.get("errors") == 0:
                self.results[result["name"]] = result

        # 打印结果
        print("\n生成测试报告...")
        self._print_results()


async def main():
    tester = LLMPerformanceTester()
    await tester.run()


if __name__ == "__main__":
    asyncio.run(main())
