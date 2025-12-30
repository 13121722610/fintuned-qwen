# Part 1/6: imports and exceptions
import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Optional
import asyncio
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 自定义异常：用于在遇到 503 等可重试错误时触发 tenacity 重试机制
class APIRetryError(Exception):
    """触发重试的内部异常"""
    pass

# Part 2/6: class declaration and __init__, session creator
class MassMedicalDataOptimizer:
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        batch_size: int = 200,
        max_workers: int = 1,        # 推荐: 1 或 2，越低越稳
        request_delay: float = 1.2   # 推荐: >=1.0 秒，避免被限流
    ):
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.request_delay = request_delay

        self.processed_count = 0
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        self.total_cost_rmb = 0.0
        self.batch_file_counter = 1

        # 用于简单估算（如需精确，请按真实价格调整）
        self.usd_to_rmb = 7.2

        # 日志初始化
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(f"optimization_log_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"初始化优化器：batch_size={batch_size}, max_workers={max_workers}, request_delay={request_delay}")

    async def _create_session(self) -> aiohttp.ClientSession:
        """
        创建并返回一个 aiohttp.ClientSession，可复用以避免频繁建立连接。
        """
        timeout = aiohttp.ClientTimeout(total=40)
        # connector.limit 控制底层 TCP 并发连接数，不是逻辑并发
        connector = aiohttp.TCPConnector(limit=30, ssl=False)
        return aiohttp.ClientSession(timeout=timeout, connector=connector)

# Part 3/6: optimize single item (with retry on 503)
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type(APIRetryError)
    )
    async def _optimize_single_async(self, session: aiohttp.ClientSession, item: Dict) -> Optional[Dict]:
        """
        使用 API 优化单条数据。
        返回:
          - 优化后的 item（包含 output 字段）
          - 原样 item（出错但非余额不足/503 的情况）
          - None（检测到余额不足，应当停止后续任务）
        """

        # 1) 提取用户问题文本（多种字段兼容）
        question_text = (
            item.get("user_query")
            or item.get("input")
            or item.get("instruction")
        )

        if not question_text and isinstance(item.get("conversations"), list):
            for msg in item["conversations"]:
                if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content"):
                    question_text = msg.get("content")
                    break

        if not question_text:
            self.logger.warning("无法提取问题文本，返回原始 item")
            return item

        # 2) 构造 prompt（你可以根据需要调整模板）
        template = """请以医疗专家的身份优化以下医疗回答：

问题：{question}
原回答：{current_answer}

请按照以下结构优化回答：
一、病情分析
1. 症状全面评估：
2. 可能疾病判断：

二、原因分析
1. 主要病因解析：
2. 鉴别诊断要点：

三、治病建议
1. 就医指导：
2. 治疗方案建议：

请确保回答专业、准确、易懂。
"""
        prompt = template.format(question=question_text, current_answer=item.get("output", ""))

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 800
        }

        # 3) 请求前等待（节流）
        await asyncio.sleep(self.request_delay)

        # 4) 发起请求
        async with session.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload
        ) as response:

            # 处理余额不足（402/403）——返回 None，调用方识别为余额耗尽
            if response.status in (402, 403):
                err_text = await response.text()
                self.logger.error(f"API 余额或权限问题（{response.status}）：{err_text}")
                return None

            # 503 -> 抛异常以触发 tenacity 的重试策略
            if response.status == 503:
                text = await response.text()
                self.logger.warning(f"503 Server Busy: {text[:200]}")
                raise APIRetryError("503 Service Busy")

            # 其他非 200 视为单条失败，返回原始 item，不中断全局流程
            if response.status != 200:
                err_body = await response.text()
                self.logger.warning(f"API 请求失败: HTTP {response.status} - {err_body}")
                return item

            # 解析响应
            result = await response.json()
            # 兼容性检查
            if not result or "choices" not in result or not result["choices"]:
                self.logger.warning("API 响应格式异常，返回原始 item")
                return item

            optimized_text = result["choices"][0]["message"]["content"]

            # 统计 tokens / 费用（若响应包含 usage）
            tokens_used = result.get("usage", {}).get("total_tokens", 0)
            self.total_tokens += tokens_used
            cost_usd = tokens_used * 0.14 / 1_000_000
            self.total_cost_usd += cost_usd
            self.total_cost_rmb += cost_usd * self.usd_to_rmb

            self.processed_count += 1
            if self.processed_count % 10 == 0:
                self.logger.info(f"已处理 {self.processed_count} 条，总 tokens: {self.total_tokens:,}, 费用: ${self.total_cost_usd:.6f}")

            # 返回新的 item（保留原始额外字段）
            new_item = {**item}
            new_item["output"] = optimized_text
            return new_item

# Part 4/6: batch processing and main runner
    async def optimize_batch(self, session: aiohttp.ClientSession, batch: List[Dict]) -> List[Dict]:
        """
        优化一个批次的数据 - 真正的并发版本
        """
        results: List[Dict] = []
        sem = asyncio.Semaphore(self.max_workers)
    
        async def worker(item):
            async with sem:
                try:
                    return await self._optimize_single_async(session, item)
                except Exception as e:
                    self.logger.error(f"单条处理异常（已降级为原始 item）: {e}")
                    return item
    
        # ✅ 正确的：同时创建所有任务，真正并发
        tasks = [worker(item) for item in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        for i, result in enumerate(batch_results):
            if result is None:
                # 余额不足
                results.append(None)
            elif isinstance(result, Exception):
                self.logger.error(f"任务异常，使用原始item: {result}")
                results.append(batch[i])  # 使用原始数据
            else:
                results.append(result)
    
        return results

    def process_all_data(self, input_file: str, output_file: str) -> List[Dict]:
        """
        主流程：加载数据文件 -> 按批次处理 -> 每批保存 -> 最终合并保存
        返回：所有成功优化（非 None）的条目列表
        """
        # 1) 加载数据（支持 JSON 数组或按行newline JSON）
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                raw = f.read()
        except Exception as e:
            raise Exception(f"读取输入文件失败: {e}")

        stripped = raw.lstrip()
        if not stripped:
            data: List[Dict] = []
        elif stripped[0] == "[":
            data = json.loads(raw)
        else:
            data = [json.loads(line) for line in raw.splitlines() if line.strip()]

        total = len(data)
        self.logger.info(f"开始优化，总数据量: {total} 条，批大小: {self.batch_size}")

        # 2) 创建事件循环并运行
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        all_optimized: List[Dict] = []

        async def runner():
            session = await self._create_session()
            try:
                total_batches = (total + self.batch_size - 1) // self.batch_size
                for i in range(0, total, self.batch_size):
                    batch = data[i:i + self.batch_size]
                    batch_idx = i // self.batch_size + 1
                    self.logger.info(f"处理批次 {batch_idx}/{total_batches} (大小 {len(batch)})")

                    batch_results = await self.optimize_batch(session, batch)

                    # 检查是否存在 None（余额不足）
                    none_count = sum(1 for item in batch_results if item is None)
                    if none_count > 0:
                        self.logger.error(f"检测到 {none_count} 条返回 None（可能余额不足或权限问题），停止后续批次")
                        # 过滤掉 None 并保存已有结果
                        batch_results = [it for it in batch_results if it is not None]
                        # 保存本批次已完成结果并退出
                        batch_filename = self._generate_batch_filename(output_file, batch_idx)
                        self._save_batch_result(batch_results, batch_filename)
                        self.logger.info(f"本批次部分结果已保存: {batch_filename}")
                        # 将结果添加到 all_optimized 并中断
                        all_optimized.extend(batch_results)
                        return

                    # 正常保存整个批次
                    batch_filename = self._generate_batch_filename(output_file, batch_idx)
                    self._save_batch_result(batch_results, batch_filename)
                    self.logger.info(f"✅ 批次 {batch_idx} 已保存: {batch_filename}")

                    all_optimized.extend(batch_results)

                    # 可选：小休眠以降低连续批次间的瞬时压力（可注释）
                    await asyncio.sleep(0.5)

            finally:
                await session.close()

        loop.run_until_complete(runner())
        loop.close()

        # 3) 保存最终合并结果并返回
        self._save_final_result(all_optimized, output_file)
        self.logger.info(f"全部处理完成：实际处理 {len(all_optimized)} 条，总 tokens {self.total_tokens:,}，美元 {self.total_cost_usd:.6f}，人民币 {self.total_cost_rmb:.4f}")
        return all_optimized

# Part 5/6: file helpers (generate filename, save batch, save final)
    def _generate_batch_filename(self, base_filename: str, batch_num: int) -> str:
        dir_name = os.path.dirname(base_filename) or "."
        name, ext = os.path.splitext(os.path.basename(base_filename))
        batch_name = f"{name}_{batch_num}{ext}"
        return os.path.join(dir_name, batch_name)

    def _save_batch_result(self, data: List[Dict], filename: str):
        try:
            os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"已保存批次文件: {filename} (条数: {len(data)})")
            self.batch_file_counter += 1
        except Exception as e:
            self.logger.error(f"保存批次文件失败: {e}")

    def _save_final_result(self, data: List[Dict], filename: str):
        try:
            os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
            wrapper = {
                "metadata": {
                    "total_count": len(data),
                    "optimized_at": datetime.now().isoformat(),
                    "model_used": self.model,
                    "total_tokens": self.total_tokens,
                    "total_cost_usd": self.total_cost_usd,
                    "total_cost_rmb": self.total_cost_rmb,
                    "processed_count": self.processed_count
                },
                "data": data
            }
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(wrapper, f, ensure_ascii=False, indent=2)
            self.logger.info(f"最终合并结果已保存到: {filename}")
        except Exception as e:
            self.logger.error(f"保存最终合并结果失败: {e}")

# Part 6/6: optional test helper and EOF
    def test_optimization(self):
        """
        测试单条样本优化（会真实调用 API）。
        若返回 None，表示余额或权限问题。
        """
        test_item = {
            "instruction": "你是一个医疗问诊专家，请根据用户的问题给出专业的回答",
            "input": "排卵日同房后小腹痛腰痛可能是什么原因？",
            "output": "可能是排卵期痛，也可能为附件炎或宫外孕等，需就医排查。"
        }

        # 直接在新事件循环中调用异步方法以便调试
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _run_test():
            session = await self._create_session()
            try:
                return await self._optimize_single_async(session, test_item)
            finally:
                await session.close()

        try:
            res = loop.run_until_complete(_run_test())
        finally:
            loop.close()

        if res is None:
            self.logger.error("测试返回 None，可能 API 余额不足或权限错误")
        else:
            self.logger.info("测试调用完成，结果长度: %d", len(res.get("output", "")))
        return res

# End of optimizer.py
