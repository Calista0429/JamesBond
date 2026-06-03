# my_reflection_agent.py
from typing import Optional, List, Dict, Any
from hello_agents import ReflectionAgent, HelloAgentsLLM, Config, Message

CODE_INITIAL_PROMPT = """请根据以下需求编写 Python 代码：

任务：{task}

要求：
- 代码逻辑正确，处理边界条件
- 命名清晰，结构易读
- 注意算法效率

请直接给出完整代码。"""

CODE_REFLECT_PROMPT = """请从以下三个维度评审代码：

任务：{task}
代码：
{content}

## 评审维度
1. 正确性：逻辑是否正确？是否处理了边界条件和异常？
2. 可读性：命名是否清晰？结构是否易于理解？
3. 性能：算法复杂度是否合理？是否存在明显的性能瓶颈？

如果三个维度均满足要求，回复"无需改进"。
否则请列出具体问题和改进建议。"""

CODE_REFINE_PROMPT = """请根据以下完整的迭代历史改进代码：

任务：{task}

{trajectory}

请综合以上所有反馈，生成最终改进版本的完整代码。"""

DEFAULT_CODE_PROMPTS = {
    "initial": CODE_INITIAL_PROMPT,
    "reflect": CODE_REFLECT_PROMPT,
    "refine":  CODE_REFINE_PROMPT,
}


class MyReflectionAgent(ReflectionAgent):
    """
    通用迭代优化 Agent，默认内置代码生成提示词。
    通过 custom_prompts 可切换至任意场景（文章写作、分析报告等）。
    与基类的核心差异：
      - refine 阶段使用完整轨迹（所有历轮代码 + 反馈），而非仅上一轮结果
      - stop_keywords 可配置
      - 新增 get_iteration_history() 供外部检查每轮输出
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        max_iterations: int = 3,
        stop_keywords: Optional[List[str]] = None,
        custom_prompts: Optional[Dict[str, str]] = None,
        config: Optional[Config] = None,
    ):
        prompts = custom_prompts if custom_prompts else DEFAULT_CODE_PROMPTS
        super().__init__(name, llm, config=config, max_iterations=max_iterations, custom_prompts=prompts)
        self.stop_keywords = stop_keywords if stop_keywords else ["无需改进", "no need for improvement"]
        print(f"✅ {name} 初始化完成，最大迭代次数: {max_iterations}")

    def run(self, task: str, **kwargs) -> str:
        """执行任务，返回最终优化结果。"""
        print(f"\n🤖 {self.name} 开始处理任务: {task}")

        # 重置记忆
        from hello_agents.agents.reflection_agent import Memory
        self.memory = Memory()

        # 1. 初始生成
        print("\n--- 正在进行初始尝试 ---")
        initial_prompt = self.prompts["initial"].format(task=task)
        initial_result = self._get_llm_response(initial_prompt, **kwargs)
        self.memory.add_record("execution", initial_result)

        # 2. 迭代：反思 → 检查早停 → 优化
        for i in range(self.max_iterations):
            print(f"\n--- 第 {i + 1}/{self.max_iterations} 轮迭代 ---")

            # 反思
            print("\n-> 正在进行反思...")
            last_result = self.memory.get_last_execution()
            reflect_prompt = self.prompts["reflect"].format(task=task, content=last_result)
            feedback = self._get_llm_response(reflect_prompt, **kwargs)
            self.memory.add_record("reflection", feedback)

            # 早停检查
            if any(kw in feedback for kw in self.stop_keywords):
                print("\n✅ 反思认为结果已无需改进，提前结束迭代。")
                break

            # 优化（传入完整轨迹）
            print("\n-> 正在进行优化...")
            trajectory = self.memory.get_trajectory()
            refine_prompt = self.prompts["refine"].format(task=task, trajectory=trajectory)
            refined_result = self._get_llm_response(refine_prompt, **kwargs)
            self.memory.add_record("execution", refined_result)

        final_result = self.memory.get_last_execution()
        print(f"\n--- 任务完成 ---\n最终结果:\n{final_result}")

        self.add_message(Message(task, "user"))
        self.add_message(Message(final_result, "assistant"))
        return final_result

    def _get_llm_response(self, prompt: str, **kwargs) -> str:
        """调用 LLM 获取完整响应。"""
        messages = [{"role": "user", "content": prompt}]
        return self.llm.invoke(messages, **kwargs) or ""

    def get_iteration_history(self) -> List[Dict[str, Any]]:
        """返回每轮迭代的代码和反馈，格式：[{"round": 1, "code": ..., "feedback": ...}]"""
        history = []
        executions = [r for r in self.memory.records if r["type"] == "execution"]
        reflections = [r for r in self.memory.records if r["type"] == "reflection"]
        for idx, execution in enumerate(executions):
            entry: Dict[str, Any] = {
                "round": idx + 1,
                "code": execution["content"],
                "feedback": reflections[idx]["content"] if idx < len(reflections) else None,
            }
            history.append(entry)
        return history
