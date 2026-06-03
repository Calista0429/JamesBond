# my_plan_solve_agent.py
import ast
from typing import Optional, List, Dict, Any
from hello_agents import PlanAndSolveAgent, HelloAgentsLLM, Config, Message

PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划,```python与```作为前后缀是必要的:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""

SYNTHESIZER_PROMPT_TEMPLATE = """
你是一位善于综合推理的专家。根据以下逐步执行的过程，给出一个完整、清晰的最终答案。

# 原始问题：
{question}

# 执行过程（计划 + 每步结果）：
{history}

请综合以上所有步骤的结论，给出针对原始问题的完整最终答案：
"""


class Planner:
    def __init__(self, llm_client: HelloAgentsLLM, prompt_template: Optional[str] = None):
        self.llm_client = llm_client
        self.prompt_template = prompt_template if prompt_template else PLANNER_PROMPT_TEMPLATE

    def plan(self, question: str, **kwargs) -> List[str]:
        prompt = self.prompt_template.format(question=question)
        messages = [{"role": "user", "content": prompt}]

        print("--- 正在生成计划 ---")
        response_text = self.llm_client.invoke(messages, **kwargs) or ""
        print(f"✅ 计划已生成:\n{response_text}")

        try:
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析计划时出错: {e}")
            print(f"原始响应: {response_text}")
            return []
        except Exception as e:
            print(f"❌ 解析计划时发生未知错误: {e}")
            return []


class Executor:
    def __init__(self, llm_client: HelloAgentsLLM, prompt_template: Optional[str] = None):
        self.llm_client = llm_client
        self.prompt_template = prompt_template if prompt_template else EXECUTOR_PROMPT_TEMPLATE

    def execute(self, question: str, plan: List[str], **kwargs) -> str:
        """逐步执行计划，返回包含所有步骤结果的历史字符串。"""
        if not plan:
            return ""

        history = ""
        print("\n--- 正在执行计划 ---")

        for i, step in enumerate(plan, 1):
            print(f"\n-> 正在执行步骤 {i}/{len(plan)}: {step}")
            prompt = self.prompt_template.format(
                question=question,
                plan=plan,
                history=history if history else "无",
                current_step=step,
            )
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.invoke(messages, **kwargs) or ""
            history += f"步骤 {i}: {step}\n结果: {response_text}\n\n"
            print(f"✅ 步骤 {i} 已完成，结果: {response_text}")

        return history


class MyPlanAndSolveAgent(PlanAndSolveAgent):
    """
    Plan & Solve Agent 增强版。
    在基类「规划 → 执行」流程基础上，新增综合总结步骤，
    将所有步骤的执行结果汇总为一个完整的最终答案。
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        custom_prompts: Optional[Dict[str, str]] = None,
        config: Optional[Config] = None,
    ):
        super().__init__(name, llm, config=config)
        planner_prompt = custom_prompts.get("planner") if custom_prompts else None
        executor_prompt = custom_prompts.get("executor") if custom_prompts else None
        synthesizer_prompt = custom_prompts.get("synthesizer") if custom_prompts else None

        self.planner = Planner(self.llm, planner_prompt)
        self.executor = Executor(self.llm, executor_prompt)
        self._synthesizer_prompt = synthesizer_prompt or SYNTHESIZER_PROMPT_TEMPLATE

        self._plan: List[str] = []
        self._execution_history: List[Dict[str, str]] = []
        print(f"✅ {name} 初始化完成")

    def run(self, question: str, **kwargs) -> str:
        print(f"\n🤖 {self.name} 开始处理问题: {question}")

        self._plan = []
        self._execution_history = []

        # 1. 规划
        plan = self.planner.plan(question, **kwargs)
        if not plan:
            final_answer = "无法生成有效的行动计划，任务终止。"
            print(f"\n--- 任务终止 ---\n{final_answer}")
            self.add_message(Message(question, "user"))
            self.add_message(Message(final_answer, "assistant"))
            return final_answer

        self._plan = plan

        # 2. 执行（返回完整历史字符串）
        history_str = self.executor.execute(question, plan, **kwargs)

        # 解析执行历史供 get_execution_history() 使用
        self._execution_history = self._parse_history(plan, history_str)

        # 3. 综合总结
        print("\n--- 正在综合所有步骤结果 ---")
        final_answer = self._synthesize(question, history_str, **kwargs)
        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")

        self.add_message(Message(question, "user"))
        self.add_message(Message(final_answer, "assistant"))
        return final_answer

    def _synthesize(self, question: str, history: str, **kwargs) -> str:
        prompt = self._synthesizer_prompt.format(question=question, history=history)
        messages = [{"role": "user", "content": prompt}]
        return self.llm.invoke(messages, **kwargs) or history

    def _parse_history(self, plan: List[str], history_str: str) -> List[Dict[str, str]]:
        result = []
        for i, step in enumerate(plan, 1):
            marker = f"步骤 {i}: {step}\n结果: "
            next_marker = f"步骤 {i + 1}:" if i < len(plan) else None
            start = history_str.find(marker)
            if start == -1:
                result.append({"step": step, "result": ""})
                continue
            start += len(marker)
            end = history_str.find(next_marker) if next_marker else len(history_str)
            result.append({"step": step, "result": history_str[start:end].strip()})
        return result

    def get_plan(self) -> List[str]:
        """返回最近一次生成的计划步骤列表。"""
        return list(self._plan)

    def get_execution_history(self) -> List[Dict[str, str]]:
        """返回每步执行结果，格式：[{"step": str, "result": str}, ...]"""
        return list(self._execution_history)
