import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM, ToolRegistry
from hello_agents.tools import CalculatorTool, SearchTool
from react_agent import MyReActAgent

load_dotenv("../../.env")

llm = HelloAgentsLLM()
CUSTOM_PROMPT = """你是一个推理Agent。

## 可用工具
{tools}

## 格式（每次只输出一步）
Thought: 分析问题
Action: tool_name[input] 或 Finish[最终答案]

## 当前任务
Question: {question}

## 执行历史
{history}

开始："""

# ============================================================
# 测试1：不传 custom_prompt，暴露 MY_REACT_PROMPT 未定义 bug
# ============================================================
# print("=== 测试1: 默认提示词初始化（预期 NameError）===")
# registry1 = ToolRegistry()
# registry1.register_tool(CalculatorTool())
# try:
#     agent_default = MyReActAgent(
#         name="默认提示词Agent",
#         llm=llm,
#         tool_registry=registry1,
#     )
#     print("✅ 初始化成功（MY_REACT_PROMPT 已定义）")
# except NameError as e:
#     print(f"❌ NameError: {e}")

# # ============================================================
# # 测试2：传入 custom_prompt，验证基本初始化
# # ============================================================
# print("\n=== 测试2: 自定义提示词初始化 ===")
# registry2 = ToolRegistry()
# registry2.register_tool(CalculatorTool())
# agent = MyReActAgent(
#     name="ReAct计算助手",
#     llm=llm,
#     tool_registry=registry2,
#     max_steps=5,
#     custom_prompt=CUSTOM_PROMPT,
# )

# # ============================================================
# # 测试3：工具调用——让 Agent 用计算器解题
# # ============================================================
# print("\n=== 测试3: 工具调用（计算器）===")
# answer3 = agent.run("请计算 (25 + 75) * 3 的结果")
# print(f"\n最终答案: {answer3}\n")

# # ============================================================
# # 测试4：不需要工具，直接 Finish
# # ============================================================
# print("=== 测试4: 无工具直接回答 ===")
# answer4 = agent.run("请计算 (25 + 75) * 3 的结果")
# print(f"\n最终答案: {answer4}\n")

# # ============================================================
# # 测试5：max_steps 限制——给一个复杂问题观察步数上限
# # ============================================================
# print("=== 测试5: 最大步数限制（max_steps=2）===")
# registry3 = ToolRegistry()
# registry3.register_tool(CalculatorTool())
# agent_limited = MyReActAgent(
#     name="限步Agent",
#     llm=llm,
#     tool_registry=registry3,
#     max_steps=2,
#     custom_prompt=CUSTOM_PROMPT,
# )
# answer5 = agent_limited.run("先计算1+1，再计算2+2，再计算3+3，再计算4+4")
# print(f"\n最终答案: {answer5}\n")


registry4 = ToolRegistry()
registry4.register_tool(SearchTool())
agent_search = MyReActAgent(
    name="搜索Agent",
    llm=llm,
    tool_registry=registry4,
    max_steps=5,
    custom_prompt=CUSTOM_PROMPT,
)
answer6 = agent_search.run("请给我2款性价比最高的耳机")
print(f"\n最终答案: {answer6}\n")

# # ============================================================
# # 查看对话历史
# # ============================================================
# print("=== 对话历史 ===")
# print(f"agent 历史消息数: {len(agent.get_history())}")
