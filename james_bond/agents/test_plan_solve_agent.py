import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM
from plan_solve_agent import MyPlanAndSolveAgent

load_dotenv("../../.env")
llm = HelloAgentsLLM()

# ============================================================
# 测试1：多步推理任务（默认提示词）
# ============================================================
print("=== 测试1: 多步推理（默认提示词）===")
agent = MyPlanAndSolveAgent(name="推理Agent", llm=llm)
result1 = agent.run("请分析：为什么冬天北半球比南半球更冷？")

plan = agent.get_plan()
history = agent.get_execution_history()
print(f"\n[计划步骤数]: {len(plan)}")
print(f"[执行历史条数]: {len(history)}")
for entry in history:
    print(f"  - 步骤: {entry['step'][:30]}... | 结果长度: {len(entry['result'])} 字符")

# ============================================================
# 测试2：数学/逻辑推理任务
# ============================================================
print("\n=== 测试2: 数学推理 ===")
agent2 = MyPlanAndSolveAgent(name="数学Agent", llm=llm)
result2 = agent2.run("一个水池，甲管单独注满需6小时，乙管单独注满需4小时，丙管单独排空需12小时。三管同时开，几小时注满水池？")
print(f"\n最终答案: {result2}")

# ============================================================
# 测试3：计划解析失败时的兜底处理
# ============================================================
print("\n=== 测试3: 计划解析失败兜底 ===")
BAD_PLANNER_PROMPT = "请回答：{question}\n（不要输出任何列表格式）"
agent3 = MyPlanAndSolveAgent(
    name="兜底测试Agent",
    llm=llm,
    custom_prompts={"planner": BAD_PLANNER_PROMPT},
)
result3 = agent3.run("1+1等于几")
print(f"兜底返回: {result3!r}")
print(f"计划为空: {agent3.get_plan() == []}")

# ============================================================
# 测试4：get_execution_history() 结构验证
# ============================================================
print("\n=== 测试4: get_execution_history() 结构验证 ===")
for entry in agent.get_execution_history():
    assert "step" in entry and "result" in entry, "结构错误"
    print(f"  ✅ step 存在: {bool(entry['step'])}, result 存在: {bool(entry['result'])}")

print("\n所有测试完成。")
