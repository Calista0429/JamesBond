import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM
from reflection_agent import MyReflectionAgent

load_dotenv("../../.env")
llm = HelloAgentsLLM()

# ============================================================
# 测试1：默认提示词 —— 代码生成任务
# ============================================================
print("=== 测试1: 代码生成（默认提示词）===")
agent = MyReflectionAgent(
    name="代码生成Agent",
    llm=llm,
    max_iterations=2,
)
result1 = agent.run("用 Python 实现一个二分查找函数，支持升序列表")
print(f"\n[迭代历史条数]: {len(agent.get_iteration_history())}\n")

# ============================================================
# 测试2：自定义提示词 —— 通用写作任务
# ============================================================
print("=== 测试2: 通用写作（自定义提示词）===")
WRITING_PROMPTS = {
    "initial": "请根据以下主题写一段简短介绍（100字以内）：\n\n主题：{task}",
    "reflect": (
        "请评审以下文字：\n\n主题：{task}\n内容：{content}\n\n"
        "评审维度：表达是否清晰、是否切题、字数是否合适。"
        "如果满足要求，回复\"无需改进\"，否则给出修改建议。"
    ),
    "refine": (
        "请根据以下反馈改进文字：\n\n主题：{task}\n\n{trajectory}\n\n请给出改进后的版本。"
    ),
}
agent2 = MyReflectionAgent(
    name="写作Agent",
    llm=llm,
    max_iterations=2,
    custom_prompts=WRITING_PROMPTS,
)
result2 = agent2.run("人工智能对教育的影响")
print(f"\n[迭代历史条数]: {len(agent2.get_iteration_history())}\n")

# ============================================================
# 测试3：自定义 stop_keywords
# ============================================================
print("=== 测试3: 自定义 stop_keywords ===")
agent3 = MyReflectionAgent(
    name="早停测试Agent",
    llm=llm,
    max_iterations=3,
    stop_keywords=["无需改进", "already good", "完美"],
)
result3 = agent3.run("用 Python 写一个计算阶乘的函数")
history3 = agent3.get_iteration_history()
print(f"\n[实际迭代轮数]: {len(history3)}（上限为3）")
print(f"[早停是否触发]: {'是' if len(history3) < 3 else '否（跑满了3轮）'}\n")

# ============================================================
# 测试4：get_iteration_history() 结构验证
# ============================================================
print("=== 测试4: get_iteration_history() 结构验证 ===")
for entry in history3:
    has_code = bool(entry.get("code"))
    has_feedback = entry.get("feedback") is not None
    print(f"  第{entry['round']}轮 — code存在: {has_code}, feedback存在: {has_feedback}")

print("\n所有测试完成。")
