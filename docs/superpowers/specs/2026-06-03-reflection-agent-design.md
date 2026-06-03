# MyReflectionAgent 设计文档

**日期：** 2026-06-03  
**场景：** 通用（默认内置代码生成提示词，可通过 `custom_prompts` 切换至任意场景）

---

## 1. 背景

`hello_agents` 框架已提供 `ReflectionAgent` 基类，实现了「初始生成 → 反思 → 优化」的迭代循环。`MyReflectionAgent` 是面向**通用场景**的增强版本，适用于代码生成、文章写作、分析报告等任何需要迭代优化的任务，并解决基类的两个不足：

1. 提示词为极简通用文本，缺少结构化的评审维度
2. refine 阶段仅传入上一轮结果，丢失了历轮反馈上下文

默认内置**代码生成**提示词（含正确性/可读性/性能三维评审）；用户可通过 `custom_prompts` 参数传入任意场景的提示词替换默认值。

---

## 2. 架构

```
MyReflectionAgent(ReflectionAgent)
├── 模块级常量
│   ├── CODE_INITIAL_PROMPT    生成初始代码
│   ├── CODE_REFLECT_PROMPT    三维评审（正确性/可读性/性能）
│   └── CODE_REFINE_PROMPT     基于完整轨迹优化
│
├── __init__(name, llm, max_iterations, stop_keywords, custom_prompts, config)
│   └── 调用 super().__init__()，注入代码专用提示词
│
├── run(task)                  覆盖：refine 使用完整轨迹
├── _get_llm_response(prompt)  覆盖：使用 self.llm.invoke()（修复基类 think() 问题）
└── get_iteration_history()    新增：返回每轮 {round, code, feedback} 列表
```

---

## 3. 提示词设计

默认提示词针对代码生成场景；用户可通过 `custom_prompts={"initial": ..., "reflect": ..., "refine": ...}` 完整替换为其他场景（如文章写作、报告分析等）。

### CODE_INITIAL_PROMPT
```
请根据以下需求编写 Python 代码：

任务：{task}

要求：
- 代码逻辑正确，处理边界条件
- 命名清晰，结构易读
- 注意算法效率

请直接给出完整代码。
```

### CODE_REFLECT_PROMPT
```
请从以下三个维度评审代码：

任务：{task}
代码：{content}

## 评审维度
1. 正确性：逻辑是否正确？是否处理了边界条件和异常？
2. 可读性：命名是否清晰？结构是否易于理解？
3. 性能：算法复杂度是否合理？是否存在明显的性能瓶颈？

如果三个维度均满足要求，回复"无需改进"。
否则请列出具体问题和改进建议。
```

### CODE_REFINE_PROMPT
```
请根据以下完整的迭代历史改进代码：

任务：{task}

{trajectory}

请综合以上所有反馈，生成最终改进版本的完整代码。
```

---

## 4. 接口定义

```python
class MyReflectionAgent(ReflectionAgent):
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        max_iterations: int = 3,
        stop_keywords: Optional[List[str]] = None,  # 默认 ["无需改进"]
        custom_prompts: Optional[Dict[str, str]] = None,
        config: Optional[Config] = None,
    ): ...

    def run(self, task: str, **kwargs) -> str:
        """执行代码生成任务，返回最终代码字符串"""

    def get_iteration_history(self) -> List[Dict[str, Any]]:
        """返回 [{"round": int, "code": str, "feedback": str}, ...]"""
```

---

## 5. run() 核心逻辑变更

| 步骤 | 基类行为 | MyReflectionAgent 行为 |
|---|---|---|
| refine prompt 输入 | `last_attempt`（上一轮代码） | `trajectory`（完整历轮代码+反馈） |
| 早停检测 | `"无需改进"` 或 `"no need for improvement"` | 可配置 `stop_keywords` 列表 |
| 日志 | 基本打印 | 保留，增加轮次进度标识 |

---

## 6. 错误处理

- `llm.invoke()` 返回空字符串时继续迭代（不中断）
- `max_iterations` 达到上限时返回最后一次执行结果

---

## 7. 测试场景

| 测试 | 场景 | 验证点 |
|---|---|---|
| 默认提示词代码生成 | 代码生成 | run() 返回非空字符串 |
| 自定义提示词通用任务 | 文章/分析 | custom_prompts 生效，run() 正常返回 |
| 早停触发 | 任意 | 迭代次数 < max_iterations |
| 完整迭代 | 任意 | `get_iteration_history()` 长度 == max_iterations |
| 自定义 stop_keywords | 任意 | 自定义关键词触发早停 |
