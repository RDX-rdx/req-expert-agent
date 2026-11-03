"""
需求工程评审专家 - 基于 LangChain + DeepSeek + Gradio
功能：输入需求文本，自动输出质量评分、问题诊断、修改建议
"""

from __future__ import annotations

import html
import json
import os
import re
from typing import Any

import gradio as gr
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# 加载环境变量
load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    print("警告：未找到 DEEPSEEK_API_KEY，请复制 .env.example 为 .env 并填入密钥")


# 初始化大模型（DeepSeek 兼容 OpenAI 接口格式）
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=API_KEY or "missing-api-key",
    base_url="https://api.deepseek.com/v1",
    temperature=0.2,
)


def load_knowledge() -> str:
    """加载知识库文档作为评审参考。"""
    knowledge_dir = os.path.join(os.path.dirname(__file__), "knowledge")
    docs: list[str] = []
    for filename in sorted(os.listdir(knowledge_dir)):
        if filename.endswith(".txt"):
            with open(os.path.join(knowledge_dir, filename), "r", encoding="utf-8") as f:
                docs.append(f"【{filename}】\n" + f.read())
    return "\n\n".join(docs)


KNOWLEDGE_BASE = load_knowledge()


# 让模型返回结构化数据，界面层负责呈现，而不是直接渲染 Markdown 长文。
REVIEW_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一位资深的软件需求工程专家，精通 IEEE 830 软件需求规格标准。
你的任务是评审用户输入的需求文本，给出专业、客观、可操作的评审报告。

评审参考标准：
{knowledge_base}

只输出一个合法的 JSON 对象，不要输出 Markdown 代码围栏、前言、结语或其他解释。
JSON 必须严格符合以下结构，所有文本字段都使用简体中文：

{{
  "score": 4.0,
  "grade": "较差",
  "summary": "一句话说明总体质量、主要风险和打分理由",
  "dimensions": [
    {{"name": "完整性", "score": 4.0, "status": "需优化"}},
    {{"name": "无歧义性", "score": 3.0, "status": "较弱"}},
    {{"name": "可验证性", "score": 3.0, "status": "较弱"}},
    {{"name": "可追踪性", "score": 4.0, "status": "需优化"}}
  ],
  "issues": [
    {{
      "severity": "高",
      "category": "可验证性",
      "title": "简洁明确的问题标题",
      "location": "引用需求中的具体原文，不要编造",
      "reason": "解释为什么这是问题",
      "impact": "说明该问题对设计、开发、测试或验收的影响",
      "basis": ["IEEE 830 条款或反模式清单中的简短依据"],
      "original": "需要改写的原句",
      "rewrite": "可直接使用的完整改写需求",
      "acceptance": "可执行的验收标准或验证方法"
    }}
  ],
  "final_requirement": "整合全部修改后可直接复制的完整需求文本",
  "references": [
    "IEEE 830 或反模式清单中的具体条款说明"
  ]
}}

评审规则：
1. 如果需求写得很好，要如实给高分，不要硬找问题。
2. 问题必须具体，不能泛泛而谈。
3. severity 只能是“高”“中”“低”。
4. 高=不可验证、矛盾或会直接导致理解错误；中=明显模糊或缺少关键条件；低=表达或完整性的小问题。
5. 每个问题必须包含位置、原因、依据、改写建议和验收方法。
6. score 与 dimensions.score 均为 0 到 10 的数字，允许一位小数。
7. references 只列出报告中真实引用到的依据。
""",
        ),
        ("human", "请评审以下需求文本：\n\n{requirement_text}"),
    ]
)

review_chain = REVIEW_PROMPT | llm.bind(
    response_format={"type": "json_object"}
) | StrOutputParser()


class ReviewException(RuntimeError):
    """带有模型原始返回内容的评审异常。"""

    def __init__(self, message: str, raw: str = "") -> None:
        super().__init__(message)
        self.raw = raw


def _escape(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        return [
            f"{_text(key)}：{_text(item)}"
            for key, item in value.items()
            if _text(key) or _text(item)
        ]
    if isinstance(value, (list, tuple)):
        items: list[str] = []
        for item in value:
            if isinstance(item, dict):
                label = _text(item.get("name") or item.get("标准") or item.get("条款"))
                detail = _text(item.get("comment") or item.get("说明") or item.get("content"))
                combined = "：".join(part for part in (label, detail) if part)
                if combined:
                    items.append(combined)
            else:
                text = _text(item)
                if text:
                    items.append(text)
        return items
    return [_text(value)] if _text(value) else []


def _clamp_score(value: Any, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    return round(max(0.0, min(10.0, score)), 1)


def _normalize_severity(value: Any) -> str:
    text = _text(value).lower()
    if text in {"高", "严重", "high", "critical"} or "高" in text:
        return "高"
    if text in {"低", "low", "minor"} or "低" in text:
        return "低"
    return "中"


def _grade_for_score(score: float, model_grade: str = "") -> str:
    if model_grade in {"优秀", "良好", "需优化", "较差"}:
        return model_grade
    if score >= 9:
        return "优秀"
    if score >= 8:
        return "良好"
    if score >= 6:
        return "需优化"
    return "较差"


def extract_json_object(raw: str) -> dict[str, Any]:
    """从模型返回内容中提取 JSON，兼容少量代码围栏或前后说明。"""
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ReviewException("模型未返回有效的 JSON 评审结果。", raw=raw)
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ReviewException(f"解析评审结果失败：{exc}", raw=raw) from exc

    if not isinstance(parsed, dict):
        raise ReviewException("模型返回的评审结果不是 JSON 对象。", raw=raw)
    return parsed


def normalize_report(payload: dict[str, Any], requirement_text: str) -> dict[str, Any]:
    """统一模型字段，减少前端渲染对模型输出波动的依赖。"""
    score = _clamp_score(payload.get("score"))
    grade = _grade_for_score(score, _text(payload.get("grade")))

    raw_issues = payload.get("issues")
    if not isinstance(raw_issues, list):
        raw_issues = []

    issues: list[dict[str, Any]] = []
    for index, raw_issue in enumerate(raw_issues, start=1):
        if not isinstance(raw_issue, dict):
            continue
        issues.append(
            {
                "id": index,
                "severity": _normalize_severity(raw_issue.get("severity")),
                "category": _text(raw_issue.get("category"), "需求质量"),
                "title": _text(raw_issue.get("title"), f"问题 {index}"),
                "location": _text(raw_issue.get("location")),
                "reason": _text(raw_issue.get("reason")),
                "impact": _text(raw_issue.get("impact")),
                "basis": _as_text_list(raw_issue.get("basis")),
                "original": _text(raw_issue.get("original")),
                "rewrite": _text(raw_issue.get("rewrite")),
                "acceptance": _text(raw_issue.get("acceptance")),
            }
        )

    dimensions: list[dict[str, Any]] = []
    raw_dimensions = payload.get("dimensions")
    if isinstance(raw_dimensions, list):
        for item in raw_dimensions:
            if not isinstance(item, dict):
                continue
            name = _text(item.get("name"))
            if not name:
                continue
            dimensions.append(
                {
                    "name": name,
                    "score": _clamp_score(item.get("score")),
                    "status": _text(item.get("status")),
                }
            )

    counts = {
        "high": sum(1 for issue in issues if issue["severity"] == "高"),
        "medium": sum(1 for issue in issues if issue["severity"] == "中"),
        "low": sum(1 for issue in issues if issue["severity"] == "低"),
    }

    priority_issues = [
        issue["title"]
        for issue in sorted(
            issues,
            key=lambda issue: {"高": 0, "中": 1, "低": 2}[issue["severity"]],
        )[:3]
    ]

    final_requirement = _text(payload.get("final_requirement"))
    if not final_requirement:
        final_requirement = "\n\n".join(
            issue["rewrite"] for issue in issues if issue["rewrite"]
        )

    return {
        "score": score,
        "grade": grade,
        "summary": _text(payload.get("summary"), "未提供总体评价。"),
        "dimensions": dimensions,
        "issues": issues,
        "counts": counts,
        "priority_issues": priority_issues,
        "final_requirement": final_requirement or requirement_text,
        "references": _as_text_list(payload.get("references")),
        "source_requirement": requirement_text,
    }


def review_requirement(requirement_text: str) -> dict[str, Any]:
    """调用模型评审需求，返回结构化报告。"""
    if not requirement_text or not requirement_text.strip():
        raise ReviewException("请输入需要评审的需求文本。")
    if not API_KEY:
        raise ReviewException(
            "未配置 DEEPSEEK_API_KEY。请复制 .env.example 为 .env，并填写有效的 API Key。"
        )

    try:
        result = review_chain.invoke(
            {
                "knowledge_base": KNOWLEDGE_BASE,
                "requirement_text": requirement_text,
            }
        )
    except Exception as exc:
        raise ReviewException(
            f"调用大模型失败：{exc}\n请检查 API Key 和网络连接。"
        ) from exc

    payload = extract_json_object(result)
    return normalize_report(payload, requirement_text)


EXAMPLES = [
    (
        "模糊需求",
        "登录时限与主观描述",
        "用户必须能在3秒内完成登录操作。系统界面要友好，操作要简单方便。",
    ),
    (
        "可量化需求",
        "网络、设备与响应时间",
        "在5G网络环境下，使用近3年发布的主流安卓机型，并发100人同时登录时，"
        '登录接口P95响应时间不超过3秒，超时后自动重试1次，重试失败则提示用户'
        '"网络异常，请稍后重试"。',
    ),
    (
        "复杂条件需求",
        "并发、体验与数据量",
        "系统应该快速响应，界面要美观大方，用户体验要好，支持大数据量处理。",
    ),
]


GLOBAL_CSS = """
:root {
    --ink: #142033;
    --muted: #667085;
    --faint: #98a2b3;
    --line: #e4eaf2;
    --line-strong: #d4dce8;
    --canvas: #f3f6fb;
    --card: #ffffff;
    --blue: #2563eb;
    --blue-dark: #1d4ed8;
    --blue-soft: #eff5ff;
    --green: #16865b;
    --green-soft: #ecfdf3;
    --amber: #b45309;
    --amber-soft: #fff7e6;
    --red: #c93b43;
    --red-soft: #fff1f2;
}

body {
    background: var(--canvas);
    color: var(--ink);
}

.gradio-container {
    max-width: none !important;
    padding: 0 !important;
    background: transparent !important;
    font-family: "PingFang SC", "Microsoft YaHei UI", "Microsoft YaHei",
        "Noto Sans CJK SC", sans-serif !important;
}

#app-shell {
    width: min(1240px, calc(100% - 56px));
    margin: 0 auto;
    padding: 28px 0 44px;
    gap: 18px !important;
}

.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
    padding: 6px 2px 2px;
}

.brand-lockup {
    display: flex;
    align-items: center;
    gap: 14px;
}

.brand-mark {
    width: 48px;
    height: 48px;
    display: grid;
    place-items: center;
    border-radius: 14px;
    color: white;
    background: linear-gradient(145deg, #2f6fed, #1746b8);
    box-shadow: 0 12px 26px rgba(37, 99, 235, 0.2);
}

.brand-mark svg {
    width: 25px;
    height: 25px;
}

.brand-copy h1 {
    margin: 0;
    font-size: 28px;
    line-height: 1.2;
    letter-spacing: -0.02em;
    font-weight: 700;
    color: #111b2e;
}

.brand-copy p {
    margin: 7px 0 0;
    color: var(--muted);
    font-size: 14px;
}

.header-meta {
    display: flex;
    align-items: center;
    gap: 8px;
}

.header-meta span {
    padding: 7px 10px;
    border: 1px solid var(--line);
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.72);
    color: #536176;
    font-size: 12px;
    font-weight: 600;
}

.stepbar {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    min-height: 54px;
    padding: 10px 18px;
    margin: 2px 0 4px;
    border: 1px solid var(--line);
    border-radius: 14px;
    background: rgba(255, 255, 255, 0.78);
    box-shadow: 0 8px 26px rgba(31, 49, 80, 0.04);
}

.step {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: #8994a7;
    font-size: 13px;
    font-weight: 600;
}

.step-index {
    width: 24px;
    height: 24px;
    display: grid;
    place-items: center;
    border: 1px solid #d8e0eb;
    border-radius: 50%;
    background: white;
    font-size: 11px;
}

.step.is-current {
    color: var(--blue);
}

.step.is-current .step-index {
    color: white;
    border-color: var(--blue);
    background: var(--blue);
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
}

.step.is-done {
    color: var(--green);
}

.step.is-done .step-index {
    color: white;
    border-color: var(--green);
    background: var(--green);
}

.step-line {
    width: 54px;
    height: 1px;
    background: #dfe6ef;
}

#input-workspace {
    display: flex !important;
    gap: 20px !important;
    align-items: stretch !important;
}

#input-panel,
#examples-panel {
    padding: 24px !important;
    border: 1px solid var(--line) !important;
    border-radius: 16px !important;
    background: var(--card) !important;
    box-shadow: 0 14px 36px rgba(34, 54, 86, 0.06) !important;
    gap: 0 !important;
}

.section-heading {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 16px;
}

.heading-group {
    display: flex;
    align-items: flex-start;
    gap: 12px;
}

.heading-icon {
    width: 34px;
    height: 34px;
    display: grid;
    place-items: center;
    flex: 0 0 auto;
    border-radius: 10px;
    color: var(--blue);
    background: var(--blue-soft);
}

.heading-icon svg {
    width: 18px;
    height: 18px;
}

.heading-title {
    margin: 0;
    color: var(--ink);
    font-size: 17px;
    line-height: 1.35;
    font-weight: 700;
}

.heading-caption {
    margin: 5px 0 0;
    color: var(--muted);
    font-size: 13px;
    line-height: 1.5;
}

.soft-badge {
    flex: 0 0 auto;
    padding: 6px 9px;
    border-radius: 999px;
    color: #31527d;
    background: #f2f7fd;
    font-size: 11px;
    font-weight: 700;
}

#requirement-textarea {
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

#requirement-textarea textarea {
    min-height: 292px !important;
    padding: 18px !important;
    border: 1px solid var(--line-strong) !important;
    border-radius: 12px !important;
    background: #fbfcfe !important;
    color: #1b2738 !important;
    font-size: 15px !important;
    line-height: 1.75 !important;
    box-shadow: inset 0 1px 2px rgba(20, 32, 51, 0.02) !important;
    resize: vertical !important;
}

#requirement-textarea textarea:focus {
    border-color: #84aef8 !important;
    background: white !important;
    box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.08) !important;
}

#input-action-row {
    align-items: center !important;
    gap: 10px !important;
    margin-top: 12px;
}

.char-counter {
    color: var(--muted);
    font-size: 12px;
}

.char-counter strong {
    color: #344054;
    font-variant-numeric: tabular-nums;
}

#clear-example-row {
    display: flex !important;
    justify-content: flex-end;
    min-width: 76px;
}

#clear-btn {
    min-height: 42px !important;
    border: 1px solid var(--line-strong) !important;
    border-radius: 9px !important;
    background: white !important;
    color: #526071 !important;
    box-shadow: none !important;
    font-size: 13px !important;
}

#review-btn {
    min-width: 196px !important;
    min-height: 46px !important;
    border: 0 !important;
    border-radius: 10px !important;
    color: white !important;
    background: var(--blue) !important;
    box-shadow: 0 10px 24px rgba(37, 99, 235, 0.2) !important;
    font-size: 14px !important;
    font-weight: 700 !important;
}

#review-btn:hover {
    background: var(--blue-dark) !important;
}

#review-btn:disabled {
    color: rgba(255, 255, 255, 0.86) !important;
    background: #6f99ee !important;
    box-shadow: none !important;
}

.example-stack {
    display: grid;
    gap: 10px;
    margin: 4px 0 22px;
}

.example-button {
    width: 100% !important;
    margin-bottom: 9px !important;
    border: 0 !important;
    border-radius: 10px !important;
    background: transparent !important;
    box-shadow: none !important;
}

.example-button button,
button.example-button,
#examples-panel .example-button {
    width: 100% !important;
    min-height: 55px !important;
    justify-content: flex-start !important;
    padding: 10px 14px !important;
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
    background: #f8fafc !important;
    color: #334155 !important;
    text-align: left !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    line-height: 1.45 !important;
    box-shadow: none !important;
}

.example-button button:hover,
button.example-button:hover,
#examples-panel .example-button:hover {
    border-color: #b9cdf4 !important;
    background: var(--blue-soft) !important;
    color: #1e4fae !important;
}

.guide-box {
    padding: 16px;
    border: 1px solid var(--line);
    border-radius: 12px;
    background: #fbfcfe;
}

.guide-title {
    margin: 0 0 11px;
    color: #344054;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.04em;
}

.guide-list {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.guide-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 9px;
    border-radius: 8px;
    color: #40516a;
    background: white;
    border: 1px solid #e8edf4;
    font-size: 12px;
    font-weight: 600;
}

.guide-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--blue);
}

.guide-note {
    margin: 13px 0 0;
    color: var(--muted);
    font-size: 12px;
    line-height: 1.65;
}

.guide-outcomes {
    display: grid;
    gap: 8px;
    margin-top: 16px;
    padding-top: 14px;
    border-top: 1px solid var(--line);
}

.guide-outcome {
    display: grid;
    grid-template-columns: 25px minmax(0, 1fr);
    gap: 9px;
    align-items: start;
}

.guide-outcome span {
    color: #96a2b4;
    font-size: 10px;
    font-weight: 750;
    line-height: 1.8;
}

.guide-outcome p {
    margin: 0;
    color: #4f5f73;
    font-size: 12px;
    line-height: 1.65;
}

#result-workspace {
    gap: 14px !important;
    margin-top: 2px;
}

#result-toolbar {
    align-items: center !important;
    gap: 20px !important;
}

.result-toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    padding: 2px 2px 0;
}

.result-title {
    margin: 0;
    color: #111b2e;
    font-size: 20px;
    font-weight: 700;
}

.result-subtitle {
    margin: 5px 0 0;
    color: var(--muted);
    font-size: 12px;
}

#back-btn {
    flex: 0 0 112px !important;
    width: 112px !important;
    max-width: 112px !important;
    min-width: 108px !important;
    min-height: 38px !important;
    border: 1px solid var(--line-strong) !important;
    border-radius: 9px !important;
    background: white !important;
    color: #475467 !important;
    box-shadow: none !important;
    font-size: 12px !important;
}

#report-root {
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
}

.report-summary {
    display: grid;
    grid-template-columns: minmax(380px, 1.4fr) minmax(310px, 1fr);
    gap: 22px;
    padding: 24px;
    border: 1px solid var(--line);
    border-radius: 16px;
    background: white;
    box-shadow: 0 14px 36px rgba(34, 54, 86, 0.06);
}

.score-overview {
    display: flex;
    align-items: center;
    gap: 18px;
}

.score-ring {
    --score: 40%;
    --ring: #d97706;
    width: 104px;
    height: 104px;
    display: grid;
    place-items: center;
    flex: 0 0 auto;
    position: relative;
    border-radius: 50%;
    background: conic-gradient(var(--ring) var(--score), #edf1f6 0);
}

.score-ring::after {
    content: "";
    position: absolute;
    inset: 8px;
    border-radius: 50%;
    background: white;
}

.score-value {
    position: relative;
    z-index: 1;
    display: flex;
    align-items: baseline;
    gap: 2px;
}

.score-value strong {
    color: #172033;
    font-size: 30px;
    line-height: 1;
    font-weight: 750;
    font-variant-numeric: tabular-nums;
}

.score-value span {
    color: var(--faint);
    font-size: 12px;
    font-weight: 600;
}

.grade-line {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}

.grade-pill,
.severity-pill,
.category-pill,
.risk-chip,
.basis-chip {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    font-weight: 700;
}

.grade-pill {
    padding: 6px 10px;
    color: #9a4c00;
    background: var(--amber-soft);
    font-size: 12px;
}

.grade-pill.is-high {
    color: var(--red);
    background: var(--red-soft);
}

.grade-pill.is-good {
    color: var(--green);
    background: var(--green-soft);
}

.summary-copy {
    max-width: 580px;
    margin: 0;
    color: #526071;
    font-size: 15px;
    line-height: 1.75;
}

.summary-side {
    display: grid;
    gap: 12px;
    align-content: start;
}

.risk-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.risk-chip {
    gap: 6px;
    padding: 7px 10px;
    border: 1px solid var(--line);
    background: #fafbfd;
    color: #526071;
    font-size: 12.5px;
}

.risk-chip::before {
    content: "";
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #98a2b3;
}

.risk-chip.high::before {
    background: var(--red);
}

.risk-chip.medium::before {
    background: #d97706;
}

.risk-chip.low::before {
    background: #3b82f6;
}

.dimension-list {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 9px 14px;
}

.dimension-item {
    display: grid;
    gap: 6px;
}

.dimension-meta {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    color: #536176;
    font-size: 12px;
    font-weight: 650;
}

.dimension-track {
    height: 5px;
    overflow: hidden;
    border-radius: 999px;
    background: #edf1f6;
}

.dimension-fill {
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, #3b82f6, #2563eb);
}

.priority-strip {
    padding: 12px 14px;
    border-radius: 10px;
    border: 1px solid #f1dfbd;
    background: #fffaf0;
    color: #7c4d0d;
    font-size: 13px;
    line-height: 1.65;
}

.priority-strip strong {
    color: #694008;
}

.report-grid {
    display: grid;
    grid-template-columns: 320px minmax(0, 1fr);
    gap: 16px;
    margin-top: 16px;
    align-items: start;
}

.report-sidebar {
    position: sticky;
    top: 18px;
    display: grid;
    gap: 12px;
}

.sidebar-card {
    border: 1px solid var(--line);
    border-radius: 14px;
    background: white;
    box-shadow: 0 10px 28px rgba(34, 54, 86, 0.045);
    overflow: hidden;
}

.sidebar-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 15px 16px 12px;
    color: #344054;
    font-size: 13px;
    font-weight: 750;
    letter-spacing: 0.03em;
}

.sidebar-heading span {
    color: var(--faint);
    font-weight: 600;
    letter-spacing: 0;
}

.issue-nav-list {
    display: grid;
    gap: 4px;
    padding: 0 8px 9px;
}

.issue-nav-item {
    width: 100%;
    display: grid;
    grid-template-columns: 24px minmax(0, 1fr) auto;
    align-items: start;
    gap: 10px;
    padding: 12px 10px;
    border: 0;
    border-radius: 9px;
    background: transparent;
    color: #344054;
    text-align: left;
    cursor: pointer;
    transition: background 160ms ease, color 160ms ease;
}

.issue-nav-item:hover {
    background: #f6f8fb;
}

.issue-nav-item.is-active {
    background: var(--blue-soft);
    color: #1749a8;
}

.nav-index {
    width: 26px;
    height: 26px;
    display: grid;
    place-items: center;
    border-radius: 7px;
    background: #f1f4f8;
    color: #667085;
    font-size: 11px;
    font-weight: 750;
}

.issue-nav-item.is-active .nav-index {
    color: white;
    background: var(--blue);
}

.nav-title {
    display: -webkit-box;
    overflow: hidden;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: 2;
    font-size: 14px;
    line-height: 1.5;
    font-weight: 650;
}

.nav-severity {
    width: 7px;
    height: 7px;
    margin-top: 7px;
    border-radius: 50%;
    background: #d97706;
}

.nav-severity.high {
    background: var(--red);
}

.nav-severity.low {
    background: #3b82f6;
}

.source-box {
    padding: 0;
}

.source-box summary {
    list-style: none;
    padding: 14px 16px;
    color: #344054;
    font-size: 13px;
    font-weight: 750;
    cursor: pointer;
}

.source-box summary::-webkit-details-marker {
    display: none;
}

.source-box summary::after {
    content: "＋";
    float: right;
    color: #98a2b3;
    font-weight: 500;
}

.source-box[open] summary::after {
    content: "－";
}

.source-text {
    max-height: 280px;
    overflow: auto;
    margin: 0;
    padding: 0 16px 16px;
    color: #526071;
    font-size: 14px;
    line-height: 1.75;
    white-space: pre-wrap;
}

.issue-panel {
    display: none;
    padding: 24px;
    border: 1px solid var(--line);
    border-radius: 16px;
    background: white;
    box-shadow: 0 14px 36px rgba(34, 54, 86, 0.06);
}

.issue-panel.is-active {
    display: block;
}

.issue-kicker {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
}

.issue-number {
    color: #98a2b3;
    font-size: 12px;
    font-weight: 750;
    letter-spacing: 0.08em;
}

.severity-pill {
    padding: 5px 9px;
    color: #9a4c00;
    background: var(--amber-soft);
    font-size: 12px;
}

.severity-pill.high {
    color: var(--red);
    background: var(--red-soft);
}

.severity-pill.low {
    color: #2563eb;
    background: var(--blue-soft);
}

.category-pill {
    padding: 5px 9px;
    color: #516173;
    background: #f2f5f9;
    font-size: 12px;
}

.issue-title {
    margin: 0 0 20px;
    color: #182335;
    font-size: 22px;
    line-height: 1.45;
    font-weight: 720;
    letter-spacing: -0.015em;
}

.diagnosis-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
}

.info-block {
    padding: 16px;
    border: 1px solid var(--line);
    border-radius: 11px;
    background: #fbfcfe;
}

.info-block.full {
    grid-column: 1 / -1;
}

.info-label {
    display: flex;
    align-items: center;
    gap: 7px;
    margin-bottom: 9px;
    color: #667085;
    font-size: 12px;
    font-weight: 750;
    letter-spacing: 0.04em;
}

.info-label::before {
    content: "";
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #6d8ec9;
}

.info-text {
    margin: 0;
    color: #344054;
    font-size: 15px;
    line-height: 1.75;
    white-space: pre-wrap;
}

.location-text {
    color: #253754;
    font-weight: 600;
}

.basis-list {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
}

.basis-chip {
    padding: 6px 9px;
    color: #4b5e78;
    background: #f1f5fa;
    font-size: 12.5px;
    line-height: 1.5;
}

.rewrite-card {
    margin-top: 18px;
    padding: 18px;
    border: 1px solid #dce7f8;
    border-radius: 13px;
    background: #f8fbff;
}

.rewrite-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    margin-bottom: 13px;
}

.rewrite-heading h3,
.final-heading h3 {
    margin: 0;
    color: #233a5c;
    font-size: 16px;
    font-weight: 750;
}

.copy-btn {
    min-height: 32px;
    padding: 0 11px;
    border: 1px solid #cbd9ef;
    border-radius: 8px;
    background: white;
    color: #31527d;
    font-size: 12px;
    font-weight: 700;
    cursor: pointer;
}

.copy-btn:hover {
    border-color: #8fb1e7;
    color: #1749a8;
    background: #f7faff;
}

.diff-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
}

.diff-pane {
    min-width: 0;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #e7ecf2;
    background: white;
}

.diff-head {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 9px 12px;
    border-bottom: 1px solid #edf0f4;
    color: #667085;
    font-size: 11.5px;
    font-weight: 750;
    letter-spacing: 0.05em;
}

.diff-head::before {
    content: "";
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #98a2b3;
}

.diff-pane.revised {
    border-color: #cfe6da;
}

.diff-pane.revised .diff-head {
    color: #267052;
    background: var(--green-soft);
    border-bottom-color: #d9eee3;
}

.diff-pane.revised .diff-head::before {
    background: var(--green);
}

.diff-text {
    margin: 0;
    padding: 14px;
    color: #344054;
    font-size: 15px;
    line-height: 1.78;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}

.diff-pane.revised .diff-text {
    color: #174b39;
    background: #fbfffd;
}

.acceptance-note {
    display: flex;
    gap: 10px;
    margin-top: 12px;
    padding: 12px 13px;
    border-radius: 9px;
    color: #26644d;
    background: #effaf4;
    font-size: 13.5px;
    line-height: 1.65;
}

.acceptance-note strong {
    flex: 0 0 auto;
}

.final-section {
    margin-top: 16px;
    padding: 20px;
    border: 1px solid var(--line);
    border-radius: 14px;
    background: white;
    box-shadow: 0 10px 28px rgba(34, 54, 86, 0.045);
}

.final-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    margin-bottom: 13px;
}

.final-text {
    margin: 0;
    padding: 16px;
    border: 1px solid #dde7f4;
    border-radius: 10px;
    background: #f8fafc;
    color: #243247;
    font-family: "Cascadia Code", "Microsoft YaHei UI", monospace;
    font-size: 14px;
    line-height: 1.78;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}

.reference-box {
    margin-top: 12px;
    padding: 0 18px;
    border: 1px solid var(--line);
    border-radius: 12px;
    background: white;
}

.reference-box summary {
    padding: 15px 0;
    color: #475467;
    font-size: 13px;
    font-weight: 700;
    cursor: pointer;
}

.reference-box ul {
    margin: 0;
    padding: 0 0 16px 19px;
    color: #667085;
    font-size: 13px;
    line-height: 1.72;
}

.empty-issues {
    padding: 34px;
    border: 1px solid #cfe6da;
    border-radius: 16px;
    background: var(--green-soft);
    color: #216347;
}

.empty-issues h2 {
    margin: 0 0 8px;
    font-size: 20px;
}

.empty-issues p {
    margin: 0;
    line-height: 1.7;
}

.loading-panel,
.error-panel {
    padding: 32px;
    border: 1px solid var(--line);
    border-radius: 16px;
    background: white;
    box-shadow: 0 14px 36px rgba(34, 54, 86, 0.06);
}

.loading-head {
    display: flex;
    align-items: center;
    gap: 13px;
}

.loading-spinner {
    width: 28px;
    height: 28px;
    border: 3px solid #dce7f8;
    border-top-color: var(--blue);
    border-radius: 50%;
    animation: spin 0.9s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

.loading-panel h2,
.error-panel h2 {
    margin: 0;
    color: #1b2738;
    font-size: 18px;
}

.loading-panel p,
.error-panel p {
    margin: 10px 0 0;
    color: var(--muted);
    font-size: 13px;
    line-height: 1.7;
}

.skeleton-list {
    display: grid;
    gap: 10px;
    margin-top: 24px;
}

.skeleton-row {
    height: 12px;
    border-radius: 999px;
    background: linear-gradient(90deg, #f1f4f8 20%, #e7ecf3 50%, #f1f4f8 80%);
    background-size: 200% 100%;
    animation: shimmer 1.4s infinite;
}

.skeleton-row:nth-child(2) { width: 84%; }
.skeleton-row:nth-child(3) { width: 68%; }

@keyframes shimmer {
    to { background-position: -200% 0; }
}

.error-panel {
    border-color: #f0d2d5;
    background: #fffafb;
}

.error-detail {
    margin-top: 14px;
    padding: 13px;
    border: 1px solid #f1dfe1;
    border-radius: 9px;
    background: white;
    color: #6b4b50;
    font-size: 12px;
    line-height: 1.7;
    white-space: pre-wrap;
}

.copy-source {
    display: none;
}

.app-footer {
    padding: 10px 0 0;
    color: #98a2b3;
    font-size: 11px;
    line-height: 1.6;
    text-align: center;
}

footer {
    display: none !important;
}

@media (max-width: 1100px) {
    #app-shell {
        width: calc(100% - 32px);
    }

    .report-summary {
        grid-template-columns: 1fr;
    }

    .report-grid {
        grid-template-columns: 250px minmax(0, 1fr);
    }
}
"""


HEADER_HTML = """
<div class="app-header">
  <div class="brand-lockup">
    <div class="brand-mark" aria-hidden="true">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"
           stroke-linecap="round" stroke-linejoin="round">
        <path d="M9 5H7a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-2"/>
        <rect x="9" y="3" width="6" height="4" rx="1.5"/>
        <path d="m9 14 2 2 4-5"/>
      </svg>
    </div>
    <div class="brand-copy">
      <h1>需求工程评审专家</h1>
      <p>基于大模型的智能需求质量评审工具</p>
    </div>
  </div>
  <div class="header-meta">
    <span>IEEE 830</span>
    <span>结构化评审</span>
  </div>
</div>
"""


INPUT_HEADER_HTML = """
<div class="section-heading">
  <div class="heading-group">
    <div class="heading-icon" aria-hidden="true">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"
           stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 20h9"/>
        <path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/>
      </svg>
    </div>
    <div>
      <h2 class="heading-title">输入需求文本</h2>
      <p class="heading-caption">粘贴单条需求或完整需求文档，系统将按质量标准逐项诊断。</p>
    </div>
  </div>
  <span class="soft-badge">支持批量文本</span>
</div>
"""


EXAMPLES_HEADER_HTML = """
<div class="section-heading">
  <div class="heading-group">
    <div class="heading-icon" aria-hidden="true">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"
           stroke-linecap="round" stroke-linejoin="round">
        <path d="M9 18h6"/>
        <path d="M10 22h4"/>
        <path d="M8.4 14.7A7 7 0 1 1 15.6 14.7C14.6 15.5 14 16.6 14 18h-4c0-1.4-.6-2.5-1.6-3.3Z"/>
      </svg>
    </div>
    <div>
      <h2 class="heading-title">示例与检查范围</h2>
      <p class="heading-caption">选择示例快速体验，或直接输入自己的需求。</p>
    </div>
  </div>
</div>
"""


GUIDE_HTML = """
<div class="guide-box">
  <p class="guide-title">核心质量检查</p>
  <div class="guide-list">
    <span class="guide-item"><i class="guide-dot"></i>完整性</span>
    <span class="guide-item"><i class="guide-dot"></i>无歧义性</span>
    <span class="guide-item"><i class="guide-dot"></i>可验证性</span>
    <span class="guide-item"><i class="guide-dot"></i>可追踪性</span>
  </div>
  <p class="guide-note">结果会给出总体评分、问题优先级、原文对照改写和可执行验收标准，便于直接回填到需求文档。</p>
  <div class="guide-outcomes">
    <div class="guide-outcome"><span>01</span><p>质量总分、风险等级与维度评分</p></div>
    <div class="guide-outcome"><span>02</span><p>按优先级排列的问题导航与定位说明</p></div>
    <div class="guide-outcome"><span>03</span><p>可直接复制的改写需求与验收标准</p></div>
  </div>
</div>
"""


RESULT_HEADER_HTML = """
<div>
  <h2 class="result-title">评审工作台</h2>
  <p class="result-subtitle">从左侧选择问题，右侧查看诊断、标准依据和修改结果。</p>
</div>
"""


FOOTER_HTML = """
<div class="app-footer">
  支持单条需求评审，也支持粘贴整段需求文档进行批量评审
</div>
"""


LOADING_HTML = """
<div class="loading-panel">
  <div class="loading-head">
    <div class="loading-spinner" aria-hidden="true"></div>
    <h2>正在分析需求质量</h2>
  </div>
  <p>正在检查完整性、歧义性、可验证性和可追踪性，并生成可操作的修改建议。</p>
  <div class="skeleton-list" aria-hidden="true">
    <div class="skeleton-row"></div>
    <div class="skeleton-row"></div>
    <div class="skeleton-row"></div>
  </div>
</div>
"""


REPORT_JS = """
const activateIssue = (issueId) => {
  element.querySelectorAll(".issue-panel").forEach((panel) => {
    panel.classList.toggle("is-active", panel.dataset.issueId === issueId);
  });
  element.querySelectorAll(".issue-nav-item").forEach((item) => {
    const active = item.dataset.issueNav === issueId;
    item.classList.toggle("is-active", active);
    item.setAttribute("aria-selected", active ? "true" : "false");
  });
};

const fallbackCopy = (text) => {
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.focus();
  textarea.select();
  document.execCommand("copy");
  textarea.remove();
};

element.addEventListener("click", async (event) => {
  const navButton = event.target.closest("[data-issue-nav]");
  if (navButton) {
    activateIssue(navButton.dataset.issueNav);
    return;
  }

  const copyButton = event.target.closest("[data-copy]");
  if (!copyButton) return;

  const sourceId = copyButton.dataset.copy;
  const source = element.querySelector(`[data-copy-source="${sourceId}"]`);
  if (!source) return;

  const text = source.value !== undefined ? source.value : source.textContent;
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
    } else {
      fallbackCopy(text);
    }
    const originalLabel = copyButton.textContent;
    copyButton.textContent = "已复制";
    setTimeout(() => { copyButton.textContent = originalLabel; }, 1400);
  } catch (error) {
    const originalLabel = copyButton.textContent;
    copyButton.textContent = "复制失败";
    setTimeout(() => { copyButton.textContent = originalLabel; }, 1400);
  }
});
"""


def render_steps(active_step: int) -> str:
    """渲染桌面端步骤条。"""
    labels = ["输入需求", "智能评审", "查看结果"]
    parts: list[str] = []
    for index, label in enumerate(labels, start=1):
        state = "is-done" if index < active_step else "is-current" if index == active_step else ""
        parts.append(
            f'<div class="step {state}"><span class="step-index">{index}</span>'
            f"<span>{label}</span></div>"
        )
        if index < len(labels):
            parts.append('<span class="step-line" aria-hidden="true"></span>')
    return '<div class="stepbar" aria-label="评审流程">' + "".join(parts) + "</div>"


def render_counter(text: str | None) -> str:
    count = len(text or "")
    return f'<div class="char-counter"><strong>{count:,}</strong> / 12,000 字符</div>'


def _severity_class(severity: str) -> str:
    return {"高": "high", "中": "medium", "低": "low"}.get(severity, "medium")


def _grade_class(score: float) -> str:
    if score < 6:
        return "is-high"
    if score >= 8:
        return "is-good"
    return ""


def render_error_html(message: str, raw: str = "") -> str:
    raw_html = ""
    if raw:
        raw_html = (
            '<details class="reference-box"><summary>查看模型原始返回</summary>'
            f'<pre class="error-detail">{_escape(raw)}</pre></details>'
        )
    return f"""
    <div class="error-panel">
      <h2>本次评审未完成</h2>
      <p>{_escape(message)}</p>
      <p>可以返回修改输入后重新提交，或检查 API Key 与网络连接。</p>
      {raw_html}
    </div>
    """


def render_report_html(report: dict[str, Any]) -> str:
    """把结构化评审结果渲染成桌面评审工作台。"""
    score = _clamp_score(report.get("score"))
    grade = _text(report.get("grade"), "待评估")
    grade_class = _grade_class(score)
    issues: list[dict[str, Any]] = report.get("issues") or []
    counts = report.get("counts") or {}
    source_requirement = _text(report.get("source_requirement"))
    summary = _text(report.get("summary"))

    dimensions_html = ""
    dimensions = report.get("dimensions") or []
    if dimensions:
        dimension_items = []
        for dimension in dimensions:
            dim_score = _clamp_score(dimension.get("score"))
            percentage = max(0, min(100, round(dim_score * 10)))
            dimension_items.append(
                f"""
                <div class="dimension-item">
                  <div class="dimension-meta">
                    <span>{_escape(dimension.get("name"))}</span>
                    <span>{dim_score:.1f}</span>
                  </div>
                  <div class="dimension-track" aria-hidden="true">
                    <div class="dimension-fill" style="width: {percentage}%"></div>
                  </div>
                </div>
                """
            )
        dimensions_html = f'<div class="dimension-list">{"".join(dimension_items)}</div>'

    priority_issues = report.get("priority_issues") or []
    priority_html = ""
    if priority_issues:
        priority_text = "；".join(
            f"{index}. {_escape(title)}" for index, title in enumerate(priority_issues, start=1)
        )
        priority_html = (
            '<div class="priority-strip"><strong>优先处理：</strong>'
            f"{priority_text}</div>"
        )

    nav_items = []
    issue_panels = []
    copy_sources = []

    for index, issue in enumerate(issues):
        issue_id = str(issue.get("id", index + 1))
        severity = _text(issue.get("severity"), "中")
        severity_class = _severity_class(severity)
        active_class = " is-active" if index == 0 else ""
        active_nav_class = " is-active" if index == 0 else ""
        title = _text(issue.get("title"), f"问题 {index + 1}")
        category = _text(issue.get("category"), "需求质量")
        location = _text(issue.get("location"))
        reason = _text(issue.get("reason"))
        impact = _text(issue.get("impact"))
        acceptance = _text(issue.get("acceptance"))
        original = _text(issue.get("original"))
        rewrite = _text(issue.get("rewrite"))
        basis = issue.get("basis") or []

        nav_items.append(
            f"""
            <button type="button" class="issue-nav-item{active_nav_class}"
                    data-issue-nav="{_escape(issue_id)}"
                    aria-selected="{'true' if index == 0 else 'false'}">
              <span class="nav-index">{index + 1:02d}</span>
              <span class="nav-title">{_escape(title)}</span>
              <span class="nav-severity {severity_class}" aria-label="{_escape(severity)}优先级"></span>
            </button>
            """
        )

        basis_html = ""
        if basis:
            chips = "".join(
                f'<span class="basis-chip">{_escape(item)}</span>' for item in basis
            )
            basis_html = f'<div class="basis-list">{chips}</div>'

        impact_block = ""
        if impact:
            impact_block = f"""
            <div class="info-block full">
              <div class="info-label">影响</div>
              <p class="info-text">{_escape(impact)}</p>
            </div>
            """

        original_pane = ""
        if original:
            original_pane = f"""
            <div class="diff-pane">
              <div class="diff-head">原文</div>
              <p class="diff-text">{_escape(original)}</p>
            </div>
            """

        rewrite_pane = ""
        if rewrite:
            rewrite_pane = f"""
            <div class="diff-pane revised">
              <div class="diff-head">建议改写</div>
              <p class="diff-text">{_escape(rewrite)}</p>
            </div>
            """
            source_key = f"rewrite-{issue_id}"
            copy_sources.append(
                f'<textarea class="copy-source" data-copy-source="{source_key}">'
                f"{_escape(rewrite)}</textarea>"
            )
            copy_button = (
                f'<button type="button" class="copy-btn" data-copy="{source_key}">'
                "复制改写</button>"
            )
        else:
            copy_button = ""

        diff_html = ""
        if original_pane or rewrite_pane:
            diff_html = f"""
            <section class="rewrite-card">
              <div class="rewrite-heading">
                <h3>修改建议</h3>
                {copy_button}
              </div>
              <div class="diff-grid">
                {original_pane}
                {rewrite_pane}
              </div>
            </section>
            """

        acceptance_html = ""
        if acceptance:
            acceptance_html = f"""
            <div class="acceptance-note">
              <strong>验收：</strong>
              <span>{_escape(acceptance)}</span>
            </div>
            """

        issue_panels.append(
            f"""
            <article class="issue-panel{active_class}" data-issue-id="{_escape(issue_id)}">
              <div class="issue-kicker">
                <span class="issue-number">ISSUE {index + 1:02d}</span>
                <span class="severity-pill {severity_class}">{_escape(severity)}优先级</span>
                <span class="category-pill">{_escape(category)}</span>
              </div>
              <h2 class="issue-title">{_escape(title)}</h2>
              <div class="diagnosis-grid">
                <div class="info-block full">
                  <div class="info-label">位置</div>
                  <p class="info-text location-text">{_escape(location or "整段需求")}</p>
                </div>
                <div class="info-block">
                  <div class="info-label">原因</div>
                  <p class="info-text">{_escape(reason)}</p>
                </div>
                <div class="info-block">
                  <div class="info-label">标准依据</div>
                  {basis_html or '<p class="info-text">未提供具体依据</p>'}
                </div>
                {impact_block}
              </div>
              {diff_html}
              {acceptance_html}
            </article>
            """
        )

    final_requirement = _text(report.get("final_requirement"))
    final_section = ""
    if final_requirement:
        copy_sources.append(
            '<textarea class="copy-source" data-copy-source="final-requirement">'
            f"{_escape(final_requirement)}</textarea>"
        )
        final_section = f"""
        <section class="final-section">
          <div class="final-heading">
            <div>
              <h3>完整改写版本</h3>
            </div>
            <button type="button" class="copy-btn" data-copy="final-requirement">
              复制完整需求
            </button>
          </div>
          <pre class="final-text">{_escape(final_requirement)}</pre>
        </section>
        """

    references = report.get("references") or []
    references_section = ""
    if references:
        reference_items = "".join(f"<li>{_escape(item)}</li>" for item in references)
        references_section = f"""
        <details class="reference-box">
          <summary>参考依据（{len(references)} 项）</summary>
          <ul>{reference_items}</ul>
        </details>
        """

    if issues:
        report_content = f"""
        <div class="report-grid">
          <aside class="report-sidebar">
            <section class="sidebar-card">
              <div class="sidebar-heading">问题导航 <span>{len(issues)} 项</span></div>
              <div class="issue-nav-list" role="tablist" aria-label="问题列表">
                {''.join(nav_items)}
              </div>
            </section>
            <details class="sidebar-card source-box" open>
              <summary>原始需求</summary>
              <p class="source-text">{_escape(source_requirement)}</p>
            </details>
          </aside>
          <main class="report-main">
            {''.join(issue_panels)}
          </main>
        </div>
        """
    else:
        report_content = """
        <div class="empty-issues">
          <h2>未发现需要修正的问题</h2>
          <p>本次检查未识别出明显缺陷。仍建议结合业务背景和验收条件做人工复核。</p>
        </div>
        """

    return f"""
    <div class="report-shell">
      <section class="report-summary">
        <div class="score-overview">
          <div class="score-ring"
               style="--score: {max(0, min(100, round(score * 10)))}%; --ring: {'#16865b' if score >= 8 else '#d97706' if score >= 6 else '#c93b43'}"
               aria-label="质量评分 {score:.1f} 分，满分 10 分">
            <div class="score-value"><strong>{score:.1f}</strong><span>/10</span></div>
          </div>
          <div>
            <div class="grade-line">
              <span class="grade-pill {grade_class}">{_escape(grade)}</span>
              <span class="issue-number">质量评估结论</span>
            </div>
            <p class="summary-copy">{_escape(summary)}</p>
          </div>
        </div>
        <div class="summary-side">
          <div class="risk-row">
            <span class="risk-chip high">高优先级 {counts.get("high", 0)}</span>
            <span class="risk-chip medium">中优先级 {counts.get("medium", 0)}</span>
            <span class="risk-chip low">低优先级 {counts.get("low", 0)}</span>
          </div>
          {dimensions_html}
          {priority_html}
        </div>
      </section>
      {report_content}
      {final_section}
      {references_section}
      {''.join(copy_sources)}
    </div>
    """


def fill_example(example_text: str) -> tuple[str, str]:
    return example_text, render_counter(example_text)


def clear_input() -> tuple[str, str]:
    return "", render_counter("")


def run_review(requirement_text: str):
    """执行评审并用生成器提供明确的加载态和结果态。"""
    if not requirement_text or not requirement_text.strip():
        raise gr.Error("请先输入需要评审的需求文本。")

    yield (
        gr.update(visible=False),
        gr.update(visible=True),
        render_steps(2),
        gr.update(interactive=False, value="正在分析需求..."),
        LOADING_HTML,
    )

    try:
        report = review_requirement(requirement_text)
        report_html = render_report_html(report)
    except ReviewException as exc:
        report_html = render_error_html(str(exc), raw=exc.raw)
    except Exception as exc:
        report_html = render_error_html(f"评审过程中发生未知错误：{exc}")

    yield (
        gr.update(visible=False),
        gr.update(visible=True),
        render_steps(3),
        gr.update(interactive=True, value="开始智能评审"),
        report_html,
    )


def back_to_input():
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        render_steps(1),
    )


with gr.Blocks(title="需求工程评审专家", fill_width=True) as demo:
    with gr.Column(elem_id="app-shell"):
        gr.HTML(HEADER_HTML, apply_default_css=False, js_on_load=None)
        step_bar = gr.HTML(render_steps(1), apply_default_css=False, js_on_load=None)

        with gr.Row(elem_id="input-workspace") as input_workspace:
            with gr.Column(scale=7, elem_id="input-panel"):
                gr.HTML(INPUT_HEADER_HTML, apply_default_css=False, js_on_load=None)
                input_text = gr.Textbox(
                    value="",
                    label="需求文本",
                    show_label=False,
                    placeholder="在这里粘贴你的需求文本，例如：用户必须能在3秒内完成登录操作……",
                    lines=12,
                    max_lines=24,
                    max_length=12000,
                    elem_id="requirement-textarea",
                    container=False,
                )
                with gr.Row(elem_id="input-action-row"):
                    char_counter = gr.HTML(
                        render_counter(""),
                        apply_default_css=False,
                        js_on_load=None,
                        scale=2,
                    )
                    with gr.Column(scale=1, min_width=76, elem_id="clear-example-row"):
                        clear_btn = gr.Button("清空", elem_id="clear-btn", size="md")
                    review_btn = gr.Button(
                        "开始智能评审  →",
                        variant="primary",
                        elem_id="review-btn",
                        size="md",
                    )

            with gr.Column(scale=4, elem_id="examples-panel"):
                gr.HTML(EXAMPLES_HEADER_HTML, apply_default_css=False, js_on_load=None)
                example_buttons = []
                for tag, subtitle, text in EXAMPLES:
                    button = gr.Button(
                        f"{tag}  ·  {subtitle}",
                        elem_classes=["example-button"],
                        size="md",
                    )
                    button.click(
                        fn=lambda example_text=text: fill_example(example_text),
                        outputs=[input_text, char_counter],
                    )
                    example_buttons.append(button)
                gr.HTML(GUIDE_HTML, apply_default_css=False, js_on_load=None)

        with gr.Column(visible=False, elem_id="result-workspace") as result_workspace:
            with gr.Row(elem_id="result-toolbar"):
                gr.HTML(
                    RESULT_HEADER_HTML,
                    apply_default_css=False,
                    js_on_load=None,
                    scale=1,
                )
                back_btn = gr.Button(
                    "返回修改",
                    elem_id="back-btn",
                    size="sm",
                    scale=0,
                    min_width=112,
                )
            report_html = gr.HTML(
                "",
                elem_id="report-root",
                apply_default_css=False,
                js_on_load=REPORT_JS,
            )

        gr.HTML(FOOTER_HTML, apply_default_css=False, js_on_load=None)

    input_text.input(
        fn=render_counter,
        inputs=input_text,
        outputs=char_counter,
        show_progress="hidden",
    )
    clear_btn.click(
        fn=clear_input,
        outputs=[input_text, char_counter],
        show_progress="hidden",
    )
    review_btn.click(
        fn=run_review,
        inputs=input_text,
        outputs=[
            input_workspace,
            result_workspace,
            step_bar,
            review_btn,
            report_html,
        ],
        show_progress="hidden",
    )
    back_btn.click(
        fn=back_to_input,
        outputs=[input_workspace, result_workspace, step_bar],
        show_progress="hidden",
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4)
    demo.launch(
        server_name="127.0.0.1",
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=False,
        css=GLOBAL_CSS,
    )
