import json
import os

import httpx

API_BASE = os.getenv("API_BASE", "http://localhost:8000/api")

CASES = [
    {
        "question": "迟到超过三十分钟怎么处理？",
        "expected": "事假半天",
    },
    {
        "question": "年假满三年不满五年有多少天？",
        "expected": "七天",
    },
    {
        "question": "报销申请一般要在费用发生后多久提交？",
        "expected": "三十天",
    },
    {
        "question": "企业版智能客服支持什么部署方式？",
        "expected": "私有化",
    },
    {
        "question": "知识库数据会不会用于训练模型？",
        "expected": "不会",
    },
]


def main() -> None:
    passed = 0
    for case in CASES:
        response = httpx.post(
            f"{API_BASE}/chat",
            json={"query": case["question"], "top_k": 5},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        haystack = data["answer"] + json.dumps(
            data.get("citations", []), ensure_ascii=False
        )
        ok = case["expected"] in haystack
        passed += int(ok)
        print(
            f"[{'PASS' if ok else 'FAIL'}] {case['question']} "
            f"expected={case['expected']}"
        )

    print(f"Score: {passed}/{len(CASES)} = {passed / len(CASES):.1%}")


if __name__ == "__main__":
    main()