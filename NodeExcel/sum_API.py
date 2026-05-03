import json
import requests
import time

# ===================== 八大模块 + 五大能力 全局配置 =====================
# 1.能力定义
ABILITY_DEF = {
    "t2i": {"name": "文生图", "need_img": False, "need_text": True},
    "i2i": {"name": "图生图", "need_img": True, "need_text": True},
    "chat": {"name": "文生文", "need_img": False, "need_text": True},
    "t2v": {"name": "文生视频", "need_img": False, "need_text": True},
    "i2v": {"name": "图生视频", "need_img": True, "need_text": True},
}

# 2.路由目标 URL
ROUTE_URL = {
    "openai": {"chat": "https://api.openai.com/v1/chat/completions", "t2i": "https://api.openai.com/v1/images/generations"},
    "zhipu": {"chat": "https://open.bigmodel.cn/api/paas/v4/chat/completions"},
    "ali": {"chat": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"},
    "volc": {"chat": "https://ark.cn-beijing.volces.com/api/v3/chat/completions"},
    "google": {"chat": "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent"},
    "ollama": {"chat": "http://localhost:11434/api/chat"}
}

# 3.协议模板
PROTOCOL = {
    "openai": {"auth_type": "Bearer", "content_type": "application/json"},
    "zhipu": {"auth_type": "Bearer", "content_type": "application/json"},
    "ali": {"auth_type": "Bearer", "content_type": "application/json"},
    "volc": {"auth_type": "Bearer", "content_type": "application/json"},
    "google": {"auth_type": "x-goog-api-key", "content_type": "application/json"},
    "ollama": {"auth_type": "none", "content_type": "application/json"}
}

# 4.认证密钥池
KEY_POOL = {
    "openai": "sk-xxx",
    "zhipu": "zhipu-xxx",
    "ali": "ali-xxx",
    "volc": "volc-xxx",
    "google": "google-xxx",
    "ollama": ""
}

# 5.模型选择
MODEL_LIST = {
    "openai": {"chat": "gpt-3.5-turbo", "t2i": "dall-e-3"},
    "zhipu": {"chat": "glm-4"},
    "ali": {"chat": "qwen-turbo"},
    "volc": {"chat": "doubao-pro"},
    "google": {"chat": "gemini-pro"},
    "ollama": {"chat": "llama3"}
}

# 6.运行策略
RUN_POLICY = {
    "timeout": 30,
    "retry": 2,
    "concurrency": 5
}

# 7.响应适配 后端统一抹平
def response_adapt(raw, vendor, ability):
    res = {"code": 200, "ability": ability, "content": "", "media": []}
    if vendor == "openai":
        if ability == "chat":
            res["content"] = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        if ability == "t2i":
            res["media"] = [item.get("url","") for item in raw.get("data", [])]
    elif vendor == "zhipu":
        res["content"] = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
    elif vendor == "ali":
        res["content"] = raw.get("output", {}).get("text", "")
    elif vendor == "volc":
        res["content"] = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
    elif vendor == "google":
        res["content"] = raw.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    elif vendor == "ollama":
        res["content"] = raw.get("message", {}).get("content", "")
    return res

# 8.可观测性日志
def log_print(info):
    print(f"[API框架日志] {time.strftime('%H:%M:%S')} | {info}")



class APIAbilityFrameworkNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ability": (["t2i","i2i","chat","t2v","i2v"],),
                "vendor": (["openai","zhipu","ali","volc","google","ollama"],),
                "prompt": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("content_text", "media_list")
    FUNCTION = "run"
    CATEGORY = "API/AbilityFramework"

    def run(self, ability, vendor, prompt):
        log_print(f"能力:{ability} 厂商:{vendor}")

        # 取配置
        url = ROUTE_URL[vendor].get(ability, ROUTE_URL[vendor]["chat"])
        proto = PROTOCOL[vendor]
        api_key = KEY_POOL[vendor]
        model = MODEL_LIST[vendor].get(ability, MODEL_LIST[vendor]["chat"])

        # 构造请求头
        headers = {"Content-Type": proto["content_type"]}
        if proto["auth_type"] == "Bearer":
            headers["Authorization"] = f"Bearer {api_key}"
        if proto["auth_type"] == "x-goog-api-key":
            headers["x-goog-api-key"] = api_key

        # 构造请求体
        body = {}
        if ability == "chat":
            body = {"model": model, "messages": [{"role":"user","content":prompt}]}
        elif ability == "t2i":
            body = {"model": model, "prompt": prompt, "n":1}

        # 重试请求
        resp_raw = {}
        for i in range(RUN_POLICY["retry"]+1):
            try:
                r = requests.post(
                    url, headers=headers, json=body,
                    timeout=RUN_POLICY["timeout"]
                )
                resp_raw = r.json()
                break
            except Exception as e:
                log_print(f"第{i+1}次请求失败:{str(e)}")
                time.sleep(1)

        # 响应适配
        unified = response_adapt(resp_raw, vendor, ability)
        log_print(f"统一适配结果:{json.dumps(unified, ensure_ascii=False)[:200]}")

        return (unified["content"], json.dumps(unified["media"]))


NODE_CLASS_MAPPINGS = {
    "APIAbilityFrameworkNode": APIAbilityFrameworkNode
}
