"""配置文件"""
import os
from typing import Optional

class Settings:
    """应用配置"""

    # API配置
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_TITLE: str = "HelloAgents AI Town"
    API_VERSION: str = "1.0.0"

    # NPC配置
    NPC_UPDATE_INTERVAL = 30  # NPC状态更新间隔

    # LLM配置(从环境变量读取)
    LLM_MODEL_ID: str = os.getenv("LLM_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.wjstest.com/v1")

    # CORS配置(非生产环境)
    CORS_ORIGINS: Optional[list] = ["*"]

    @classmethod
    def validate(cls):
        """验证配置"""
        if not cls.LLM_API_KEY:
            print("""LLM_API_KEY未设置""")
            return False

        print(f"LLM配置：\n模型：{cls.LLM_MODEL_ID}\n服务地址：{cls.LLM_BASE_URL}")
        return True

settings = Settings()