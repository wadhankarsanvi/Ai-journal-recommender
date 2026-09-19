from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env file."""

    # Academic API settings (No paid keys required)
    openalex_email: str | None = "researcher@academic-assistant.ai"
    crossref_email: str | None = "researcher@academic-assistant.ai"

    # Optional LLM API keys for advanced multi-model extraction & reasoning
    gemini_api_key: str | None = None
    openai_api_key: str | None = None
    groq_api_key: str | None = None
    semantic_scholar_api_key: str | None = None

    # LLM Model Configuration
    llm_provider: str = "auto"  # 'auto', 'gemini', 'openai', 'groq', or 'heuristic'
    gemini_model: str = "gemini-2.5-flash"
    openai_model: str = "gpt-4o-mini"

    # Server settings
    host: str = "127.0.0.1"
    port: int = 7860
    debug: bool = False

    # Vector store settings
    vectorstore_dir: str = "data/vectorstore"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
