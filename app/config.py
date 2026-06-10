from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Single source of truth for embedding dimension (CLAUDE.md §3.2).
# Both the Neo4j vector index DDL and the fastembed wrapper import from here.
EMBED_DIM = 384


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    llm_mode: str = "fake"
    fake_violation: int = 0
    generation_model: str = "claude-sonnet-4-6"
    judge_model: str = "claude-haiku-4-5"
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    redis_url: str = "redis://redis:6379/0"
    result_backend: str = "redis://redis:6379/1"
    token_budget_facts: int = 2000
    token_budget_passages: int = 2000

    @model_validator(mode="after")
    def require_key_in_anthropic_mode(self) -> "Settings":
        if self.llm_mode == "anthropic" and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set when LLM_MODE=anthropic")
        return self


settings = Settings()
