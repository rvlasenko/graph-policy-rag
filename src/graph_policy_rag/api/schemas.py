from pydantic import BaseModel, Field


class QueryApiRequest(BaseModel):
    text: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1)
