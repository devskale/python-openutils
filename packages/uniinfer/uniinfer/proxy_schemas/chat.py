import re
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ChatMessageInput(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequestInput(BaseModel):
    model: str
    messages: list[ChatMessageInput]
    temperature: float | None = 0.7
    max_tokens: int | None = 32768
    max_completion_tokens: int | None = None
    stream: bool | None = False
    base_url: str | None = None
    tools: list[dict] | None = None
    tool_choice: Any | None = None
    reasoning_effort: str | None = None
    think: bool | str | None = None

    def get_effective_max_tokens(self) -> int | None:
        return self.max_completion_tokens or self.max_tokens

    @field_validator("messages")
    @classmethod
    def validate_messages_count(cls, v):
        if len(v) > 500:
            raise ValueError("Too many messages. Maximum allowed is 500.")
        return v

    @field_validator("model")
    @classmethod
    def validate_model_format(cls, v):
        if "@" not in v:
            raise ValueError("Invalid model format. Expected 'provider@modelname'.")
        if not re.match(r"^[^@]+@[^@]+$", v):
            raise ValueError("Incorrect model format. Exactly one '@' expected.")
        return v


class ChatMessageOutput(BaseModel):
    role: str
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[dict] | None = None


class ChoiceDelta(BaseModel):
    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[dict] | None = None


class StreamingChoice(BaseModel):
    index: int = 0
    delta: ChoiceDelta
    finish_reason: str | None = None


class StreamingChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[StreamingChoice]


class NonStreamingChoice(BaseModel):
    index: int = 0
    message: ChatMessageOutput
    finish_reason: str = "stop"


class NonStreamingChatCompletion(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[NonStreamingChoice]
    usage: dict[str, int] | None = None


class EmbeddingRequest(BaseModel):
    model: str
    input: list[str]
    user: str | None = None


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: dict[str, int] | None = None
