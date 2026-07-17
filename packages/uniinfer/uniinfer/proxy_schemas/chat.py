import re
import time
import uuid
from typing import Any, ClassVar

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
    # Forwarded to the backend's chat template (vLLM/HuggingFace). This is the
    # RELIABLE way to control thinking for Qwen3.x / GLM-5.x models: many
    # backends silently ignore top-level `enable_thinking` but honor
    # chat_template_kwargs.enable_thinking (see vLLM #35574).
    # e.g. {"enable_thinking": false} fully disables reasoning.
    chat_template_kwargs: dict | None = None

    def get_effective_max_tokens(self) -> int | None:
        return self.max_completion_tokens or self.max_tokens

    MAX_MESSAGES: ClassVar[int] = 5000

    @field_validator("messages")
    @classmethod
    def validate_messages_count(cls, v):
        if len(v) > cls.MAX_MESSAGES:
            raise ValueError(f"Too many messages. Maximum allowed is {cls.MAX_MESSAGES}.")
        return v

    @field_validator("model")
    @classmethod
    def validate_model_format(cls, v):
        if "@" not in v:
            raise ValueError("Invalid model format. Expected 'provider@modelname'.")
        parts = v.split("@", 1)
        if not parts[0] or not parts[1]:
            raise ValueError("Invalid model format. Provider or model name is empty.")
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


class PromptTokensDetails(BaseModel):
    cached_tokens: int | None = None


class CompletionTokensDetails(BaseModel):
    reasoning_tokens: int | None = None


class CompletionUsage(BaseModel):
    """OpenAI-spec usage. Fields the proxy doesn't get default to 0/None; unknown
    provider fields are ignored. Models the nested *_details objects that some
    providers (e.g. Mistral, OpenAI reasoning models) return — a flat
    ``dict[str, int]`` rejected those and 400'd non-streaming responses."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: PromptTokensDetails | None = None
    completion_tokens_details: CompletionTokensDetails | None = None


class NonStreamingChatCompletion(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[NonStreamingChoice]
    usage: CompletionUsage | None = None


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
    usage: CompletionUsage | None = None
