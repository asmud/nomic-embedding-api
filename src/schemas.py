from typing import List, Union, Optional, Literal
from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(
        ..., 
        description="Text or list of texts to embed. Can be a single string or array of strings.",
        examples=["Hello world", ["Hello world", "How are you?"]]
    )
    model: str = Field(
        default="nomic-embed-text-v2-moe-distilled", 
        description="Model identifier to use for generating embeddings. Currently supports 'nomic-embed-text-v2-moe-distilled'.",
        examples=["nomic-embed-text-v2-moe-distilled"]
    )
    encoding_format: str = Field(
        default="float", 
        description="Format for returned embeddings. 'float' returns arrays of floating point numbers, 'base64' returns base64-encoded strings.",
        examples=["float", "base64"]
    )
    user: Optional[str] = Field(
        default=None, 
        description="Unique identifier representing your end-user for tracking and rate limiting purposes.",
        examples=["user-123", "session-abc"]
    )
    stream: Optional[bool] = Field(
        default=False, 
        description="Whether to stream back partial progress as embeddings are generated. If true, returns Server-Sent Events.",
        examples=[False, True]
    )
    priority: Optional[Literal["low", "normal", "high", "urgent"]] = Field(
        default="normal", 
        description="Request priority level for queue processing. Higher priority requests are processed first. Options: 'low', 'normal', 'high', 'urgent'.",
        examples=["normal", "high"]
    )


class EmbeddingData(BaseModel):
    object: str = Field(default="embedding", description="Object type")
    embedding: List[float] = Field(..., description="The embedding vector")
    index: int = Field(..., description="Index of the input text")


class EmbeddingUsage(BaseModel):
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    total_tokens: int = Field(..., description="Total number of tokens")


class EmbeddingResponse(BaseModel):
    object: str = Field(default="list", description="Object type")
    data: List[EmbeddingData] = Field(..., description="List of embeddings")
    model: str = Field(..., description="Model used for embedding")
    usage: EmbeddingUsage = Field(..., description="Token usage information")


class ErrorResponse(BaseModel):
    error: dict = Field(..., description="Error information")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    model: str = Field(..., description="Current model")
    embedding_dimension: int = Field(..., description="Embedding dimension")


class ModelsResponse(BaseModel):
    object: str = Field(default="list", description="Object type")
    data: List[dict] = Field(..., description="List of available models")


class StreamingEmbeddingChunk(BaseModel):
    """Individual chunk in streaming response"""
    object: str = Field(default="embedding.chunk", description="Object type")
    index: int = Field(..., description="Index of the embedding being streamed")
    embedding: List[float] = Field(..., description="The embedding vector")
    model: str = Field(..., description="Model used")
    

class StreamingEmbeddingDelta(BaseModel):
    """Delta for streaming embedding (partial embedding data)"""
    object: str = Field(default="embedding.delta", description="Object type")
    index: int = Field(..., description="Index of the embedding")
    partial_embedding: List[float] = Field(..., description="Partial embedding vector")
    is_complete: bool = Field(default=False, description="Whether this completes the embedding")


class StreamingEmbeddingResponse(BaseModel):
    """Final response in streaming mode"""
    object: str = Field(default="embedding.done", description="Object type")
    model: str = Field(..., description="Model used")
    usage: EmbeddingUsage = Field(..., description="Token usage information")