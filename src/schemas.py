from typing import List, Union, Optional, Literal
from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(..., description="Text or list of texts to embed")
    model: str = Field(default="nomic-embed-text-v2-moe-distilled", description="Model to use for embedding")
    encoding_format: str = Field(default="float", description="Encoding format (float or base64)")
    dimensions: Optional[int] = Field(default=None, description="Number of dimensions to return")
    user: Optional[str] = Field(default=None, description="User identifier")
    stream: Optional[bool] = Field(default=False, description="Whether to stream the response")
    priority: Optional[Literal["low", "normal", "high", "urgent"]] = Field(default="normal", description="Request priority level")


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