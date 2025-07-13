from fastapi import HTTPException, status


class ModelNotLoadedException(HTTPException):
    def __init__(self, model_name: str):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model '{model_name}' is not loaded or unavailable"
        )


class InvalidInputException(HTTPException):
    def __init__(self, message: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {message}"
        )


class EncodingException(HTTPException):
    def __init__(self, message: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Encoding error: {message}"
        )


class ModelLoadException(HTTPException):
    def __init__(self, model_name: str, error: str):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to load model '{model_name}': {error}"
        )