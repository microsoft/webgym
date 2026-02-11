from fastapi import FastAPI, HTTPException, status, Response, Depends, Header

class GetApiKey:
    def __init__(self, api_key):
        self.api_key = api_key

    async def __call__(self, x_api_key: str = Header(None, alias="x-api-key")):
        """Dependency to verify the API key in the request header."""
        if not x_api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API Key is missing",
            )
        if x_api_key != self.api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid API Key.",
            )
        return x_api_key