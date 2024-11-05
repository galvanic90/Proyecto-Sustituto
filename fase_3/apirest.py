from fastapi import FastAPI
from loguru import logger

app = FastAPI(swagger_ui_parameters={"syntaxHighlight": True})

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/users/{username}")
async def read_user(username: str):
    return {"message": f"Hello {username}"}

