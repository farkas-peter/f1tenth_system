import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from gemini_agent import GeminiAgent


class AudioPath(BaseModel):
    path: str

class BoundingBoxes(BaseModel):
    bb_list: list[list[int]]

app = FastAPI()
agent = GeminiAgent()

@app.get("/hello")
def hello():
    return "hello, i am the gemini agent server :)"

@app.post("/run_agent_pipeline", response_model=BoundingBoxes)
def run_agent_pipeline(data: AudioPath):
    print("Processing audio:", data.path)
    bbs = agent.run_pipeline(data.path)

    return BoundingBoxes(bb_list=bbs)


if __name__ == "__main__":
    uvicorn.run("gemini_server:app", host="127.0.0.1", port=8000, reload=True)

