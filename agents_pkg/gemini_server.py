import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image

from gemini_agent import GeminiAgent


class PipelineInput(BaseModel):
    audio_path: str
    base64_image: str

class BoundingBoxes(BaseModel):
    bb_list: list[list[int]]

def decode_base64_to_pil(base64_str: str) -> Image.Image:
    # Remove header if base64 has a prefix like "data:image/png;base64,..."
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data)).convert("RGB")

app = FastAPI()
agent = GeminiAgent()

@app.get("/hello")
def hello():
    return "hello, i am the gemini agent server :)"

@app.post("/run_agent_pipeline", response_model=BoundingBoxes)
def run_agent_pipeline(data: PipelineInput):
    pil_img = decode_base64_to_pil(data.base64_image)
    
    print("Decoded image size:", pil_img.size)
    print("Processing audio:", data.audio_path)

    bbs = agent.run_pipeline(data.audio_path, pil_img)

    return BoundingBoxes(bb_list=bbs)


if __name__ == "__main__":
    uvicorn.run("gemini_server:app", host="127.0.0.1", port=8000, reload=True)

