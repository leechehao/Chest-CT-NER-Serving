import os

import transformers
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel

import utils


app = FastAPI()


class NERRequest(BaseModel):
    text: str


cuda_available = os.environ.get("CUDA_AVAILABLE", "false") == "true"
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda_available else ["CPUExecutionProvider"]

session = ort.InferenceSession("./onnx_model/model.onnx", providers=providers)
tokenizer = transformers.AutoTokenizer.from_pretrained("./tokenizer")

with open("./label_list.txt") as file:
    label_list = eval(file.read())

id2label = {i: tag for i, tag in enumerate(label_list)}

pipeline = utils.Pipeline(
    session=session,
    tokenizer=tokenizer,
    id2label=id2label,
    aggregation_strategy=utils.AggregationStrategy.FIRST,
)


@app.post("/ner")
async def perform_ner(request: NERRequest):
    # 使用相同的 NER 處理邏輯
    text = request.text
    output = [
        {
            "entity": ent["entity_group"],
            "score": ent["score"].item(),
            "word": ent["word"],
            "start": ent["start"],
            "end": ent["end"],
        }
        for ent in pipeline(text) if ent["entity_group"] != "O"
    ]
    return {"text": text, "entities": output}


@app.get("/liveness")
async def liveness_check():
    return {"status": "ok"}


@app.get("/readiness")
async def readiness_check():
    return {"status": "ok"}
