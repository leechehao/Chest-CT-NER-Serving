import transformers
import onnxruntime as ort
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from spacy import displacy

import utils


session = ort.InferenceSession("./onnx_model/model.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

with open("./label_list.txt") as file:
    label_list = eval(file.read())

tokenizer = transformers.AutoTokenizer.from_pretrained("./tokenizer")
id2label = {i: tag for i, tag in enumerate(label_list)}

ent_type = set([label.split("-")[-1] for label in label_list if "-" in label])
colors = {key: utils.generate_random_color_hex() for key in ent_type}
options = {"ents": ent_type, "colors": colors}

pipeline = utils.Pipeline(
    session=session,
    tokenizer=tokenizer,
    id2label=id2label,
    aggregation_strategy=utils.AggregationStrategy.FIRST,
)


def display_ent(text: str):
    grouped_entities = pipeline(text)
    ents = [
        {"start": ent["start"], "end": ent["end"], "label": ent["entity_group"]}
        for ent in grouped_entities if ent["entity_group"] != "O"
    ]
    raw_data = {
        "text": text,
        "ents": ents,
        "title": None,
    }
    html = displacy.render(raw_data, style="ent", manual=True, options=options)
    return HTMLResponse(content=html)


app = FastAPI()

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/submit")
async def handle_form(request: Request, item: str = Form(...)):
    return display_ent(item)


# "The CT image reveals an irregularly bordered nodule in the right upper lobe, approximately 1.5 cm in diameter, with heterogeneous internal density."
# "Multiple small nodules with mild to moderate ground-glass opacities are observed in the left lower lobe."
# "The chest CT has disclosed a lobulated lesion situated between the hilum and pleura, with clear demarcation from the surrounding tissue."
# "A round shadow approximately 2 cm in size is found at the posterior aspect of the thoracic cavity, potentially indicative of a benign pulmonary hamartoma."
# "A flat lesion on the upper lobe of the left lung is visible on the CT scan, characterized by tiny calcifications on its surface."
# "Bilateral lower lobes exhibit honeycombing patterns, which may be associated with interstitial lung disease."
# "A lesion with cavitation is detected in the apical region of the lung, surrounded by mild inflammatory changes."
# "The CT scan demonstrates a pleural-based lesion containing fluid in the right lung, suggesting further diagnostic evaluation is warranted."
