import requests
import gradio as gr


def ner(text):
    response = requests.post("http://192.168.1.76:9528/ner", json={"text": text})
    return response.json()


examples = [
    "The CT image reveals an irregularly bordered nodule in the right upper lobe, approximately 1.5 cm in diameter, with heterogeneous internal density.",
    "Multiple small nodules with mild to moderate ground-glass opacities are observed in the left lower lobe.",
    "The chest CT has disclosed a lobulated lesion situated between the hilum and pleura, with clear demarcation from the surrounding tissue.",
    "A round shadow approximately 2 cm in size is found at the posterior aspect of the thoracic cavity, potentially indicative of a benign pulmonary hamartoma.",
    "A flat lesion on the upper lobe of the left lung is visible on the CT scan, characterized by tiny calcifications on its surface.",
    "Bilateral lower lobes exhibit honeycombing patterns, which may be associated with interstitial lung disease.",
    "A lesion with cavitation is detected in the apical region of the lung, surrounded by mild inflammatory changes.",
    "The CT scan demonstrates a pleural-based lesion containing fluid in the right lung, suggesting further diagnostic evaluation is warranted.",
]

demo = gr.Interface(
    ner,
    gr.Textbox(placeholder="Enter sentence here..."),
    gr.HighlightedText(),
    examples=examples
)

demo.launch(server_name="0.0.0.0", server_port=9529)
