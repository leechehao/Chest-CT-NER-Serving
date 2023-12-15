import os

import mlflow
import transformers


mlflow.set_tracking_uri(os.getenv("TRACKING_URI"))
client = mlflow.MlflowClient()
model_name = os.getenv("MODEL_NAME")
run_id = client.get_registered_model(model_name).latest_versions[0].run_id

run = mlflow.get_run(run_id)
pretrained_model_name_or_path = run.data.params["pretrained_model_name_or_path"]
tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
tokenizer.save_pretrained("./src/tokenizer")
with open("./src/label_list.txt", "w") as file:
    file.write(run.data.params["label_list"])

registered_model = client.get_registered_model(model_name).latest_versions[0]
mlflow.artifacts.download_artifacts(run_id=registered_model.run_id, artifact_path="onnx_model", dst_path="./src")
print(registered_model.version)
