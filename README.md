# Chest CT NER Serving
在 Chest CT NER 專案中，Continuous Training (CT) 流程的 Model Validation 階段，一旦有新模型註冊則觸發 Chest CT NER Serving 的 Jenkins pipeline。

Pipeline 主要分為三步驟：
+ Download Model：從 Mlflow 模型註冊中下載最新版本的模型及 Tokenizer。
+ Build and Push Docker Image：構建模型服務 API 的 Docker Image 並上傳到私人的 Docker Image Registry（Harbor）。
+ Model Serving：啟動模型服務的 Container。

## 前端介面
前端介面則是利用 Gradio 套件實現。