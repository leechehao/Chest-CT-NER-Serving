#!/bin/bash

latest_version=$(curl -X POST -H "Content-Type: application/json" -d "{\"name\": \"$MODEL_NAME\"}" $TRACKING_URI/api/2.0/mlflow/registered-models/get-latest-versions | jq -r '.model_versions[0].version')

echo $latest_version
