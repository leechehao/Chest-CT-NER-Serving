#!/bin/bash

latest_model_version=$1

docker build -t $HARBOR_DOMAIN_NAME/model-serving/chest_ct_ner:$latest_model_version .
docker tag $HARBOR_DOMAIN_NAME/model-serving/chest_ct_ner:$latest_model_version $HARBOR_DOMAIN_NAME/model-serving/chest_ct_ner:latest 
docker login $HARBOR_DOMAIN_NAME --username $HARBOR_USERNAME --password $HARBOR_PASSEORD
docker push $HARBOR_DOMAIN_NAME/model-serving/chest_ct_ner:$latest_model_version
docker push $HARBOR_DOMAIN_NAME/model-serving/chest_ct_ner:latest
