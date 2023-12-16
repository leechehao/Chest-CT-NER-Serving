if docker ps -a --format "{{.Names}}" | grep -q $CONTAINER_NAME; then
    docker rm -f $CONTAINER_NAME
fi

docker run --name $CONTAINER_NAME -d -p $PORT:$PORT --gpus all ${HARBOR_DOMAIN_NAME}/model-serving/chest_ct_ner:latest
