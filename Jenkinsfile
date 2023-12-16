pipeline {
    agent any
    environment {
            TRACKING_URI = credentials('75541b4f-29c2-4667-8903-986d54e66a0b')
            MODEL_NAME = 'Chest_CT_NER'
            AWS_ACCESS_KEY_ID = credentials('57a926de-56bd-4634-840d-986f704ce9bd')
            AWS_SECRET_ACCESS_KEY = credentials('c522bcb8-8c81-486a-8022-f1f70c016a26')
            MLFLOW_S3_ENDPOINT_URL = credentials('f9527c9d-6b16-4677-b7e7-0bfac884a319')
            MLFLOW_S3_IGNORE_TLS = true
            HARBOR_DOMAIN_NAME = credentials('882ed71f-6d60-4d55-8a3d-b00cddf97c43')
            HARBOR_USERNAME = credentials('ad6a7fbe-ab1e-4ee9-ad88-53f01c149452')
            HARBOR_PASSEORD = credentials('f11bdace-06a9-4614-83d9-92a3eb7d2409')
            LATEST_MODEL_VERSION = ''
            CONTAINER_NAME = 'Chest_CT_NER-Serving'
            PORT = '9528'
        }
    stages {
        stage('Download model') {
            agent {
                docker {
                    image 'jenkins-chest_ct_ner:latest'
                    reuseNode true
                }
            }
            steps {
                script {
                    LATEST_MODEL_VERSION = sh(script: '1_download_model/run_download_model.sh', returnStdout: true).trim()
                    echo "最新的版本: ${LATEST_MODEL_VERSION}"
                }
            }
        }
        stage('Build and Push image') {
            steps {
                sh "2_build_push_image/run_build_push_image.sh ${LATEST_MODEL_VERSION}"
            }
        }
        stage('Model serving') {
            steps {
                sh "3_model_serving/run_model_serving.sh"
            }
        }
    }
    post{
        always {
            cleanWs()
        }
    }
}
