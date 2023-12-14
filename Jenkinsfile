pipeline {
    agent any
    environment {
            TRACKING_URI = credentials('75541b4f-29c2-4667-8903-986d54e66a0b')
            MODEL_NAME = 'Chest_CT_NER'
            HARBOR_DOMAIN_NAME = credentials('882ed71f-6d60-4d55-8a3d-b00cddf97c43')
            HARBOR_USERNAME = credentials('ad6a7fbe-ab1e-4ee9-ad88-53f01c149452')
            HARBOR_PASSEORD = credentials('f11bdace-06a9-4614-83d9-92a3eb7d2409')
            LATEST_MODEL_VERSION = ''
        }
    stages {
        stage('Get Model version') {
            agent {
                dockerfile {
                    filename 'Dockerfile-agent'
                }
            }
            steps {
                script {
                    LATEST_MODEL_VERSION = sh(script: 'model_version/run_model_version.sh', returnStdout: true).trim()
                    echo "最新的版本: ${LATEST_MODEL_VERSION}"
                }
            }
        }
        stage('Push image') {
            steps {
                script {
                    sh "push_image/run_push_image.sh ${LATEST_MODEL_VERSION}"
                }
            }
        }
    }
    post{
        always {
            cleanWs()
        }
    }
}
