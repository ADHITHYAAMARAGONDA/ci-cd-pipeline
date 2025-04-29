pipeline {
    agent any

    environment {
        DOCKERHUB_CREDENTIALS = credentials('dockerhub')
        IMAGE_NAME = "adhithya143/ci-cd-pipeline"
        IMAGE_TAG = "${env.BUILD_NUMBER}"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build Docker Image') {
            steps {
                bat "docker build -t %IMAGE_NAME%:%IMAGE_TAG% ."
                bat "docker tag %IMAGE_NAME%:%IMAGE_TAG% %IMAGE_NAME%:latest"
            }
        }

        stage('Login to Docker Hub') {
            steps {
                bat '''
                echo|set /p=%DOCKERHUB_CREDENTIALS_PSW%|docker login -u %DOCKERHUB_CREDENTIALS_USR% --password-stdin
                '''
            }
        }

        stage('Push to Docker Hub') {
            steps {
                bat "docker push %IMAGE_NAME%:%IMAGE_TAG%"
                bat "docker push %IMAGE_NAME%:latest"
            }
        }

        stage('Stop Old Container') {
            steps {
                bat 'docker stop facerecognition || exit 0'
                bat 'docker rm facerecognition || exit 0'
            }
        }

        stage('Run New Container') {
            steps {
                bat "docker run -d -p 8501:8501 --name facerecognition %IMAGE_NAME%:latest"
            }
        }

        stage('Deploy') {
            steps {
                echo 'Deployment completed to local Docker runtime.'
            }
        }
    }

    post {
        always {
            bat 'docker logout'
            bat "docker rmi %IMAGE_NAME%:%IMAGE_TAG% || exit 0"
        }
    }
}
