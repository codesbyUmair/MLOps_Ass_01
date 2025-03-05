pipeline {
    agent any
    
    environment {
        DOCKER_HUB_CREDENTIALS = credentials('dockerhub-credentials')
        DOCKER_IMAGE = 'yourusername/devops-demo-app'
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Build Docker Image') {
            steps {
                script {
                    docker.build("${DOCKER_IMAGE}:${env.BUILD_NUMBER}")
                }
            }
        }
        
        stage('Push to Docker Hub') {
            steps {
                script {
                    docker.withRegistry('https://registry.hub.docker.com', 'dockerhub-credentials') {
                        docker.image("${DOCKER_IMAGE}:${env.BUILD_NUMBER}").push()
                        docker.image("${DOCKER_IMAGE}:${env.BUILD_NUMBER}").push('latest')
                    }
                }
            }
        }
        
        stage('Send Admin Notification') {
            steps {
                script {
                    emailext (
                        subject: "Deployment Successful",
                        body: "The application has been successfully deployed.\n\nDocker Image: ${DOCKER_IMAGE}:${env.BUILD_NUMBER}",
                        to: "admin@yourcompany.com"
                    )
                }
            }
        }
    }
}