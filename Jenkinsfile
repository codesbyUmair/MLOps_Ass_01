pipeline {
    agent any
    
    environment {
        DOCKER_HUB_CREDENTIALS = credentials('dockerhub-credentials')
        DOCKER_IMAGE = 'saaramcheema/i211182-i211226-assignment01'
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
        
            post {
        success {
            mail to: 'admin@example.com',
                subject: "Deployment Successful - ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                body: """The latest version has been deployed successfully.

                Job: ${env.JOB_NAME}
                Build Number: ${env.BUILD_NUMBER}
                Docker Image: ${DOCKER_IMAGE}:${env.BUILD_NUMBER}
                
                View Details: ${env.BUILD_URL}
                """
        }

        failure {
            mail to: 'admin@example.com',
                subject: "Deployment Failed - ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                body: """The deployment has failed.

                Job: ${env.JOB_NAME}
                Build Number: ${env.BUILD_NUMBER}
                
                View Logs: ${env.BUILD_URL}
                """
        }
    }
    }
}
