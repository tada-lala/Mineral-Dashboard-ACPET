pipeline {
    agent any 
    environment {
        DOCKERHUB_CREDENTIALS = credentials('dockerhub-cred')
    }
    stages {
        stage('Build docker image') {
            steps {  
                echo "Building Docker image..."
                sh '''
                    docker pull tada8102/flaskapp:latest || true
                    docker build \
                        --cache-from tada8102/flaskapp:latest \
                        -t tada8102/flaskapp:$BUILD_NUMBER \
                        -t tada8102/flaskapp:latest .
                '''
            }
        }

        stage('login to dockerhub') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub-cred', usernameVariable: 'DOCKERHUB_USERNAME', passwordVariable: 'DOCKERHUB_PASSWORD')]) {
                    echo "Logging in to Docker Hub..."
                    sh "docker login -u $DOCKERHUB_USERNAME -p $DOCKERHUB_PASSWORD"
                }
            }
        }
        stage('push image') {
            steps {
                echo "Pushing Docker image to Docker Hub..."
                sh 'docker push tada8102/flaskapp:$BUILD_NUMBER'
            }
        }
        stage('Deploy to Staging') {
            steps {
                // Pull the latest image from Docker Hub
                sh 'docker pull tada8102/flaskapp:$BUILD_NUMBER'

                // Stop and remove any existing containers
                sh 'docker stop myapp-container || true'
                sh 'docker rm myapp-container || true'

                // Run the new container with the latest image
                sh 'docker run -d --name myapp-container -p 8000:8000 tada8102/flaskapp:$BUILD_NUMBER'
            }
        }
    }
    post {
        always {
            echo "Logging out from Docker Hub..."
            sh 'docker logout'
        }
    }
}
