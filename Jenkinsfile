// Jenkinsfile
pipeline {
    agent any // Assumes the Jenkins agent has Docker installed

    stages {
        stage('Checkout') {
            steps {
                // Checkout code from your version control (e.g., Git)
                checkout scm
            }
        }

        stage('Run Linter') {
            // It's good practice to lint your code for style and errors.
            // We run this inside a temporary Python container for a clean environment.
            agent {
                docker { image 'python:3.10-slim' }
            }
            steps {
                sh 'pip install flake8'
                // Run flake8, ignoring common non-critical errors if needed
                // Example: sh 'flake8 app.py --ignore=E501,W503'
                sh 'flake8 app.py'
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    echo "Building the Docker image..."
                    // Build the image and tag it
                    sh "docker build -t global-mineral-dashboard:latest ."
                }
            }
        }

        stage('Scan Image (Placeholder)') {
            // In a real pipeline, you would scan the built image
            // for vulnerabilities here using tools like Trivy, Snyk, or Docker Scout.
            steps {
                echo "Skipping vulnerability scan for this example."
                // Example with Trivy (if installed on agent):
                // sh "trivy image global-mineral-dashboard:latest"
            }
        }
    }
    
    post {
        // This block runs after all stages
        success {
            echo 'Pipeline finished successfully.'
        }
        failure {
            echo 'Pipeline failed. Check the logs.'
        }
    }
}