# Machine Learning Pipeline for Flight Delay Prediction

### Repository Structure
```
pipeline/
│
├── Dockerfile
├── src/
│   ├── aws_utils.py
│   ├── evaluate_performance.py
│   ├── generate_feature.py
│   ├── score_model.py
│   ├── train_model.py
├── config/
│   ├── default-config.yaml
│   ├── logs/
│       └── logging.conf
├── logs/
├── tests/
│   ├── test_evaluate_performance.py
│   ├── test_generate_features.py
├── ml_pipeline.py
├── README.md
└── requirements.txt
```

- `src`: Contains all source codes and fucntion for running ml_pipeline.py.
- `config`: Main config and logging command center.
- `logs`: Save logging result.
- `tests`: Contains python files for Unit Testing purpose.

**Note**: requirements.txt has slightly different lists of packages compared to `web-app` folder's requirements.txt due to the inclusion of streamlit related libaries.


### How We Built ML Pipeline

Once we have all the codes and assume that the processed data is stored in the S3 bucket, we could start building docker and publish image to ECR. 

```bash
aws sso login

aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/XXXX
```

**Note:** We anonymized the public ECR unique ID to be `XXXX` for privacy & security purpose.

Once we have the new Docker Image built, make sure to follow the command below to deploy in ECR:

```bash
# Build the Docker Image
docker build -t project .

# Tag the Image to Public ECR AWS Image
docker tag project:latest public.ecr.aws/XXXX/project:latest

# Publish the Image into ECR Repository
docker push public.ecr.aws/XXXX/project:latest
```

After deploying the Pipeline Docker Image to ECR, connect the ECR to the ECS. Create task definition and run task through FARGATE. A task should show up within Cluster. 

Once the task is finished, all model artifacts should be saved to the s3 bucket, under model_artifacts 