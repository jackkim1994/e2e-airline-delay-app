## Data Processing Lambda Python Script

This script will be the first lambda handler triggered after the first raw data was uploaded. This will trigger SQS notification concurrently that will run this python script to filter any unrelated data (in this case, cancellation) and output the cleansed, processed data. 

Then, the script will send the processed file back to S3 bucket.

## ECS Lambda Python Script

After pre-filtering process completes, it will trigger the next lambda handler that will order the Lambda to start running feature engineering and modeling inside ECS that already has docker image imported into ECR. This will run a single task when the new file is added. Depending on the size of the processed file, it may take around 5-10 minutes or more.

If we use full-size data, it can last potentially at least 30 minutes. This factor is taken into account in the AWS cost calculation.


