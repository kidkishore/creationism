import os
import json
import time
import uuid
import boto3
import requests

dynamo = boto3.resource('dynamodb')
job_table = dynamo.Table(os.environ['JOB_TABLE'])
s3 = boto3.client('s3')

def truncate_error_msg(error_msg, max_length=1024):
    if len(error_msg) > max_length:
        return error_msg[:max_length] + "... (truncated)"
    return error_msg

def update_job_status(job_id, status, error=None, model_url=None):
    try:
        expr = "SET #s = :s"
        ean = {"#s": "status"}
        eav = {":s": status}

        if error:
            expr += ", #e = :e"
            ean["#e"] = "error"
            eav[":e"] = truncate_error_msg(error)
            
        if model_url:
            expr += ", model_url = :u"
            eav[":u"] = model_url

        job_table.update_item(
            Key={"job_id": job_id},
            UpdateExpression=expr,
            ExpressionAttributeNames=ean,
            ExpressionAttributeValues=eav
        )
    except Exception as e:
        print(f"Error updating job status: {str(e)}")
        if error and "ValidationException" in str(e):
            try:
                job_table.update_item(
                    Key={"job_id": job_id},
                    UpdateExpression="SET #s = :s, #e = :e",
                    ExpressionAttributeNames={"#s": "status", "#e": "error"},
                    ExpressionAttributeValues={
                        ":s": status,
                        ":e": truncate_error_msg("Error occurred. Check CloudWatch logs for details.", 256)
                    }
                )
            except Exception as e2:
                print(f"Second attempt to update job status failed: {str(e2)}")

def post_to_client(domain_name, stage, conn_id, message):
    try:
        endpoint_url = f"https://{domain_name}/{stage}"
        print(f"Sending to client. URL: {endpoint_url}, ConnID: {conn_id}, Message: {json.dumps(message)}")
        
        client = boto3.client('apigatewaymanagementapi', endpoint_url=endpoint_url)
        
        if 'error' in message and isinstance(message['error'], str):
            message['error'] = truncate_error_msg(message['error'], 1024)
            
        response = client.post_to_connection(
            Data=json.dumps(message),
            ConnectionId=conn_id
        )
        print(f"Successfully sent message to client")
        return response
    except Exception as e:
        print(f"Error sending to client: {str(e)}")
        raise

def lambda_handler(event, context):
    replicate_token = os.environ["REPLICATE_API_KEY"]
    for record in event["Records"]:
        try:
            body = json.loads(record["body"])
            job_id = body["job_id"]
            prompt = body["prompt"]
            conn_id = body["connectionId"]
            domain_name = body["domainName"]
            stage = body["stage"]

            print(f"Processing job {job_id} for prompt: {prompt}")

            create_resp = requests.post(
                "https://api.replicate.com/v1/predictions",
                headers={
                    "Authorization": f"Token {replicate_token}",
                    "Content-Type": "application/json"
                },
                json={
                    "version": "1a4da7adf0bc84cd786c1df41c02db3097d899f5c159f5fd5814a11117bdf02b",
                    "input": {
                        "prompt": prompt,
                        "output_format": "json_file"
                    }
                }
            )

            if create_resp.status_code != 201:
                error_msg = f"Replicate API error: {create_resp.status_code}"
                update_job_status(job_id, "FAILED", error_msg)
                post_to_client(domain_name, stage, conn_id, {"error": error_msg})
                continue

            prediction = create_resp.json()
            prediction_id = prediction["id"]
            print(f"Created prediction {prediction_id}")

            while True:
                poll_resp = requests.get(
                    f"https://api.replicate.com/v1/predictions/{prediction_id}",
                    headers={"Authorization": f"Token {replicate_token}"}
                )
                prediction = poll_resp.json()
                status = prediction["status"]
                
                post_to_client(domain_name, stage, conn_id, {
                    "statusMessage": f"Still working... Current status: {status}"
                })

                if status == "succeeded":
                    output = prediction.get("output")
                    print(f"Prediction succeeded! Output: {json.dumps(output)}")
                    
                    if not output:
                        raise ValueError("Empty output received")

                    # Extract the json_file data
                    json_data = output.get('json_file')
                    if not json_data:
                        raise ValueError("No json_file in output")

                    post_to_client(domain_name, stage, conn_id, {
                        "meshData": json_data,
                        "status": "completed"
                    })
                    
                    update_job_status(job_id, "COMPLETED")
                    break
                elif status in ["failed", "canceled"]:
                    error_msg = f"Prediction {status}"
                    update_job_status(job_id, "FAILED", error_msg)
                    post_to_client(domain_name, stage, conn_id, {"error": error_msg})
                    break
                    
                time.sleep(3)

        except Exception as e:
            error_msg = f"Error processing job: {str(e)}"
            print(f"Exception in worker: {error_msg}")
            try:
                update_job_status(job_id, "FAILED", error_msg)
                post_to_client(domain_name, stage, conn_id, {"error": truncate_error_msg(error_msg)})
            except Exception as e2:
                print(f"Error sending failure notice: {str(e2)}")