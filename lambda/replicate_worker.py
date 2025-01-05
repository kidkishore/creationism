import os
import json
import time
import uuid
import boto3
import requests

dynamo = boto3.resource('dynamodb')
job_table = dynamo.Table(os.environ['JOB_TABLE'])
s3 = boto3.client('s3')

def lambda_handler(event, context):
    replicate_token = os.environ["REPLICATE_API_KEY"]
    for record in event["Records"]:
        body = json.loads(record["body"])
        job_id = body["job_id"]
        prompt = body["prompt"]
        conn_id = body["connectionId"]
        domain_name = body["domainName"]
        stage = body["stage"]

        try:
            # 1) Call Replicate to start - using Point-E model
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
            
            print(f"Replicate API Response: {create_resp.status_code} - {create_resp.text}")  # Debug log
            
            if create_resp.status_code != 201:
                msg = f"Replicate error: {create_resp.text}"
                update_job_status(job_id, "FAILED", msg)
                post_to_client(domain_name, stage, conn_id, {"error": msg})
                continue

            prediction = create_resp.json()
            prediction_id = prediction["id"]
            status = prediction["status"]

            # 2) Poll for completion
            while status not in ['succeeded', 'failed', 'canceled']:
                time.sleep(3)  # avoid tight loops
                poll_resp = requests.get(
                    f"https://api.replicate.com/v1/predictions/{prediction_id}",
                    headers={"Authorization": f"Token {replicate_token}"}
                )
                poll_data = poll_resp.json()
                status = poll_data["status"]
                
                # Send progress update to client
                post_to_client(domain_name, stage, conn_id, {
                    "statusMessage": f"Still working... Current status: {status}"
                })

            if status != 'succeeded':
                msg = f"Prediction failed or was canceled. Status: {status}"
                update_job_status(job_id, "FAILED", msg)
                post_to_client(domain_name, stage, conn_id, {"error": msg})
                continue

            # Success path continues here...
            print(f"Prediction succeeded! Output: {json.dumps(poll_data.get('output'))}")
            
        except Exception as e:
            error_msg = f"Error processing job: {str(e)}"
            print(f"Exception: {error_msg}")
            update_job_status(job_id, "FAILED", error_msg)
            post_to_client(domain_name, stage, conn_id, {"error": error_msg})
            continue

def update_job_status(job_id, status, error=None, model_url=None):
    try:
        expr = "SET #s = :s"
        ean = {"#s": "status"}
        eav = {":s": status}

        if error:
            expr += ", #e = :e"
            ean["#e"] = "error"
            eav[":e"] = error
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
        raise

def post_to_client(domain_name, stage, conn_id, message):
    """Send a message to a connected WebSocket client"""
    try:
        endpoint_url = f"https://{domain_name}/{stage}"
        print(f"Sending to client. URL: {endpoint_url}, ConnID: {conn_id}, Message: {json.dumps(message)}")
        
        client = boto3.client('apigatewaymanagementapi', endpoint_url=endpoint_url)
        
        response = client.post_to_connection(
            Data=json.dumps(message),
            ConnectionId=conn_id
        )
        print(f"Successfully sent message to client. Response: {json.dumps(response)}")
        return response
    except Exception as e:
        print(f"Error sending to client: {str(e)}")
        raise