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

        # 1) Call Replicate to start
        create_resp = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers={
                "Authorization": f"Token {replicate_token}",
                "Content-Type": "application/json"
            },
            json={
                "version": "1a4da7adf0bc84cd786c1df41c02db3097d899f5c159f5fd5814a11117bdf02b",
                "input": { "prompt": prompt }
            }
        )
        if create_resp.status_code != 201:
            # Mark job as failed, notify user
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

        if status != 'succeeded':
            msg = "Prediction failed or canceled."
            update_job_status(job_id, "FAILED", msg)
            post_to_client(domain_name, stage, conn_id, {"error": msg})
            continue

        # 3) Download & store result in S3
        print(f"Full Replicate response: {json.dumps(poll_data, indent=2)}")  # Debug log
        output = poll_data.get("output")
        print(f"Output value: {output}")  # Debug log
        
        if output is None:
            msg = f"No output from Replicate. Full response: {json.dumps(poll_data)}"
            update_job_status(job_id, "FAILED", msg)
            post_to_client(domain_name, stage, conn_id, {"error": msg})
            continue
            
        if isinstance(output, list):
            if not output:
                msg = "Replicate returned empty list"
                update_job_status(job_id, "FAILED", msg)
                post_to_client(domain_name, stage, conn_id, {"error": msg})
                continue
            model_url = output[0]
        elif isinstance(output, str):
            model_url = output
        else:
            msg = f"Unexpected output format from Replicate: {type(output)}"
            update_job_status(job_id, "FAILED", msg)
            post_to_client(domain_name, stage, conn_id, {"error": msg})
            continue

        file_resp = requests.get(model_url)
        if file_resp.status_code != 200:
            msg = "Failed to download 3D model"
            update_job_status(job_id, "FAILED", msg)
            post_to_client(domain_name, stage, conn_id, {"error": msg})
            continue

        s3_key = f"generated-3d/{job_id}.obj"
        bucket_name = os.environ["BUCKET_NAME"]
        s3.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=file_resp.content,
            ContentType="text/plain"
        )
        presigned_url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": s3_key},
            ExpiresIn=3600
        )

        # 4) Update job status
        update_job_status(job_id, "COMPLETED", None, model_url=presigned_url)

        # 5) Notify user with final URL
        post_to_client(domain_name, stage, conn_id,
                       {"finalModelUrl": presigned_url})


def update_job_status(job_id, status, error=None, model_url=None):
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


def post_to_client(domain_name, stage, conn_id, message):
    endpoint_url = f"https://{domain_name}/{stage}"
    client = boto3.client('apigatewaymanagementapi', endpoint_url=endpoint_url)
    client.post_to_connection(
        Data=json.dumps(message),
        ConnectionId=conn_id
    )