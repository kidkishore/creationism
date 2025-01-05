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
            
            print(f"Replicate API Response: {create_resp.status_code} - {create_resp.text}")
            
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
                time.sleep(3)
                poll_resp = requests.get(
                    f"https://api.replicate.com/v1/predictions/{prediction_id}",
                    headers={"Authorization": f"Token {replicate_token}"}
                )
                poll_data = poll_resp.json()
                status = poll_data["status"]
                
                post_to_client(domain_name, stage, conn_id, {
                    "statusMessage": f"Still working... Current status: {status}"
                })

            if status == 'succeeded':
                print(f"✓ Prediction succeeded! Full response: {json.dumps(poll_data)}")
                
                output = poll_data.get("output")
                print(f"Output from Replicate: {output}")
                
                if isinstance(output, dict):
                    json_file = output.get("json_file")
                elif isinstance(output, str):
                    json_file = output
                else:
                    json_file = None
                
                if not json_file:
                    msg = f"No JSON file in output. Output: {output}"
                    update_job_status(job_id, "FAILED", msg)
                    post_to_client(domain_name, stage, conn_id, {"error": msg})
                    continue

                # Download and process point cloud
                try:
                    file_resp = requests.get(json_file)
                    if file_resp.status_code != 200:
                        raise Exception(f"Failed to download point cloud. Status: {file_resp.status_code}")
                    
                    point_cloud_data = file_resp.json()
                    obj_content = convert_point_cloud_to_obj(point_cloud_data)
                    
                    # Store in S3
                    s3_key = f"generated-3d/{job_id}.obj"
                    bucket_name = os.environ["BUCKET_NAME"]
                    
                    s3.put_object(
                        Bucket=bucket_name,
                        Key=s3_key,
                        Body=obj_content,
                        ContentType="application/x-tgif"
                    )
                    
                    presigned_url = s3.generate_presigned_url(
                        "get_object",
                        Params={"Bucket": bucket_name, "Key": s3_key},
                        ExpiresIn=3600
                    )
                    
                    # Update status and notify client
                    update_job_status(job_id, "COMPLETED", model_url=presigned_url)
                    post_to_client(domain_name, stage, conn_id, {
                        "status": "completed",
                        "finalModelUrl": presigned_url,
                        "message": "Model generation complete!"
                    })
                    
                except Exception as e:
                    error_msg = f"Error processing point cloud: {str(e)}"
                    print(f"Error: {error_msg}")
                    update_job_status(job_id, "FAILED", error_msg)
                    post_to_client(domain_name, stage, conn_id, {"error": error_msg})
                    
            else:
                msg = f"Prediction failed or was canceled. Status: {status}"
                update_job_status(job_id, "FAILED", msg)
                post_to_client(domain_name, stage, conn_id, {"error": msg})

        except Exception as e:
            error_msg = f"Error processing job: {str(e)}"
            print(f"Exception: {error_msg}")
            update_job_status(job_id, "FAILED", error_msg)
            post_to_client(domain_name, stage, conn_id, {"error": error_msg})

def convert_point_cloud_to_obj(point_cloud_data):
    """Convert point cloud JSON data to OBJ format."""
    coords = point_cloud_data.get("coords", [])
    colors = point_cloud_data.get("colors", [])
    
    if not coords or len(coords) != len(colors):
        raise ValueError("Invalid point cloud data format")
    
    # Generate OBJ content
    obj_lines = []
    
    # Add vertices with colors
    for i, (coord, color) in enumerate(zip(coords, colors)):
        x, y, z = coord
        r, g, b = color
        # Write vertex position and color
        obj_lines.append(f"v {x} {y} {z} {r} {g} {b}")
        
        # Create point primitive
        obj_lines.append(f"p {i+1}")
    
    return "\n".join(obj_lines).encode('utf-8')

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