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

        # 1) Call Replicate to start - using Point-E model
        create_resp = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers={
                "Authorization": f"Token {replicate_token}",
                "Content-Type": "application/json"
            },
            json={
                "version": "b96a2f33cc8e33e0fd4c28c52c50b0db8d9d220389658601b4976d4eb5c6847a",  # Point-E model version
                "input": {
                    "prompt": prompt,
                    "output_format": "json_file"
                }
            }
        )
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
            msg = "Prediction failed or canceled."
        else:
            print(f"✓ Prediction succeeded! Proceeding to handle output")
            update_job_status(job_id, "FAILED", msg)
            post_to_client(domain_name, stage, conn_id, {"error": msg})
            continue

        # 3) Handle the JSON point cloud output
        print(f"Full Replicate response: {json.dumps(poll_data, indent=2)}")  # Debug log
        output = poll_data.get("output")
        print(f"Output type: {type(output)}")
        print(f"Output content: {output}")
        
        # Handle different output structures
        if isinstance(output, dict):
            json_file = output.get("json_file")
        elif isinstance(output, str):
            # If output is directly a URL
            json_file = output
        else:
            json_file = None
            
        print(f"Extracted json_file: {json_file}")
        
        if not json_file:
            msg = f"No JSON file in output. Full response: {json.dumps(poll_data)}"
            update_job_status(job_id, "FAILED", msg)
            post_to_client(domain_name, stage, conn_id, {"error": msg})
            continue

        # Download the JSON point cloud data
        try:
            file_resp = requests.get(json_file)
            print(f"File response status: {file_resp.status_code}")
            print(f"File response content: {file_resp.text[:500]}...")  # Print first 500 chars of response
            
            if file_resp.status_code != 200:
                msg = f"Failed to download point cloud data from {json_file}. Status: {file_resp.status_code}"
                update_job_status(job_id, "FAILED", msg)
                post_to_client(domain_name, stage, conn_id, {"error": msg})
                continue
        except Exception as e:
            msg = f"Exception while downloading point cloud data: {str(e)}"
            update_job_status(job_id, "FAILED", msg)
            post_to_client(domain_name, stage, conn_id, {"error": msg})
            continue

        # Convert point cloud data to OBJ format
        try:
            point_cloud_data = file_resp.json()
            obj_content = convert_point_cloud_to_obj(point_cloud_data)
        except Exception as e:
            msg = f"Failed to convert point cloud to OBJ: {str(e)}"
            update_job_status(job_id, "FAILED", msg)
            post_to_client(domain_name, stage, conn_id, {"error": msg})
            continue

        # Store the converted OBJ file in S3
        s3_key = f"generated-3d/{job_id}.obj"
        bucket_name = os.environ["BUCKET_NAME"]
        s3.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=obj_content,
            ContentType="application/x-tgif"  # Proper MIME type for .obj files
        )
        presigned_url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": s3_key},
            ExpiresIn=3600
        )

        # 4) Update job status
        update_job_status(job_id, "COMPLETED", None, model_url=presigned_url)

        # 5) Notify user with final URL
        try:
            print(f"Attempting to send final URL to client: {presigned_url}")
            final_message = {
                "status": "completed",
                "finalModelUrl": presigned_url,
                "message": "Model generation complete!"
            }
            post_to_client(domain_name, stage, conn_id, final_message)
            print(f"Successfully sent final message to client: {json.dumps(final_message)}")
        except Exception as e:
            print(f"Error sending final URL to client: {str(e)}")
            # Try one more time with just the URL
            try:
                post_to_client(domain_name, stage, conn_id, {"finalModelUrl": presigned_url})
            except Exception as e2:
                print(f"Second attempt to send URL failed: {str(e2)}")

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
    """Send a message to a connected WebSocket client"""
    try:
        endpoint_url = f"https://{domain_name}/{stage}"
        print(f"Sending to client. URL: {endpoint_url}, ConnID: {conn_id}, Message: {json.dumps(message)}")  # Debug log
        
        client = boto3.client('apigatewaymanagementapi', endpoint_url=endpoint_url)
        
        response = api_client.post_to_connection(
            Data=json.dumps(message),
            ConnectionId=conn_id
        )
        print(f"Successfully sent message to client. Response: {response}")
        return response
    except Exception as e:
        print(f"Error sending to client: {str(e)}")  # Debug log
        raise