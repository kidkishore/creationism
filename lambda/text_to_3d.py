import json
import os
import time
import uuid
import boto3
import requests

def lambda_handler(event, context):
    try:
        body = json.loads(event.get('body', '{}'))
        prompt = body.get('text')
        if not prompt:
            return create_response(400, {'error': 'No text prompt provided'})

        job_id = body.get('jobId', str(uuid.uuid4()))
        table = boto3.resource('dynamodb').Table(os.environ['TABLE_NAME'])
        expiry_time = int(time.time()) + 86400
        table.put_item(Item={
            'job_id': job_id,
            'status': 'PROCESSING',
            'prompt': prompt,
            'created_at': int(time.time()),
            'expiry_time': expiry_time
        })

        replicate_token = os.environ['REPLICATE_API_KEY']
        # Use Point-E version here:
        model_version = "cjwbw/point-e:1a4da7ad"

        # Create prediction
        prediction_resp = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers={
                "Authorization": f"Token {replicate_token}",
                "Content-Type": "application/json"
            },
            json={
                "version": model_version,
                "input": {
                    "prompt": prompt
                }
            }
        )
        if prediction_resp.status_code != 201:
            return handle_failure(table, job_id, prediction_resp.json().get('detail', 'Replicate error'))

        prediction_data = prediction_resp.json()
        prediction_id = prediction_data['id']
        status = prediction_data['status']

        # Poll until complete
        while status not in ['succeeded', 'failed', 'canceled']:
            time.sleep(3)
            poll_resp = requests.get(
                f"https://api.replicate.com/v1/predictions/{prediction_id}",
                headers={"Authorization": f"Token {replicate_token}"}
            )
            poll_data = poll_resp.json()
            status = poll_data['status']

        if status != 'succeeded':
            return handle_failure(table, job_id, poll_data.get('error', 'Prediction failed'))

        output = poll_data.get('output', [])
        if not output or not isinstance(output, list):
            return handle_failure(table, job_id, 'No valid output from Replicate')

        # Assume the first URL is the 3D file
        model_url = output[0]  
        if not model_url.startswith("http"):
            return handle_failure(table, job_id, 'Invalid model URL')

        # Download file
        file_resp = requests.get(model_url)
        if file_resp.status_code != 200:
            return handle_failure(table, job_id, 'File download failed')

        # Save to S3
        s3 = boto3.client('s3')
        bucket_name = os.environ['BUCKET_NAME']
        file_ext = ".obj"  # Adjust if Point-E returns something else
        s3_key = f"generated-3d/{job_id}{file_ext}"
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

        table.update_item(
            Key={'job_id': job_id},
            UpdateExpression='SET #s = :s, model_url = :u',
            ExpressionAttributeNames={'#s': 'status'},
            ExpressionAttributeValues={':s': 'COMPLETED', ':u': presigned_url}
        )

        return create_response(200, {'jobId': job_id, 'modelUrl': presigned_url})

    except Exception as e:
        return create_response(500, {'error': str(e)})


def handle_failure(table, job_id, msg):
    table.update_item(
        Key={'job_id': job_id},
        UpdateExpression='SET #s = :s, #e = :e',
        ExpressionAttributeNames={
            '#s': 'status',
            '#e': 'error'  # map '#e' to the actual attribute name 'error'
        },
        ExpressionAttributeValues={
            ':s': 'FAILED',
            ':e': msg
        }
    )
    return create_response(500, {'error': msg})


def create_response(status_code, body):
    return {
        'statusCode': status_code,
        'headers': {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key',
            'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
        },
        'body': json.dumps(body)
    }
