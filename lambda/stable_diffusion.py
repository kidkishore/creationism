import time
import requests
import json
import os
import boto3
import uuid
from datetime import datetime, timedelta

def lambda_handler(event, context):
    # Handle status check requests
    if event.get('queryStringParameters') and event.get('queryStringParameters').get('jobId'):
        return check_status(event)

    # Handle image generation requests
    try:
        body = json.loads(event.get('body', '{}'))
        prompt = body.get('text', '')

        if not prompt:
            return create_response(400, {'error': 'No text prompt provided'})

        # Get or create job ID
        job_id = body.get('jobId', str(uuid.uuid4()))
        is_retry = 'jobId' in body
        
        if not is_retry:
            # Store initial job status
            dynamodb = boto3.resource('dynamodb')
            table = dynamodb.Table(os.environ['TABLE_NAME'])
            expiry_time = int((datetime.now() + timedelta(days=1)).timestamp())
            table.put_item(Item={
                'job_id': job_id,
                'status': 'PROCESSING',
                'prompt': prompt,
                'created_at': int(time.time()),
                'expiry_time': expiry_time
            })

        # Try to generate image
        success = generate_image(prompt, job_id)
        
        if not success and not is_retry:
            # Schedule a retry by invoking this lambda again
            lambda_client = boto3.client('lambda')
            lambda_client.invoke(
                FunctionName=context.function_name,
                InvocationType='Event',  # async
                Payload=json.dumps({
                    'body': json.dumps({
                        'text': prompt,
                        'jobId': job_id
                    })
                })
            )

        return create_response(200, {
            'message': 'Image generation started',
            'jobId': job_id
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return create_response(500, {'error': str(e)})

def check_status(event):
    try:
        job_id = event['queryStringParameters']['jobId']
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table(os.environ['TABLE_NAME'])
        
        response = table.get_item(Key={'job_id': job_id})
        item = response.get('Item', {})
        
        if not item:
            return create_response(404, {'error': 'Job not found'})
            
        return create_response(200, {
            'status': item.get('status'),
            'imageUrl': item.get('image_url'),
            'error': item.get('error')
        })
        
    except Exception as e:
        return create_response(500, {'error': str(e)})

def generate_image(prompt, job_id):
    try:
        api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
        headers = {
            "Authorization": f"Bearer {os.environ.get('HUGGING_FACE_API_KEY')}"
        }

        response = requests.post(
            api_url,
            headers=headers,
            json={"inputs": prompt}
        )

        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table(os.environ['TABLE_NAME'])

        if response.status_code == 200:
            # Save image to S3
            image_bytes = response.content
            bucket_name = os.environ['BUCKET_NAME']
            file_name = f"generated/{prompt[:30]}_{job_id}.png"
            
            s3 = boto3.client('s3')
            s3.put_object(
                Bucket=bucket_name,
                Key=file_name,
                Body=image_bytes,
                ContentType='image/png'
            )

            url = s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': file_name},
                ExpiresIn=3600
            )

            # Update job status
            table.update_item(
                Key={'job_id': job_id},
                UpdateExpression='SET #status = :s, image_url = :u',
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={
                    ':s': 'COMPLETED',
                    ':u': url
                }
            )
            return True

        elif response.status_code == 503:
            # Model is loading, update status for retry
            table.update_item(
                Key={'job_id': job_id},
                UpdateExpression='SET #status = :s',
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={':s': 'QUEUED'}
            )
            return False

        else:
            raise Exception(f"Failed to generate image: {response.text}")

    except Exception as e:
        print(f"Error generating image: {str(e)}")
        table.update_item(
            Key={'job_id': job_id},
            UpdateExpression='SET #status = :s, error = :e',
            ExpressionAttributeNames={'#status': 'status'},
            ExpressionAttributeValues={
                ':s': 'FAILED',
                ':e': str(e)
            }
        )
        return False

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