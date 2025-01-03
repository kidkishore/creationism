import time
import requests
import json
import os
import boto3

def lambda_handler(event, context):
    try:
        # Parse the incoming request body
        body = json.loads(event.get('body', '{}'))
        prompt = body.get('text', '')

        if not prompt:
            return {
                'statusCode': 400,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Access-Control-Allow-Methods': 'POST'
                },
                'body': json.dumps({'error': 'No text prompt provided'})
            }

        api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
        headers = {
            "Authorization": f"Bearer {os.environ.get('HUGGING_FACE_API_KEY')}"
        }

        max_retries = 5
        retry_delay = 30  # seconds

        for attempt in range(max_retries):
            response = requests.post(
                api_url,
                headers=headers,
                json={"inputs": prompt}
            )

            if response.status_code == 200:
                # Save image to S3
                image_bytes = response.content
                bucket_name = os.environ['BUCKET_NAME']
                file_name = f"generated/{prompt[:30]}_{context.aws_request_id}.png"
                s3 = boto3.client('s3')
                s3.put_object(
                    Bucket=bucket_name,
                    Key=file_name,
                    Body=image_bytes,
                    ContentType='image/png'
                )

                # Generate presigned URL
                url = s3.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': bucket_name, 'Key': file_name},
                    ExpiresIn=3600
                )

                return {
                    'statusCode': 200,
                    'headers': {
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Headers': 'Content-Type',
                        'Access-Control-Allow-Methods': 'POST'
                    },
                    'body': json.dumps({
                        'message': 'Image generated successfully',
                        'imageUrl': url
                    })
                }

            elif response.status_code == 503:
                error_data = response.json()
                estimated_time = error_data.get("estimated_time", retry_delay)
                print(f"Model loading, retrying in {estimated_time} seconds...")
                time.sleep(estimated_time)

            else:
                return {
                    'statusCode': 500,
                    'headers': {
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Headers': 'Content-Type',
                        'Access-Control-Allow-Methods': 'POST'
                    },
                    'body': json.dumps({'error': 'Failed to generate image', 'details': response.text})
                }

        return {
            'statusCode': 503,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST'
            },
            'body': json.dumps({'error': 'Model loading timed out after retries'})
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST'
            },
            'body': json.dumps({'error': str(e)})
        }
