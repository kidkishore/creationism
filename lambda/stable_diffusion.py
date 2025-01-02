import json
import os
import base64
import requests
from urllib.parse import quote

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

        # Call Stable Diffusion API (using the public demo API for this example)
        # Replace this URL with your actual Stable Diffusion API endpoint
        api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
        
        # Replace with your actual API key
        headers = {
            "Authorization": f"Bearer {os.environ.get('HUGGING_FACE_API_KEY')}"
        }

        # Make request to Stable Diffusion API
        response = requests.post(
            api_url,
            headers=headers,
            json={"inputs": prompt}
        )

        if response.status_code != 200:
            return {
                'statusCode': 500,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Access-Control-Allow-Methods': 'POST'
                },
                'body': json.dumps({'error': 'Failed to generate image'})
            }

        # Get the image data
        image_bytes = response.content
        
        # Upload to S3
        bucket_name = os.environ['BUCKET_NAME']
        file_name = f"generated/{prompt[:30]}_{context.aws_request_id}.png"
        
        import boto3
        s3 = boto3.client('s3')
        s3.put_object(
            Bucket=bucket_name,
            Key=file_name,
            Body=image_bytes,
            ContentType='image/png'
        )

        # Generate presigned URL for the uploaded image
        url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': file_name},
            ExpiresIn=3600  # URL expires in 1 hour
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

    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST'
            },
            'body': json.dumps({
                'error': str(e)
            })
        }