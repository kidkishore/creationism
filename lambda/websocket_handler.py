import os
import json
import boto3
from boto3.dynamodb.conditions import Key

dynamo = boto3.resource('dynamodb')
connections_table = dynamo.Table(os.environ['CONNECTIONS_TABLE'])
job_table = dynamo.Table(os.environ['JOB_TABLE'])
sqs = boto3.client('sqs')

# We'll need the domain name + stage to post back to the client
# e.g. wss://xxxxxx.execute-api.us-east-1.amazonaws.com/prod
# We read these from your WebSocket requestContext
def lambda_handler(event, context):
    replicate_token = os.environ["REPLICATE_API_KEY"]
    print("Processing SQS messages...")  # Debug log
    
    for record in event["Records"]:
        body = json.loads(record["body"])
        print(f"Processing job: {json.dumps(body)}")  # Debug log
        
        job_id = body["job_id"]
        prompt = body["prompt"]
        conn_id = body["connectionId"]
        domain_name = body["domainName"]
        stage = body["stage"]

        # 1) Call Replicate to start
        print(f"Starting Replicate job with prompt: {prompt}")  # Debug log
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
        print(f"Replicate response status: {create_resp.status_code}")  # Debug log
        print(f"Replicate response: {create_resp.text}")  # Debug log
        
        if create_resp.status_code != 201:
            msg = f"Replicate error: {create_resp.text}"
            print(f"Error response from Replicate: {msg}")  # Debug log
            update_job_status(job_id, "FAILED", msg)
            post_to_client(domain_name, stage, conn_id, {"error": msg})
            continue

        prediction = create_resp.json()
        prediction_id = prediction["id"]
        status = prediction["status"]

        # 2) Poll for completion
        print(f"Polling for completion of prediction {prediction_id}")  # Debug log
        while status not in ['succeeded', 'failed', 'canceled']:
            time.sleep(3)
            poll_resp = requests.get(
                f"https://api.replicate.com/v1/predictions/{prediction_id}",
                headers={"Authorization": f"Token {replicate_token}"}
            )
            poll_data = poll_resp.json()
            status = poll_data["status"]
            print(f"Current status: {status}")  # Debug log

        if status != 'succeeded':
            msg = "Prediction failed or canceled."
            print(f"Job failed: {msg}")  # Debug log
            update_job_status(job_id, "FAILED", msg)
            post_to_client(domain_name, stage, conn_id, {"error": msg})
            continue


def on_connect(conn_id):
    # Store the new connection
    connections_table.put_item(Item={"connectionId": conn_id})
    return { "statusCode": 200, "body": "Connected." }

def on_disconnect(conn_id):
    # Remove the connection
    connections_table.delete_item(Key={"connectionId": conn_id})
    return { "statusCode": 200, "body": "Disconnected." }

def on_generate(conn_id, prompt, domain_name, stage):
    if not prompt:
        return send_to_client(domain_name, stage, conn_id, 
               {"error": "No prompt provided"})

    # Create a new job ID
    import uuid, time
    job_id = str(uuid.uuid4())
    job_table.put_item(Item={
        "job_id": job_id,
        "status": "PROCESSING",
        "prompt": prompt,
        "created_at": int(time.time())
    })

    # Send SQS message for the worker
    sqs.send_message(
        QueueUrl=os.environ['REPLICATE_QUEUE_URL'],
        MessageBody=json.dumps({
            "job_id": job_id,
            "prompt": prompt,
            "connectionId": conn_id,
            "domainName": domain_name,
            "stage": stage
        })
    )

    # Immediately tell user we started
    send_to_client(domain_name, stage, conn_id, {
        "statusMessage": f"Job {job_id} started. We'll notify you when it's ready."
    })

    return { "statusCode": 200, "body": "Job queued." }

def send_to_client(domain_name, stage, conn_id, message):
    # Initialize the API client with the endpoint
    api_client = boto3.client(
        'apigatewaymanagementapi',
        endpoint_url=f"https://{domain_name}/{stage}"
    )
    
    # Post the message to the connection
    response = api_client.post_to_connection(
        Data=json.dumps(message),
        ConnectionId=conn_id
    )
    return response


def post_to_client(domain_name, stage, conn_id, message):
    print(f"Attempting to post to client. Domain: {domain_name}, Stage: {stage}, ConnID: {conn_id}")  # Debug log
    try:
        endpoint_url = f"https://{domain_name}/{stage}"
        api_client = boto3.client('apigatewaymanagementapi', endpoint_url=endpoint_url)
        response = api_client.post_to_connection(
            Data=json.dumps(message),
            ConnectionId=conn_id
        )
        print(f"Successfully posted to client: {response}")  # Debug log
    except Exception as e:
        print(f"Error posting to client: {str(e)}")  # Debug log
        raise
