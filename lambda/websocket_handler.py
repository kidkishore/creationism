import os
import json
import boto3
import uuid
import time
from boto3.dynamodb.conditions import Key

# Initialize AWS services
dynamo = boto3.resource('dynamodb')
connections_table = dynamo.Table(os.environ['CONNECTIONS_TABLE'])
job_table = dynamo.Table(os.environ['JOB_TABLE'])
sqs = boto3.client('sqs')

def lambda_handler(event, context):
    """Main handler for WebSocket events"""
    print(f"Received event: {json.dumps(event)}")  # Debug log
    
    route_key = event.get("requestContext", {}).get("routeKey", "")
    connection_id = event.get("requestContext", {}).get("connectionId")
    
    if not connection_id:
        return {"statusCode": 400, "body": "No connection ID provided"}

    if route_key == "$connect":
        return on_connect(connection_id)
    elif route_key == "$disconnect":
        return on_disconnect(connection_id)
    elif route_key == "generate":
        body = json.loads(event.get("body", "{}"))
        prompt = body.get("prompt", "")
        domain_name = event["requestContext"].get("domainName")
        stage = event["requestContext"].get("stage")
        return on_generate(connection_id, prompt, domain_name, stage)
    else:
        return {"statusCode": 400, "body": f"Unknown route: {route_key}"}

def on_connect(conn_id):
    """Handle new WebSocket connections"""
    try:
        print(f"New connection: {conn_id}")  # Debug log
        connections_table.put_item(Item={"connectionId": conn_id})
        return {"statusCode": 200, "body": "Connected"}
    except Exception as e:
        print(f"Error in connect handler: {str(e)}")  # Debug log
        return {"statusCode": 500, "body": str(e)}

def on_disconnect(conn_id):
    """Handle WebSocket disconnections"""
    try:
        print(f"Disconnection: {conn_id}")  # Debug log
        connections_table.delete_item(Key={"connectionId": conn_id})
        return {"statusCode": 200, "body": "Disconnected"}
    except Exception as e:
        print(f"Error in disconnect handler: {str(e)}")  # Debug log
        return {"statusCode": 500, "body": str(e)}

def on_generate(conn_id, prompt, domain_name, stage):
    """Handle generation requests"""
    try:
        print(f"Generate request. Prompt: {prompt}, Connection: {conn_id}")
        
        if not prompt:
            return send_to_client(domain_name, stage, conn_id, 
                   {"error": "No prompt provided"})

        # Create job record
        job_id = str(uuid.uuid4())
        print(f"Created job ID: {job_id}")  # Debug log
        
        job_table.put_item(Item={
            "job_id": job_id,
            "status": "PENDING",
            "prompt": prompt,
            "connectionId": conn_id,
            "created_at": int(time.time())
        })
        print(f"Job record created in DynamoDB")  # Debug log

        # Queue job for worker
        message = {
            "job_id": job_id,
            "prompt": prompt,
            "connectionId": conn_id,
            "domainName": domain_name,
            "stage": stage
        }
        
        print(f"Queueing message to SQS: {json.dumps(message)}")  # Debug log
        sqs_response = sqs.send_message(
            QueueUrl=os.environ['REPLICATE_QUEUE_URL'],
            MessageBody=json.dumps(message)
        )
        print(f"SQS response: {json.dumps(sqs_response)}")  # Debug log

        # Notify client
        send_to_client(domain_name, stage, conn_id, {
            "status": "pending",
            "jobId": job_id,
            "message": f"Job {job_id} queued successfully"
        })

        return {"statusCode": 200, "body": "Job queued"}
        
    except Exception as e:
        print(f"Error in generate handler: {str(e)}")
        error_message = {"error": f"Internal error: {str(e)}"}
        try:
            send_to_client(domain_name, stage, conn_id, error_message)
        except:
            pass
        return {"statusCode": 500, "body": str(e)}

def send_to_client(domain_name, stage, conn_id, message):
    """Send a message to a connected WebSocket client"""
    try:
        endpoint_url = f"https://{domain_name}/{stage}"
        print(f"Sending to client. URL: {endpoint_url}, ConnID: {conn_id}, Message: {json.dumps(message)}")  # Debug log
        
        api_client = boto3.client('apigatewaymanagementapi', 
                                endpoint_url=endpoint_url)
        
        response = api_client.post_to_connection(
            Data=json.dumps(message),
            ConnectionId=conn_id
        )
        return response
    except Exception as e:
        print(f"Error sending to client: {str(e)}")  # Debug log
        raise