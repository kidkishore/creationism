import os
import json
import boto3
from boto3.dynamodb.conditions import Key

dynamo = boto3.resource('dynamodb')
connections_table = dynamo.Table(os.environ['CONNECTIONS_TABLE'])
job_table = dynamo.Table(os.environ['JOB_TABLE'])
sqs = boto3.client('sqs')
api_client = boto3.client('apigatewaymanagementapi')

# We'll need the domain name + stage to post back to the client
# e.g. wss://xxxxxx.execute-api.us-east-1.amazonaws.com/prod
# We read these from your WebSocket requestContext
def lambda_handler(event, context):
    route_key = event.get("requestContext", {}).get("routeKey", "")
    connection_id = event["requestContext"].get("connectionId")

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
        return { "statusCode": 200, "body": "Unknown route" }

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
    endpoint_url = f"https://{domain_name}/{stage}"
    response = api_client.post_to_connection(
        Data=json.dumps(message),
        ConnectionId=conn_id,
        EndpointUrl=endpoint_url
    )
    return response
