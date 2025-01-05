import os
import json
import time
import uuid
import boto3
import requests
import base64
import struct
import numpy as np
from io import BytesIO

dynamo = boto3.resource('dynamodb')
job_table = dynamo.Table(os.environ['JOB_TABLE'])
s3 = boto3.client('s3')

def truncate_error_msg(error_msg, max_length=1024):
    if len(error_msg) > max_length:
        return error_msg[:max_length] + "... (truncated)"
    return error_msg

def update_job_status(job_id, status, error=None, model_url=None):
    try:
        expr = "SET #s = :s"
        ean = {"#s": "status"}
        eav = {":s": status}

        if error:
            expr += ", #e = :e"
            ean["#e"] = "error"
            eav[":e"] = truncate_error_msg(error)
            
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
        if error and "ValidationException" in str(e):
            try:
                job_table.update_item(
                    Key={"job_id": job_id},
                    UpdateExpression="SET #s = :s, #e = :e",
                    ExpressionAttributeNames={"#s": "status", "#e": "error"},
                    ExpressionAttributeValues={
                        ":s": status,
                        ":e": truncate_error_msg("Error occurred. Check CloudWatch logs for details.", 256)
                    }
                )
            except Exception as e2:
                print(f"Second attempt to update job status failed: {str(e2)}")

def create_glb(vertices, faces=None, colors=None):
    """Convert mesh data to GLB format"""
    # Create binary buffer for vertex positions
    vertex_data = np.array(vertices, dtype=np.float32).tobytes()
    
    # Create binary buffer for indices if faces are provided
    index_data = b''
    if faces is not None:
        index_data = np.array(faces, dtype=np.uint16).tobytes()
    
    # Create binary buffer for colors if provided
    color_data = b''
    if colors is not None:
        color_data = np.array(colors, dtype=np.float32).tobytes()

    # Combine all buffer data
    buffer_data = vertex_data + index_data + color_data
    
    # Pad buffer to 4-byte alignment
    padding_length = (4 - (len(buffer_data) % 4)) % 4
    buffer_data += b'\x00' * padding_length

    # Create JSON chunk
    json_data = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{
            "primitives": [{
                "attributes": {
                    "POSITION": 1
                },
                "indices": 2 if faces is not None else None,
                "mode": 4  # TRIANGLES
            }]
        }],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": len(vertex_data),
                "target": 34962  # ARRAY_BUFFER
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,  # FLOAT
                "count": len(vertices),
                "type": "VEC3",
                "max": np.max(vertices, axis=0).tolist(),
                "min": np.min(vertices, axis=0).tolist()
            }
        ],
        "buffers": [{"byteLength": len(buffer_data)}]
    }

    if faces is not None:
        json_data["bufferViews"].append({
            "buffer": 0,
            "byteOffset": len(vertex_data),
            "byteLength": len(index_data),
            "target": 34963  # ELEMENT_ARRAY_BUFFER
        })
        json_data["accessors"].append({
            "bufferView": 1,
            "componentType": 5123,  # UNSIGNED_SHORT
            "count": len(faces) * 3,
            "type": "SCALAR"
        })

    if colors is not None:
        json_data["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"] = 3
        json_data["bufferViews"].append({
            "buffer": 0,
            "byteOffset": len(vertex_data) + len(index_data),
            "byteLength": len(color_data),
            "target": 34962  # ARRAY_BUFFER
        })
        json_data["accessors"].append({
            "bufferView": 2,
            "componentType": 5126,  # FLOAT
            "count": len(colors),
            "type": "VEC3"
        })

    json_str = json.dumps(json_data)
    json_buffer = json_str.encode()
    
    # Pad JSON to 4-byte alignment
    json_padding = (4 - (len(json_buffer) % 4)) % 4
    json_buffer += b' ' * json_padding

    # Create GLB header and chunks
    header = struct.pack('<4sII', b'glTF', 2, 12 + 8 + len(json_buffer) + 8 + len(buffer_data))
    json_header = struct.pack('<II', len(json_buffer), 0x4E4F534A)  # JSON chunk
    bin_header = struct.pack('<II', len(buffer_data), 0x004E4942)   # BIN chunk

    # Combine everything
    glb_data = header + json_header + json_buffer + bin_header + buffer_data
    return glb_data

def post_to_client(domain_name, stage, conn_id, message):
    try:
        endpoint_url = f"https://{domain_name}/{stage}"
        client = boto3.client('apigatewaymanagementapi', endpoint_url=endpoint_url)
        
        if 'meshData' in message:
            # Convert mesh data to GLB
            glb_data = create_glb(
                message['meshData']['vertices'],
                message['meshData'].get('faces'),
                message['meshData'].get('colors')
            )
            
            # Convert to base64
            base64_glb = base64.b64encode(glb_data).decode('utf-8')
            print(f"Total base64 GLB size: {len(base64_glb)} bytes")
            
            # Split into smaller chunks (8KB each)
            chunk_size = 8000  # Reduced from 25000 to 8000
            chunks = [base64_glb[i:i + chunk_size] for i in range(0, len(base64_glb), chunk_size)]
            total_chunks = len(chunks)
            print(f"Split into {total_chunks} chunks of {chunk_size} bytes each")
            
            # Send chunks sequentially
            for i, chunk in enumerate(chunks):
                chunk_message = {
                    'status': 'chunk',
                    'chunkIndex': i,
                    'totalChunks': total_chunks,
                    'chunk': chunk
                }
                message_json = json.dumps(chunk_message)
                print(f"Sending chunk {i+1}/{total_chunks}, size: {len(message_json)} bytes")
                
                try:
                    client.post_to_connection(
                        Data=message_json,
                        ConnectionId=conn_id
                    )
                    time.sleep(0.2)  # Increased delay between chunks
                except Exception as chunk_error:
                    print(f"Error sending chunk {i+1}: {str(chunk_error)}")
                    raise
            
            # Send completion message
            print("Sending completion message")
            client.post_to_connection(
                Data=json.dumps({
                    'status': 'completed',
                    'message': 'Model data transfer complete'
                }),
                ConnectionId=conn_id
            )
        else:
            if 'error' in message and isinstance(message['error'], str):
                message['error'] = truncate_error_msg(message['error'], 1024)
                
            message_json = json.dumps(message)
            print(f"Sending regular message, size: {len(message_json)} bytes")
            client.post_to_connection(
                Data=message_json,
                ConnectionId=conn_id
            )
            
        print(f"Successfully sent message to client")
        return True
    except Exception as e:
        print(f"Error sending to client: {str(e)}")
        raise

def lambda_handler(event, context):
    replicate_token = os.environ["REPLICATE_API_KEY"]
    for record in event["Records"]:
        try:
            body = json.loads(record["body"])
            job_id = body["job_id"]
            prompt = body["prompt"]
            conn_id = body["connectionId"]
            domain_name = body["domainName"]
            stage = body["stage"]

            print(f"Processing job {job_id} for prompt: {prompt}")

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

            if create_resp.status_code != 201:
                error_msg = f"Replicate API error: {create_resp.status_code}"
                update_job_status(job_id, "FAILED", error_msg)
                post_to_client(domain_name, stage, conn_id, {"error": error_msg})
                continue

            prediction = create_resp.json()
            prediction_id = prediction["id"]
            print(f"Created prediction {prediction_id}")

            while True:
                poll_resp = requests.get(
                    f"https://api.replicate.com/v1/predictions/{prediction_id}",
                    headers={"Authorization": f"Token {replicate_token}"}
                )
                prediction = poll_resp.json()
                status = prediction["status"]
                
                post_to_client(domain_name, stage, conn_id, {
                    "statusMessage": f"Still working... Current status: {status}"
                })

                if status == "succeeded":
                    output = prediction.get("output")
                    print(f"Prediction succeeded! Output: {json.dumps(output)}")
                    
                    if not output:
                        raise ValueError("Empty output received")

                    # Extract the json_file data
                    json_data = output.get('json_file')
                    if not json_data:
                        raise ValueError("No json_file in output")

                    post_to_client(domain_name, stage, conn_id, {
                        "meshData": json_data,
                        "status": "completed"
                    })
                    
                    update_job_status(job_id, "COMPLETED")
                    break
                elif status in ["failed", "canceled"]:
                    error_msg = f"Prediction {status}"
                    update_job_status(job_id, "FAILED", error_msg)
                    post_to_client(domain_name, stage, conn_id, {"error": error_msg})
                    break
                    
                time.sleep(3)

        except Exception as e:
            error_msg = f"Error processing job: {str(e)}"
            print(f"Exception in worker: {error_msg}")
            try:
                update_job_status(job_id, "FAILED", error_msg)
                post_to_client(domain_name, stage, conn_id, {"error": truncate_error_msg(error_msg)})
            except Exception as e2:
                print(f"Error sending failure notice: {str(e2)}")