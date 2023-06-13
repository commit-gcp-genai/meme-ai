from typing import Dict, List, Union
import base64
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

def generate_image(
    project: str, location: str, instances: list, endpoint: str
):
    aiplatform.init(project=project, location=location)

    endpoint = aiplatform.Endpoint(endpoint)

    prediction = endpoint.predict(instances=instances)
    return prediction


# Run locally
if __name__ == "__main__":
    image_request = generate_image(
        project="860550299524",
        endpoint="8743606735144484864",
        location="europe-west1",
        instances=[{ "prompt": "A cat ridind a bycicle on the moon"}]
    )
    image_base64 = image_request.predictions[0]
    # Save the image to a file
    with open("/tmp/image.png", "wb") as fh:
        fh.write(base64.b64decode(image_base64))
    
    
