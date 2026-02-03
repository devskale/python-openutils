import base64
import os

image_path = "examples/image.jpg"
if not os.path.exists(image_path):
    print("Error: Image not found")
    exit(1)

with open(image_path, "rb") as f:
    b64_data = base64.b64encode(f.read()).decode("utf-8")

# data URI
data_uri = f"data:image/jpeg;base64,{b64_data}"

# Construct curl command
# Using a shorter prompt to speed it up if possible
print("curl http://localhost:8123/v1/chat/completions \\")
print("  -H \"Content-Type: application/json\" \\")
print("  -H \"Authorization: Bearer test\" \\") # Token doesn't matter much if credgoo is working in proxy
print(f"  -d '{{\"model\": \"tu@glm-4.5v-106b\", \"messages\": [{{\"role\": \"user\", \"content\": [{{\"type\": \"text\", \"text\": \"Describe this\"}}, {{\"type\": \"image_url\", \"image_url\": {{\"url\": \"{data_uri}\"}}}}]}}], \"stream\": true}}'")
