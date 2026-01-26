---
name: add-provider
description: I provide a blueprint and set of instructions for integrating new LLM providers (e.g., OpenAI, Gemini, HuggingFace) into the UniInfer framework. I ensure consistent class implementation, registration protocols, and rich error handling for proxy compatibility.
license: MIT
compatibility: opencode
metadata:
  category: development
  workflow: integrations
  tool_standard: uniioai-v1
---

## What I do
- **Code Blueprinting**: I generate the boilerplate code needed for a new `ChatProvider` subclass.
- **Error Standardizing**: I guide you in implementing `map_provider_error` with status code and body extraction to ensure the `uniioai_proxy` relays detailed upstream errors.
- **Lifecycle Integration**: I provide the necessary registration steps for `ProviderFactory` and the package index.

## Instructions

1. **Implement the Class**: Create a new file in `uniinfer/providers/[provider_name].py`. Use the following pattern:
   ```python
   from typing import Dict, Any, Iterator, Optional, List
   import requests
   import json
   from ..core import ChatProvider, ChatCompletionRequest, ChatCompletionResponse, ChatMessage
   from ..errors import map_provider_error

   class MyProvider(ChatProvider):
       def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.example.com/v1", **kwargs):
           super().__init__(api_key)
           self.base_url = base_url

       def complete(self, request: ChatCompletionRequest, **kwargs) -> ChatCompletionResponse:
           try:
               # Your completion logic here
               pass
           except Exception as e:
               status_code = getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
               response_body = getattr(e.response, 'text', str(e)) if hasattr(e, 'response') else str(e)
               raise map_provider_error("myprovider", e, status_code=status_code, response_body=response_body)

       # Repeat similar try-except for stream_complete
   ```

2. **Register Globally**: 
   - Open `uniinfer/__init__.py`.
   - Import your class.
   - Run `ProviderFactory.register_provider("name", YourClass)`.
   - Add the class name to the `__all__` list.

3. **Verify Proxy Compatibility**: Use the following checklist:
   - Does it extract `status_code`?
   - Does it extract `response_body`?
   - Does it yield `ChatCompletionResponse` objects in streaming?

## Implementation Standards
- **Wait/Throttling**: If the provider has aggressive rate limits (like TU), implement a `_throttle` class method using `threading.Lock`.
- **Message Flattening**: Many APIs expect string content. Use a `_flatten_messages` helper to convert list-style contents to string while preserving VLM structures if applicable.

## Examples

### standard-openai-clone
If the provider uses an OpenAI-compatible SDK (like SambaNova or StepFun):
```python
from openai import OpenAI

class CompatibleProvider(ChatProvider):
    def __init__(self, api_key, **kwargs):
        super().__init__(api_key)
        self.client = OpenAI(api_key=api_key, base_url=kwargs.get("base_url"))

    def complete(self, request, **kwargs):
        try:
            # ... calls self.client.chat.completions.create ...
            pass
        except Exception as e:
            # Extract rich info for the proxy!
            raise map_provider_error("compatible", e, status_code=getattr(e, 'status_code', None))
```
