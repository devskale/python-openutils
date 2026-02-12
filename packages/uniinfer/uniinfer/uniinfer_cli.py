# Import uniinfer components
from importlib.metadata import version  # Changed from pkg_resources
from uniinfer.examples.providers_config import PROVIDER_CONFIGS
from uniinfer import (
    ChatMessage,
    ChatCompletionRequest,
    ProviderFactory,
    EmbeddingRequest,
    EmbeddingProviderFactory
)
from credgoo import get_api_key
import argparse
import random
import time
import base64
import requests
# Cloudflare API Details
from dotenv import load_dotenv
import os
# load_dotenv(verbose=True, override=True)
# Load environment variables from .env file
dotenv_path = os.path.join(os.getcwd(), '.env')  # Explicitly check current dir
# Add verbose=True and override=True
found_dotenv = load_dotenv(dotenv_path=dotenv_path,
                           verbose=True, override=True)

# print(f"DEBUG: Attempted to load .env from: {dotenv_path}")  # Debug print
# print(f"DEBUG: .env file found and loaded: {found_dotenv}")  # Debug print


def _resolve_credgoo_service(provider: str) -> str:
    aliases = {
        "zai-coding": "zai-code",
    }
    return aliases.get(provider, provider)


def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description='UniInfer example script')
    parser.add_argument('-l', '--list-providers', '--list', '--list-provides', action='store_true',
                        help='List available providers')
    parser.add_argument('--list-models', action='store_true',
                        help='List available models for the specified provider or all providers when combined with --list-providers')
    parser.add_argument('--embed', action='store_true',
                        help='Use embedding instead of chat completion')
    parser.add_argument('--embed-text', type=str, action='append',
                        help='Text to embed (can be used multiple times)')
    parser.add_argument('--embed-file', type=str,
                        help='File containing text to embed (one text per line)')
    parser.add_argument('--generate-image', action='store_true',
                        help='Generate an image instead of chat completion')
    parser.add_argument('--prompt', type=str,
                        help='Prompt for image generation')
    parser.add_argument('--size', type=str, default='512x512',
                        help='Image size (e.g., 512x512, 1024x1024)')
    parser.add_argument('--output', type=str,
                        help='Output file path for generated image/audio')
    parser.add_argument('--seed', type=int,
                        help='Random seed for image generation')
    parser.add_argument('--tts', action='store_true',
                        help='Generate speech from text (text-to-speech)')
    parser.add_argument('--tts-text', type=str,
                        help='Text to convert to speech')
    parser.add_argument('--voice', type=str,
                        help='Voice to use for TTS (e.g., af_alloy)')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Speed of generated speech (0.25 to 4.0, default: 1.0)')
    parser.add_argument('--instructions', type=str,
                        help='Additional instructions for TTS model')
    parser.add_argument('--stt', action='store_true',
                        help='Transcribe audio to text (speech-to-text)')
    parser.add_argument('--audio-file', type=str,
                        help='Audio file to transcribe')
    parser.add_argument('--language', type=str,
                        help='Language of the audio (ISO-639-1 format, e.g., en, de)')
    parser.add_argument('-p', '--provider', type=str, default='stepfun',
                        help='Specify which provider to use')
    parser.add_argument('-q', '--query', type=str,
                        help='Specify the query to send to the provider')
    parser.add_argument('-m', '--model', type=str,
                        help='Specify which model to use')
    parser.add_argument('-f', '--file', type=str,
                        help='Specify a file to use as context')
    parser.add_argument('-t', '--tokens', type=int, default=4000,
                        help='Specify token limit for file context (default: 4000)')
    parser.add_argument('--max-tokens', type=int,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--tools-file', type=str,
                        help='Path to a JSON file containing tool definitions (object or array)')
    parser.add_argument('--tool-choice', type=str,
                        help='Tool choice preference (e.g., "auto", "none", or JSON object)')
    parser.add_argument('--encryption-key', type=str,
                        help='Specify the CREDGOO encryption key')
    parser.add_argument('--bearer-token', type=str,
                        help='Specify the CREDGOO bearer token')
    parser.add_argument('--image', type=str,
                        help='Image file path or URL for multimodal models')
    parser.add_argument('--version', action='version',
                        version='%(prog)s ' + version('uniinfer'),
                        help="Show program's version number and exit")

    args = parser.parse_args()

    # Retrieve credentials: prioritize CLI args, then environment variables
    credgoo_encryption_token = args.encryption_key or os.getenv(
        'CREDGOO_ENCRYPTION_KEY')
    credgoo_api_token = args.bearer_token or os.getenv('CREDGOO_BEARER_TOKEN')
    _bearer_token = f"{credgoo_api_token}@{credgoo_encryption_token}" if credgoo_api_token and credgoo_encryption_token else None

#    if not credgoo_api_token or not credgoo_encryption_token:
#        print("Error: CREDGOO_ENCRYPTION_KEY or CREDGOO_BEARER_TOKEN not found.")
#        print("Please provide them either via command-line arguments (--encryption-key, --bearer-token) or environment variables.")
#        return
    provider = args.provider
    credgoo_service = _resolve_credgoo_service(provider)
    retrieved_api_key = get_api_key(
        service=credgoo_service,
        encryption_key=credgoo_encryption_token,
        bearer_token=credgoo_api_token,)

    if args.list_providers and args.list_models:
        providers = ProviderFactory.list_providers()
        for provider in providers:
            try:
                credgoo_service = _resolve_credgoo_service(provider)
                provider_class = ProviderFactory.get_provider_class(provider)
                retrieved_api_key = get_api_key(
                    service=credgoo_service,
                    encryption_key=credgoo_encryption_token,
                    bearer_token=credgoo_api_token,)
                # models = ProviderFactory.list_models(
                #    provider=provider,
                #    api_key=retrieved_api_key,
                #    **({} if provider not in ['cloudflare', 'ollama'] else PROVIDER_CONFIGS[provider].get('extra_params', {}))
                # )
                models = provider_class.list_models(
                    api_key=retrieved_api_key,
                    **({} if provider not in ['cloudflare', 'ollama'] else PROVIDER_CONFIGS[provider].get('extra_params', {}))
                )
                print(f"\nAvailable models for {provider}:")
                for model in models:
                    print(f"- {model}")
            except Exception as e:
                print(f"\nError listing models for {provider}: {str(e)}")
        return

    if args.list_providers:
        providers = ProviderFactory.list_providers()
        print("Available providers:")
        for provider in providers:
            print(f"- {provider}")
        return

    if args.list_models:
        try:
            provider_class = ProviderFactory.get_provider_class(args.provider)
            models = provider_class.list_models(
                api_key=retrieved_api_key,
                **({} if args.provider not in ['cloudflare', 'ollama'] else PROVIDER_CONFIGS[args.provider].get('extra_params', {}))
            )

            print(f"Available models for {args.provider}:")
            for model in models:
                print(f"- {model}")
            return
        except Exception as e:
            print(f"Error listing models for {args.provider}: {str(e)}")
            return

    # Handle image generation requests
    if args.generate_image:
        if not args.prompt:
            print("Error: --prompt is required when using --generate-image")
            return

        # Only TU provider supports image generation currently
        if provider != 'tu':
            print(
                f"Error: Image generation is currently only supported for 'tu' provider, not '{provider}'")
            return

        model = args.model if args.model else 'flux-schnell'
        prompt = args.prompt
        size = args.size
        seed = args.seed if args.seed else random.randint(1, 1000000)

        print(f"Generating image using {provider}@{model}")
        print(f"Prompt: {prompt}")
        print(f"Size: {size}")
        print(f"Seed: {seed}")

        try:
            # Get API key using credgoo
            api_key = get_api_key(
                service=credgoo_service,
                encryption_key=credgoo_encryption_token,
                bearer_token=credgoo_api_token
            )

            if not api_key:
                print("Error: Could not retrieve API key for TU provider")
                return

            # Call TU provider's image generation endpoint directly
            tu_endpoint = "https://aqueduct.ai.datalab.tuwien.ac.at/v1/images/generations"

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

            payload = {
                "model": model,
                "prompt": prompt,
                "n": 1,
                "size": size
            }

            response = requests.post(
                tu_endpoint, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()

            # Save the image
            if data.get("data") and len(data["data"]) > 0:
                b64_json = data["data"][0].get("b64_json")
                url = data["data"][0].get("url")

                if b64_json:
                    image_data = base64.b64decode(b64_json)
                elif url:
                    # Fetch from URL if only URL is provided
                    img_response = requests.get(url, timeout=60)
                    img_response.raise_for_status()
                    image_data = img_response.content
                else:
                    print("Error: No image data in response")
                    return

                # Determine output filename
                if args.output:
                    output_path = args.output
                    # Create directory if it doesn't exist
                    output_dir = os.path.dirname(output_path)
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)
                else:
                    # Generate filename from prompt
                    safe_prompt = "".join(c if c.isalnum() or c in (
                        ' ', '-', '_') else '_' for c in prompt)[:50]
                    output_path = f"{provider}_{model}_{safe_prompt}_{size}_{seed}.png".replace(
                        ' ', '_')

                # Save the image
                with open(output_path, 'wb') as f:
                    f.write(image_data)

                print(f"\n✓ Image saved to: {output_path}")
                print(f"  Model: {model}")
                print(f"  Size: {len(image_data)} bytes")
            else:
                print("Error: No image generated")

        except requests.exceptions.HTTPError as e:
            print(
                f"Error generating image: HTTP {e.response.status_code if e.response else 'error'}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Error generating image: {e}")
        except Exception as e:
            print(f"Error: {e}")
        return

    # Handle TTS (text-to-speech) requests
    if args.tts:
        if not args.tts_text:
            print("Error: --tts-text is required when using --tts")
            return

        # Only TU provider supports TTS currently
        if provider != 'tu':
            print(
                f"Error: TTS is currently only supported for 'tu' provider, not '{provider}'")
            return

        model = args.model if args.model else 'kokoro'
        tts_text = args.tts_text
        voice = args.voice
        speed = args.speed
        instructions = args.instructions

        print(f"Generating speech using {provider}@{model}")
        print(f"Text: {tts_text}")
        if voice:
            print(f"Voice: {voice}")
        print(f"Speed: {speed}")

        try:
            from uniinfer import TTSRequest
            from uniinfer.providers.tu_tts import TuAITTSProvider

            # Get API key using credgoo
            api_key = get_api_key(
                service=credgoo_service,
                encryption_key=credgoo_encryption_token,
                bearer_token=credgoo_api_token
            )

            if not api_key:
                print("Error: Could not retrieve API key for TU provider")
                return

            # Create TTS provider
            tts_provider = TuAITTSProvider(api_key=api_key)

            # Create TTS request
            tts_request = TTSRequest(
                input=tts_text,
                model=model,
                voice=voice,
                speed=speed,
                instructions=instructions
            )

            # Generate speech
            response = tts_provider.generate_speech(tts_request)

            # Determine output filename
            if args.output:
                output_path = args.output
                # Create directory if it doesn't exist
                output_dir = os.path.dirname(output_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
            else:
                # Generate filename from text
                _safe_text = "".join(c if c.isalnum() or c in (
                    ' ', '-', '_') else '_' for c in tts_text)[:50]
                output_path = "cli_test.mp3"

            # Save the audio
            with open(output_path, 'wb') as f:
                f.write(response.audio_content)

            print(f"\n✓ Audio saved to: {output_path}")
            print(f"  Model: {model}")
            print(f"  Size: {len(response.audio_content)} bytes")
            print(f"  Content-Type: {response.content_type}")

        except Exception as e:
            print(f"Error generating speech: {e}")
            import traceback
            traceback.print_exc()
        return

    # Handle STT (speech-to-text) requests
    if args.stt:
        if not args.audio_file:
            print("Error: --audio-file is required when using --stt")
            return

        if not os.path.exists(args.audio_file):
            print(f"Error: Audio file not found: {args.audio_file}")
            return

        # Only TU provider supports STT currently
        if provider != 'tu':
            print(
                f"Error: STT is currently only supported for 'tu' provider, not '{provider}'")
            return

        model = args.model if args.model else 'whisper-large'
        audio_file = args.audio_file
        language = args.language

        print(f"Transcribing audio using {provider}@{model}")
        print(f"Audio file: {audio_file}")
        if language:
            print(f"Language: {language}")

        try:
            from uniinfer import STTRequest
            from uniinfer.providers.tu_stt import TuAISTTProvider

            # Get API key using credgoo
            api_key = get_api_key(
                service=credgoo_service,
                encryption_key=credgoo_encryption_token,
                bearer_token=credgoo_api_token
            )

            if not api_key:
                print("Error: Could not retrieve API key for TU provider")
                return

            # Create STT provider
            stt_provider = TuAISTTProvider(api_key=api_key)

            # Create STT request
            stt_request = STTRequest(
                file=audio_file,
                model=model,
                language=language,
                response_format="verbose_json"
            )

            # Transcribe audio
            response = stt_provider.transcribe(stt_request)

            print("\n=== Transcription ===")
            print(f"Text: {response.text}")
            print(f"\nModel: {response.model}")
            if response.language:
                print(f"Language: {response.language}")
            if response.duration:
                print(f"Duration: {response.duration:.2f} seconds")

            if response.segments:
                print(f"\n=== Segments ({len(response.segments)}) ===")
                # Show first 5 segments
                for i, segment in enumerate(response.segments[:5]):
                    print(f"\nSegment {i+1}:")
                    if 'start' in segment and 'end' in segment:
                        print(
                            f"  Time: {segment['start']:.2f}s - {segment['end']:.2f}s")
                    if 'text' in segment:
                        print(f"  Text: {segment['text']}")
                if len(response.segments) > 5:
                    print(
                        f"\n... and {len(response.segments) - 5} more segments")

        except Exception as e:
            print(f"Error transcribing audio: {e}")
            import traceback
            traceback.print_exc()
        return

    # Initialize the provider factory
    # for that i need credgoo api token and credgoo encryption key
    uni = ProviderFactory().get_provider(
        name=provider,
        api_key=retrieved_api_key,
        **({} if provider not in ['cloudflare', 'ollama'] else PROVIDER_CONFIGS[provider].get('extra_params', {}))
    )

    # Handle embedding requests
    if args.embed:
        texts_to_embed = []

        # Add texts from --embed-text arguments
        if args.embed_text:
            texts_to_embed.extend(args.embed_text)

        # Add texts from --embed-file
        if args.embed_file:
            try:
                with open(args.embed_file, 'r', encoding='utf-8') as f:
                    file_texts = [line.strip() for line in f if line.strip()]
                    texts_to_embed.extend(file_texts)
            except Exception as e:
                print(f"Error reading embed file: {e}")
                return

        if not texts_to_embed:
            print("Error: --embed-text or --embed-file is required when using --embed")
            return

        try:
            embedding_provider = EmbeddingProviderFactory().get_provider(
                name=provider,
                api_key=retrieved_api_key,
                **({} if provider not in ['cloudflare', 'ollama'] else PROVIDER_CONFIGS[provider].get('extra_params', {}))
            )

            model = args.model if args.model else PROVIDER_CONFIGS[provider]['default_model']
            print(
                f"Embedding {len(texts_to_embed)} text(s) using {provider}@{model}")

            # Create embedding request
            request = EmbeddingRequest(
                input=texts_to_embed,
                model=model
            )

            # Make the request
            response = embedding_provider.embed(request)

            print("\n=== Embedding Response ===")
            print(f"Model: {response.model}")
            print(f"Provider: {response.provider}")
            print(f"Number of embeddings: {len(response.data)}")
            print(f"Usage: {response.usage}")

            for i, embedding_data in enumerate(response.data):
                print(f"\nText {i+1}: '{texts_to_embed[i]}'")
                print(
                    f"Embedding dimensions: {len(embedding_data['embedding'])}")
                print(f"First 5 values: {embedding_data['embedding'][:5]}")
                if len(embedding_data['embedding']) > 5:
                    print(f"Last 5 values: {embedding_data['embedding'][-5:]}")

        except Exception as e:
            print(f"Error creating embeddings: {e}")
        return

    # List of machine learning topics in German
    ml_topics = [
        "Transformer in maschinellem Lernen",
        "Neuronale Netze",
        "Convolutional Neural Networks (CNNs)",
        "Recurrent Neural Networks (RNNs)",
        "Support Vector Machines (SVMs)",
        "Entscheidungsbäume",
        "Random Forests",
        "Gradient Boosting",
        "Unüberwachtes Lernen",
        "Bestärkendes Lernen",
        "Natural Language Processing (NLP)",
        "Computer Vision",
        "Generative Adversarial Networks (GANs)",
        "Transfer Learning",
        "Anomalie-Erkennung",
        "Zeitreihenanalyse",
        "Empfehlungssysteme",
        "Clustering-Algorithmen",
        "Deep Reinforcement Learning",
        "Federated Learning"
    ]

    prompt = args.query if args.query else f"Erkläre mir bitte {random.choice(ml_topics)} in einfachen Worten und auf deutsch."

    if args.file:
        try:
            with open(args.file, 'r') as f:
                file_content = f.read()
                # Simple token estimation (4 chars ~ 1 token)
                max_chars = args.tokens * 4
                if len(file_content) > max_chars:
                    file_content = file_content[:max_chars]
                prompt = f"File context: {file_content}\n\nUser query: {prompt}"
        except Exception as e:
            print(f"Error reading file: {e}")

    model = args.model if args.model else PROVIDER_CONFIGS[provider]['default_model']
    print(
        f"Prompt: {prompt} ( {provider}@{model} )")
    # Create chat messages
    content = prompt
    if args.image:
        image_input = args.image
        is_file = os.path.isfile(image_input)

        if is_file:
            import mimetypes
            mime_type, _ = mimetypes.guess_type(image_input)
            if not mime_type:
                mime_type = "image/jpeg"

            with open(image_input, "rb") as img_file:
                b64_data = base64.b64encode(img_file.read()).decode("utf-8")
                image_url_content = f"data:{mime_type};base64,{b64_data}"
        else:
            image_url_content = image_input

        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url_content}}
        ]
        print(f"Adding image input: {image_input}")

    messages = [
        ChatMessage(role="user", content=content)
    ]
    tools = None
    tool_choice = None
    if args.tools_file:
        try:
            import json
            with open(args.tools_file, 'r', encoding='utf-8') as f:
                tools_data = json.load(f)
            if isinstance(tools_data, dict):
                tools = [tools_data]
            elif isinstance(tools_data, list):
                tools = tools_data
        except Exception as e:
            print(f"Error reading tools file: {e}")
            tools = None
    if args.tool_choice:
        tc = args.tool_choice
        try:
            import json
            tool_choice = json.loads(tc)
        except Exception:
            tool_choice = tc

    request = ChatCompletionRequest(
        messages=messages,
        model=model,
        streaming=True,
        max_tokens=args.max_tokens,
        tools=tools,
        tool_choice=tool_choice
    )
    # Make the request with timing statistics
    response_text = ""
    token_count = 0
    start_time = time.time()
    first_token_time = None

    print("\n=== Response ===\n")
    for chunk in uni.stream_complete(request):
        content = chunk.message.content
        if content:  # Only count non-empty content
            if first_token_time is None:
                first_token_time = time.time()
            # Estimate token count (rough approximation: 4 chars = 1 token)
            token_count += len(content) / 4
            print(content, end="", flush=True)
            response_text += content
        if hasattr(chunk.message, 'tool_calls') and chunk.message.tool_calls:
            print(f"\nTool call: {chunk.message.tool_calls}", flush=True)

    end_time = time.time()

    # Calculate statistics
    time_to_first_token = first_token_time - start_time if first_token_time else 0
    # Calculate tokens per second from first token to end (excluding initial latency)
    token_generation_time = end_time - first_token_time if first_token_time else 0
    tokens_per_second = token_count / \
        token_generation_time if token_generation_time > 0 else 0

    # Print statistics
    print(
        f"\n\ntok/s: {tokens_per_second:.2f}          tft: {time_to_first_token:.2f} s")


if __name__ == "__main__":
    main()
