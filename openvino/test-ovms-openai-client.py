import argparse  
import requests  
from openai import OpenAI  
  
def get_first_available_model(base_url):  
    """Extract the first available model from OVMS config endpoint"""  
    config_url = base_url.replace('/v3', '/v1/config')  
    try:  
        response = requests.get(config_url)  
        response.raise_for_status()  
        config_data = response.json()  
          
        # Find first model with AVAILABLE state  
        for model_name, model_info in config_data.items():  
            if 'model_version_status' in model_info:  
                for version_status in model_info['model_version_status']:  
                    if version_status.get('state') == 'AVAILABLE':  
                        return model_name  
          
        # If no available model found, return first model name  
        return list(config_data.keys())[0] if config_data else None  
          
    except Exception as e:  
        print(f"Warning: Could not fetch model config: {e}")  
        return None  
  
def build_argparser():  
    parser = argparse.ArgumentParser(description='Test OVMS with OpenAI API')  
    parser.add_argument('-m','--model_name', default=None,  
                       help='Model name to use (auto-detected if not specified)')  
    parser.add_argument('-mt', '--max_tokens', type=int, default=None,
                       help='Max Output Tokens (default=500 for thinking, 50 for disable_thinking)')  
    parser.add_argument('-p', '--prompt', default='The capital of France is ', )
    parser.add_argument('--base_url', default='http://localhost:8000/v3',  
                       help='Base URL for the API')  
    parser.add_argument('-s', '--streaming', action='store_true',  
                       help='Enable streaming mode for responses')  
    parser.add_argument("-dt", "--disable_thinking", action='store_false',
                        dest="enable_thinking", default=True,
                        help="Disable 'thinking' mode in the chat template (default: enabled)")  
    return parser  
  
def main():
    args = build_argparser().parse_args()

    # Auto-detect model name if not provided
    model_name = args.model_name or get_first_available_model(args.base_url)
    if not model_name:
        print("Error: Could not auto-detect model name and none provided")
        return
    print(f"Auto-detected model: {model_name}")

    # Decide max_tokens
    max_tokens = (
        args.max_tokens if args.max_tokens is not None
        else 500 if args.enable_thinking else 50
    )
    print(f"Using max_tokens: {max_tokens}")
    print(f"Streaming: {args.streaming}")
    print(f"Thinking: {args.enable_thinking}")
    print(f"Prompt: {args.prompt}")
     
    client = OpenAI(base_url=args.base_url, api_key="unused")

    messages = [{"role": "user", "content": args.prompt}]

    common_params = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": args.streaming,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": args.enable_thinking}
        },
    }

    if args.streaming:
        response_printed = False
        print("\nStarted thinking...\n" if args.enable_thinking else "")
        stream = client.chat.completions.create(**common_params) 
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)
                response_printed = True
        if not response_printed:
            print(f"\nNo response — model may have used tokens for 'thinking'. "
                  f"Try increasing max_tokens (-mt, current={max_tokens}) or disable thinking (-dt).\n")
    else:
        response = client.chat.completions.create(**common_params)
        print(response.choices[0].message.content)

if __name__ == "__main__":
    main()
