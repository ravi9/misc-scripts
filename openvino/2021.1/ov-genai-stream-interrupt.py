import openvino_genai as ov_genai
import threading

# Flag to control the generation process
stop_generation = False

def streamer(subword):
    global stop_generation
    if stop_generation:
        print("\n Stopping LLM generation as per user request..")
        return True
    print(subword, end='', flush=True)
    return False

def listen_for_exit():
    global stop_generation
    input("Press Enter to stop generation...")  
    stop_generation = True  

# Start the listener thread
listener_thread = threading.Thread(target=listen_for_exit)
listener_thread.daemon = True  # Daemonize thread to exit when main program exits
listener_thread.start()

model_path = "TinyLlama-1.1B-Chat-v1.0"
print("\n Loading model..")
pipe = ov_genai.LLMPipeline(model_path, "CPU")
config = pipe.get_generation_config()
config.max_new_tokens = 200
prompt = "The Sun is yellow because"

print("\n Starting LLM generation..")
pipe.generate(prompt, config, streamer)

# Wait for the listener thread to finish
listener_thread.join()

