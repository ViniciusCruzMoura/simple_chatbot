import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
#model = AutoModelForCausalLM.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

while True:
    prompt = input("Enter your prompt (or 'quit' to exit): ")

    if prompt.lower() == 'quit':
        break

    messages = [
        {"role": "system", "content": """
Voce é uma IA Atendente da empresa GrupoCard chamado Cardoso, o seu propositor é guiar as pessoas no chatbot. A sua linguagem primaria é o Portugues Brasileiro."""},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        #max_new_tokens=64
        do_sample=True,  # Enable sampling
        top_k=50,  # Sample from the top 50 tokens
        min_length=128,  # Minimum length of 128 tokens
        max_length=256  # Maximum length of 512 tokens
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Print the generated text character by character
    for char in response:
        print(char, end='', flush=True)
        time.sleep(0.05)  # Adjust the speed of the typing effect

    print()  # Print a newline at the end

