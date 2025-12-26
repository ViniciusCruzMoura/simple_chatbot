import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
#model_name = "Qwen/Qwen3-4B-Instruct-2507"
model_name = "Qwen/Qwen3-1.7B"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

knowledge_base_entries = [
    {"role": "system", "content": "Nossas horas de serviço são das 9h às 17h, de segunda a sexta."},
    {"role": "system", "content": "Você pode solicitar um reembolso dentro de 30 dias após a compra."},
    {"role": "system", "content": "Para se cadastrar, visite nosso site e clique em 'Registrar'."},
    {"role": "system", "content": "Aceitamos cartões de crédito, débito e PayPal."},
]

messages = [
    {"role": "system", "content": """
    Eu quero que voce atue como uma IA Atendente da empresa GrupoCard chamado Cardoso, o seu proposito é guiar as pessoas no chatbot.
    A sua linguagem primaria é o Portugues Brasileiro.
    Eu quero que você responda de forma curta e direta com no maximo 300 caracteres.
    Eu quero que voce responda apenas o que esta em sua base de conhecimento, caso a pergunta não tenha resposta na base, então fale que não sabe sobre o assunto.
    """}
]

for entry in knowledge_base_entries:
    messages.append(entry)

while True:
    prompt = input("Enter your prompt (or 'quit' to exit): ")

    if prompt.lower() == 'quit':
        break

    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        do_sample=True,
        top_k=50,
        min_length=128,
        max_length=1024,#256,
        #max_new_tokens=75
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    for char in response:
        print(char, end='', flush=True)
        time.sleep(0.05)

    print()

    messages.append({"role": "assistant", "content": response})

