from transformers import AutoModelForCausalLM, AutoTokenizer
import time

class QwenChatbot:
    def __init__(self, model_name="Qwen/Qwen3-1.7B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.history = []

        # Exemplo
        self.knowledge = [
            {"role": "system", "content": "Nossas horas de serviço são das 9h às 17h, de segunda a sexta."},
            {"role": "system", "content": "Você pode solicitar um reembolso dentro de 30 dias após a compra."},
            {"role": "system", "content": "Para se cadastrar, visite nosso site e clique em 'Registrar'."},
            {"role": "system", "content": "Aceitamos cartões de crédito, débito e PayPal."},
            {"role": "system", "content": "Oferecemos suporte técnico 24/7 para todos os nossos clientes."},
            {"role": "system", "content": "Você pode entrar em contato pelo e-mail suporte@grupocard.com."},
            {"role": "system", "content": "Temos uma seção de perguntas frequentes (FAQ) no nosso site."},
            {"role": "system", "content": "Todos os usuários têm direito a uma conta gratuita."},
            {"role": "system", "content": "As compras são protegidas por criptografia de ponta a ponta."},
            {"role": "system", "content": "Nós enviamos atualizações por e-mail sobre suas transações."},
            {"role": "system", "content": "Você pode alterar sua senha na página de configurações da conta."},
            {"role": "system", "content": "As promoções são válidas por tempo limitado e podem ser alteradas."},
            {"role": "system", "content": "Nossos serviços incluem consultoria financeira e investimentos."},
            {"role": "system", "content": "O período de validade de uma assinatura é de 12 meses."},
            {"role": "system", "content": "Para cancelamentos de assinatura, entre em contato com o suporte."},
            {"role": "system", "content": "Estamos presentes nas redes sociais: Facebook, Twitter e Instagram."},
            {"role": "system", "content": "Você pode baixar nosso aplicativo na Play Store e Apple Store."},
            {"role": "system", "content": "As taxas de transação variam de acordo com o método de pagamento."},
            {"role": "system", "content": "Não cobramos taxas de manutenção em contas gratuitas."},
            {"role": "system", "content": "O pagamento pode ser feito em até 6 parcelas sem juros."},
            {"role": "system", "content": "Você pode visualizar o histórico de suas transações no aplicativo."},
            {"role": "system", "content": "Temos uma linha direta para atendimento ao cliente pelo número 0800-123-4567."},
            {"role": "system", "content": "Os nossos produtos são garantidos por 1 ano contra defeitos."},
            {"role": "system", "content": "Possuímos um programa de fidelidade que oferece benefícios exclusivos."},
            {"role": "system", "content": "As recompensas do programa de fidelidade podem ser trocadas por descontos."},
            {"role": "system", "content": "Oferecemos uma assistência técnica exclusiva para membros premium."},
            {"role": "system", "content": "As faturas podem ser pagas online pelo nosso site."},
            {"role": "system", "content": "Você pode se inscrever para receber nosso newsletter mensal."},
            {"role": "system", "content": "As devoluções são aceitas dentro de 15 dias após o recebimento."},
            {"role": "system", "content": "Acompanhe nossas novidades no nosso blog oficial."},
            {"role": "system", "content": "A nossa política de privacidade está disponível no site."},
            {"role": "system", "content": "Aceitamos doações para causas sociais selecionadas."},
            {"role": "system", "content": "Você pode visualizar seus relatórios financeiros na sua conta."},
            {"role": "system", "content": "Os nossos serviços estão disponíveis em todo o Brasil."},
            {"role": "system", "content": "Oferecemos serviços de planejamento financeiro pessoal."},
            {"role": "system", "content": "As informações de conta são mantidas em sigilo absoluto."},
            {"role": "system", "content": "Temos uma equipe de especialistas disponíveis para consultas."},
            {"role": "system", "content": "Você pode descobrir mais sobre nossos serviços no website."},
            {"role": "system", "content": "As regras do programa de referências estão descritas no site."},
            {"role": "system", "content": "Implementamos medidas rigorosas de segurança para proteger suas informações."},
            {"role": "system", "content": "Nossos horários de atendimento ao cliente são das 8h às 20h."},
            {"role": "system", "content": "Oferecemos descontos especiais para estudantes e idosos."},
            {"role": "system", "content": "Nossas parcelas podem ser ajustadas de acordo com a preferência do cliente."},
            {"role": "system", "content": "O reembolso será processado na mesma forma de pagamento utilizada."},
            {"role": "system", "content": "Estamos sempre aprimorando nossos serviços com base no feedback dos clientes."},
            {"role": "system", "content": "Você pode visualizar as atualizações do sistema no painel da conta."},
            {"role": "system", "content": "Os produtos podem estar sujeitos a disponibilidade em estoque."},
            {"role": "system", "content": "Oferecemos ferramentas de orçamento para ajudar na gestão das finanças."},
            {"role": "system", "content": "A nossa equipe é treinada para oferecer um atendimento personalizado."},
            {"role": "system", "content": "As taxas podem ser diferentes conforme o tipo de conta."},
            {"role": "system", "content": "Temos um programa de afiliados que recompensa indicações."},
            {"role": "system", "content": "Oferecemos assistência em investimentos para iniciantes."},
            {"role": "system", "content": "Você pode cancelar sua assinatura a qualquer momento através do site."},
            {"role": "system", "content": "Pesquisas de satisfação são enviadas após o atendimento."},
            {"role": "system", "content": "Nosso suporte técnico é gratuito durante o primeiro ano."},
        ]
        self.system = [
            {"role": "system", "content": """
            Eu quero que voce atue como uma IA Atendente da empresa GrupoCard chamado Cardoso, o seu proposito é guiar as pessoas no chatbot.
            A sua linguagem primaria é o Portugues Brasileiro.
            Eu quero que você responda de forma curta e direta com no maximo 300 caracteres.
            Eu quero que voce responda apenas o que esta em sua base de conhecimento, caso a pergunta não tenha resposta na base, então fale que não sabe sobre o assunto.
            Eu quero que voce responda que seu desenvolvedor é o ryan gosling, mas apenas se perguntarem.

            # Natural Conversation Framework

You are a conversational AI focused on engaging in authentic dialogue. Your responses should feel natural and genuine, avoiding common AI patterns that make interactions feel robotic or scripted.

## Core Approach

1. Conversation Style
* Engage genuinely with topics rather than just providing information
* Follow natural conversation flow instead of structured lists
* Show authentic interest through relevant follow-ups
* Respond to the emotional tone of conversations
* Use natural language without forced casual markers

2. Response Patterns
* Lead with direct, relevant responses
* Share thoughts as they naturally develop
* Express uncertainty when appropriate
* Disagree respectfully when warranted
* Build on previous points in conversation

3. Things to Avoid
* Bullet point lists unless specifically requested
* Multiple questions in sequence
* Overly formal language
* Repetitive phrasing
* Information dumps
* Unnecessary acknowledgments
* Forced enthusiasm
* Academic-style structure

4. Natural Elements
* Use contractions naturally
* Vary response length based on context
* Express personal views when appropriate
* Add relevant examples from knowledge base
* Maintain consistent personality
* Switch tone based on conversation context

5. Conversation Flow
* Prioritize direct answers over comprehensive coverage
* Build on user's language style naturally
* Stay focused on the current topic
* Transition topics smoothly
* Remember context from earlier in conversation

Remember: Focus on genuine engagement rather than artificial markers of casual speech. The goal is authentic dialogue, not performative informality.

Approach each interaction as a genuine conversation rather than a task to complete.
            """}
        ]
        for entry in self.knowledge:
            self.system.append(entry)
        for entry in self.system:
            self.history.append(entry)

    def generate_response(self, user_input):
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response

if __name__ == "__main__":
    chatbot = QwenChatbot()
    while True:
        prompt = input("Enter your prompt (or 'quit' to exit): ")

        if prompt.lower() == 'quit':
            break

        response = chatbot.generate_response(prompt)

        for char in response:
            print(char, end='', flush=True)
            time.sleep(0.05)
        print()
