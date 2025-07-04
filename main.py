from send_auth import get_api_key
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat import GigaChat
from logging_class import CustomGigaChat
from pdf_Loader import PDFProcessor

auth = get_api_key()

# Инициализация
giga = CustomGigaChat(
    credentials=auth,
    verify_ssl_certs=False,
    log_file='log.txt'
)

pdf_path = "NLP1.pdf"
pdf_processor = PDFProcessor(pdf_path)
full_text, context_prompt = pdf_processor.load_and_prepare_context()
system_msg = SystemMessage(content=context_prompt)

conversation_history = [system_msg]

while True:
    user_input = input("Пользователь: ")
    if user_input.lower() == 'stop':
        break
    
    conversation_history.append(HumanMessage(content=user_input))
    
    answer = giga.invoke(conversation_history)
    conversation_history.append(answer)
    
    print('Ассистент:', answer.content)
    print(f"Токенов использовано: {giga.get_token_stats()['total_tokens']}")