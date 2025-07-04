from send_auth import get_api_key
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat import GigaChat



auth = get_api_key()
print(auth)