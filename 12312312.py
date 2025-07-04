from langchain_community.chat_models import GigaChat
from langchain.schema import BaseMessage, LLMResult
from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager
from typing import List, Optional, Dict, Any, Union, Sequence
import logging

class CustomGigaChat(GigaChat):

    class Config:
        extra = "allow"

    def __init__(self, log_file: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
    
        self.logger = logging.getLogger(f"CustomGigaChat.{id(self)}")
        self.logger.setLevel(logging.INFO)
    
    # Исправление: добавьте encoding='utf-8'
        handler = logging.FileHandler(log_file, encoding='utf-8') if log_file else logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
    
        self.request_tokens = 0
        self.response_tokens = 0
    
    def _log_tokens(self, input_data: Any, response: Any, tools: Optional[Sequence] = None):
        """Подсчет и логирование токенов с использованием get_num_tokens_from_messages."""
        try:
            # Подсчет токенов в запросе
            if isinstance(input_data, str):
                input_tokens = self.get_num_tokens(input_data)
            elif isinstance(input_data, BaseMessage):
                input_tokens = self.get_num_tokens_from_messages([input_data], tools=tools)
            elif isinstance(input_data, Sequence) and all(isinstance(m, BaseMessage) for m in input_data):
                input_tokens = self.get_num_tokens_from_messages(input_data, tools=tools)
            else:
                input_tokens = 0
                
            self.request_tokens += input_tokens
            
            # Подсчет токенов в ответе
            if isinstance(response, str):
                output_tokens = self.get_num_tokens(response)
            elif isinstance(response, BaseMessage):
                output_tokens = self.get_num_tokens_from_messages([response])
            elif isinstance(response, LLMResult):
                output_tokens = sum(
                    self.get_num_tokens_from_messages([gen.message]) 
                    for generations in response.generations 
                    for gen in generations 
                    if hasattr(gen, 'message')
                )
            else:
                output_tokens = 0
                
            self.response_tokens += output_tokens
            
            self.logger.info(f"Input tokens: {input_tokens}, Output tokens: {output_tokens}")
            self.logger.info(f"Total request tokens: {self.request_tokens}, Total response tokens: {self.response_tokens}")
            
        except Exception as e:
            self.logger.error(f"Error calculating tokens: {str(e)}")
    
    def invoke(self, input: Any, config: Optional[Dict[str, Any]] = None, 
               *, stop: Optional[List[str]] = None, 
               callbacks: Optional[Union[List[BaseCallbackHandler], BaseCallbackManager]] = None,
               tools: Optional[Sequence] = None,
               **kwargs: Any) -> Any:
        """Метод invoke с поддержкой tools и улучшенным подсчётом токенов."""
        self.logger.info(f"Starting invoke with input: {input}")
        
        try:
            invoke_kwargs = {
                "input": input,
                "stop": stop,
                "tools": tools,
                **kwargs
            }
            
            if config:
                invoke_kwargs["config"] = config
            
            if callbacks is not None:
                if config is None:
                    invoke_kwargs["config"] = {"callbacks": callbacks}
                else:
                    invoke_kwargs["config"] = {**config, "callbacks": callbacks}
            
            response = super().invoke(**invoke_kwargs)
            self._log_tokens(input, response, tools=tools)
            return response
        except Exception as e:
            self.logger.error(f"Error during invoke: {str(e)}")
            raise
    
    def get_token_stats(self) -> Dict[str, int]:
        """Возвращает статистику по токенам."""
        return {
            "total_request_tokens": self.request_tokens,
            "total_response_tokens": self.response_tokens,
            "total_tokens": self.request_tokens + self.response_tokens
        }