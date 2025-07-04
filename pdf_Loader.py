# pdf_processor.py
from langchain_community.document_loaders import PyPDFLoader
from typing import Tuple

class PDFProcessor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
    
    def load_and_prepare_context(self) -> Tuple[str, str]:
        """
        Загружает PDF и создает контекстный промпт
        Возвращает:
            - полный текст PDF
            - системный промпт с ограничениями
        """
        try:
            # Загрузка PDF
            loader = PyPDFLoader(self.pdf_path)
            pages = loader.load()
            full_text = "\n".join([page.page_content for page in pages])
            
            # Создание контекстного промпта
            context_prompt = f"""
            Ты должен отвечать ТОЛЬКО на основе следующего документа:
            {full_text}
            
            Правила:
            1. Если вопрос не относится к документу: "Это вне рамок предоставленных материалов"
            2. Не придумывай информацию, которой нет в документе
            3. Для сложных вопросов объединяй информацию из разных частей документа
            """
            
            return full_text, context_prompt
            
        except Exception as e:
            raise RuntimeError(f"Ошибка обработки PDF: {e}")