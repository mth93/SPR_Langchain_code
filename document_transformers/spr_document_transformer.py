from prompts.SPR_prompts import spr_compression_chat_template
from typing import List, Any
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import BaseMessage

class SPRTextTransformer():
   """
   Transforms text or code into sparse priming representations (SPRs).

   Args:
       type (str): The type of input to be transformed, either "text" or "code".
       llm (BaseChatModel): A language model to use for generating SPRs.
   """

   def __init__(self, type: str, llm: BaseChatModel):
       super().__init__()
       self.type = type
       self.llm = llm

   def from_documents(self, docs: List[Document]) -> List[str]:
       """
       Generates SPRs from a list of documents.

       Args:
           docs (List[Document]): A list of documents to transform.

       Returns:
           List[str]: A list of SPRs, one for each document.

       Raises:
           ValueError: If an invalid document type is encountered.
       """

       responses = []
    #    print(type(docs))
       for doc in docs:
        #    print(type(doc))
           try:
               messages = self.get_prompt_messages(doc)
            #    print(messages)
               response = self.llm.invoke(messages)
            #    print(response)
               responses.append(Document(page_content=response.content))
           except Exception as e:
               print(f"Error processing document: {e}")
               # Add more specific error handling here if needed

       return responses

   def from_texts(self, texts: str) -> str:
       """
       Generates SPRs from a list of text strings.
       """

       chunks = self.split_text_into_chunks(texts)
       docs = self.from_documents(chunks)  # Corrected the method call
       sprs = '\n\n'.join([doc.page_content for doc in docs])
       return sprs

   def split_text_into_chunks(self, text: str, chunk_size=10000, chunk_overlap=1000) -> List[str]:
       """
       Splits a text string into smaller chunks.
       """

       text_splitter = RecursiveCharacterTextSplitter(
           chunk_size=chunk_size, chunk_overlap=chunk_overlap
       )
       chunks = text_splitter.split_text(text)
       return chunks

   def get_prompt_messages(self, text: str) -> List[BaseMessage]:
       """
       Generates prompt messages for the language model.
       """

       if self.type == "text":
           prompt_template = spr_compression_chat_template
       elif self.type == "code":
           prompt_template = spr_compression_chat_template
       else:
           raise ValueError("Invalid type. Must be 'text' or 'code'.")

       messages = prompt_template.format_messages(
           name="SPR Compressor",
           user_input=text,
           code=text
       )
       return messages
