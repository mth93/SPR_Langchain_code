from document_transformers import spr_document_transformer  # Replace with your actual package name
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader


# Instantiate a language model (replace with your actual LM setup)
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key="AIzaSyCQ1z-oU-W1yjJzZeRg0j9dtrwMgx1p1JI", language="en", convert_system_message_to_human=True)

# Create an SPR transformer for text
text_transformer = spr_document_transformer.SPRTextTransformer("text", llm)

# Example usage with text
text = "This is a sample text to be transformed into SPR."
spr = text_transformer.from_texts(text)
print("SPR representation of text:", spr)

# Example usage with documents

loader = PyPDFLoader("/home/hammad/Desktop/LLM_Supremacy/Learning/simulacra.pdf")
pages = loader.load_and_split()

sprs = text_transformer.from_documents(pages)
print("SPR representations of documents:")
for spr in sprs:
    print(spr)
