from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import numpy as np
import re

class HuggingFaceEmbeddings:
    def __init__(self, api_key, model="sentence-transformers/all-MiniLM-L6-v2"):
        self.client = InferenceClient(model=model, token=api_key)
    
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            response = self.client.feature_extraction(text)
            embeddings.append(np.mean(response, axis=0) if len(response.shape) > 1 else response)
        return embeddings
    
    def embed_query(self, text):
        response = self.client.feature_extraction(text)
        return np.mean(response, axis=0) if len(response.shape) > 1 else response

def extract_video_id(url):
    # Regular expression to find the video ID in a YouTube URL
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        return None

def get_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        return transcript
    except TranscriptsDisabled:
        return None

def create_vector_store(transcript, api_key):
    if not transcript:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    
    # Create the embeddings object
    embeddings = HuggingFaceEmbeddings(api_key=api_key)
    
    # Embed the documents manually
    texts = [doc.page_content for doc in chunks]
    embedded_docs = embeddings.embed_documents(texts)
    
    # Pass a callable lambda function for embedding queries
    def embed_function(text):
        return embeddings.embed_query(text)
    
    # Use FAISS.from_embeddings with the callable embed_function
    vector_store = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, embedded_docs)),
        embedding=embed_function,
        normalize_L2=True
    )
    return vector_store

def setup_qa_chain(vector_store, api_key):
    if not vector_store:
        return None
    
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    
    llm_client = InferenceClient(model="mistralai/Mixtral-8x7B-Instruct-v0.1", token=api_key)
    
    def llm_invoke(prompt):
        response = llm_client.text_generation(prompt, max_new_tokens=200, stop_sequences=["\n\n"])
        return response
    
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        Context: {context}
        Question: {question}
        Answer:
        """,
        input_variables=['context', 'question']
    )
    
    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })
    
    parser = StrOutputParser()
    
    main_chain = parallel_chain | prompt | RunnableLambda(lambda x: llm_invoke(x.text)) | parser
    return main_chain

def answer_question(video_url, question, api_key):
    # Extract video ID from the URL
    video_id = extract_video_id(video_url)
    if not video_id:
        return "Invalid YouTube URL. Please provide a valid URL with a video ID."
    
    transcript = get_transcript(video_id)
    if not transcript:
        return "No captions available for this video."
    
    vector_store = create_vector_store(transcript, api_key)
    if not vector_store:
        return "Failed to create vector store."
    
    qa_chain = setup_qa_chain(vector_store, api_key)
    if not qa_chain:
        return "Failed to set up QA chain."
    
    return qa_chain.invoke(question)