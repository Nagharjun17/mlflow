import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex

def setup():
    print('*** Starting ***')

    model_name = "microsoft/phi-2" #"meta-llama/Llama-2-7b-hf"

    SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
    - Generate human readable output, avoid creating output with gibberish text.
    - Generate only the requested output, don't include any other language before or after the requested output.
    - Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
    - Generate professional language typically used in business documents in North America.
    - Never generate offensive or foul language.
    """

    query_wrapper_prompt = PromptTemplate(
        "[INST]<<SYS>>
" + SYSTEM_PROMPT + "<</SYS>>

{query_str}[/INST] "
    )

    llm = HuggingFaceLLM(
        context_window=2048,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.0, "do_sample": False},
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name=model_name,
        model_name=model_name,
        device_map="cpu",
        # model_kwargs={
        #     "torch_dtype": torch.float16, 
            # "load_in_4bit": True
            # },
    )
    print('*** LLM has been set up ***')

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    print('*** Embedding Model has been set up ***')

    Settings.llm = llm
    Settings.embed_model = embed_model

    documents = SimpleDirectoryReader(
        input_files=["/Volumes/shr2/Med_Pulmonary/Sleep_Research/A_DATA/Sleep_Report_NLP/NLP_EDA/Data_Common_txt_rev/1283ed44-f2e2-4f41-9c4f-4400dbc92fcd-REPORT.txt"]
        ).load_data()
    print('*** Data has been read ***')

    index = VectorStoreIndex.from_documents(documents)

    query_engine = index.as_query_engine()

    return query_engine

def main():
    query_engine = setup()

    response = query_engine.query("What is the BMI of the Patient?")
    print(response)


if __name__ == "__main__":
    main()