from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model= "gemini-2.5-flash")

prompt_1 = PromptTemplate(template= "Write a joke about {topic}",
                        input_variables= ['topic'])

prompt_2 = PromptTemplate(template= "Explain the following joke {text}",
                          input_variables= ['text'])

parser = StrOutputParser()

chain = RunnableSequence(prompt_1, model, parser, prompt_2, model, parser)

print(chain.invoke({'topic': 'basketball'}))

