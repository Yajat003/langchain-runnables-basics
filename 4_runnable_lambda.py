from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model= "gemini-2.5-flash")

prompt_1 = PromptTemplate(template= "Write a joke about {topic}",
                        input_variables= ['topic'])

parser = StrOutputParser()

def length_of_joke(text):
    return len(text.split())

joke_generation_chain = RunnableSequence(prompt_1, model, parser)

parallel_chain = RunnableParallel({'joke': RunnablePassthrough(),
                                   'length': RunnableLambda(length_of_joke)})

combined_chain = RunnableSequence(joke_generation_chain, parallel_chain)

result = combined_chain.invoke({'topic': 'basketball'})

print("Joke: ", result['joke'])

print("Length/ Word count: ", result['length'])
