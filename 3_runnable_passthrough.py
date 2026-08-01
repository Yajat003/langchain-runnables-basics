from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model= "gemini-2.5-flash")

prompt_1 = PromptTemplate(template= "Write a joke about {topic}",
                        input_variables= ['topic'])

prompt_2 = PromptTemplate(template= "Explain the following joke {text}",
                          input_variables= ['text'])

parser = StrOutputParser()

joke_generation_chain = RunnableSequence(prompt_1, model, parser)

parallel_chain = RunnableParallel({'joke': RunnablePassthrough(),
                                   'explanation': RunnableSequence(prompt_2, model, parser)})

combined_chain = RunnableSequence(joke_generation_chain, parallel_chain)

result = combined_chain.invoke({'topic': 'football'})

print("Joke: ", result['joke'])
print("Explanation of the joke: ", result['explanation'])