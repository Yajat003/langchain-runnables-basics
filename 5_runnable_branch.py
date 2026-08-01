from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model= "gemini-2.5-flash")

prompt_1 = PromptTemplate(template= "Write a detailed report on {topic}",
                         input_variables= ['topic'])

prompt_2 = PromptTemplate(template= "Summarise the following text \n {text}",
                          input_variables= ['text'])

parser = StrOutputParser()

report_generation_chain = RunnableSequence(prompt_1, model, parser)

branched_chain = RunnableBranch((lambda x: len(x.split()) > 300, RunnableSequence(prompt_2, model, parser)),
                                RunnablePassthrough())

combined_chain = RunnableSequence(report_generation_chain, branched_chain)

result = combined_chain.invoke({'topic': 'Russia vs Ukraine'})

print(result)