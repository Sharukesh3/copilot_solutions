def reword(text):
    from Reword_prompt.remove_think import remove_think_tags
    from dotenv import load_dotenv
    import os

    load_dotenv()  # Load .env variables

    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model="deepseek-r1-distill-llama-70b",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )
    
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant whose sole task is to transform the provided {input_prompt} into a detailed reworded prompt. Your goal is to capture all the dependencies and nuances of the user's intended meaning, explaining in depth how the original prompt should be interpreted. Do not provide an answer—only rephrase the prompt.",
            ),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm
    result = chain.invoke(
        {
            "input_prompt": text,
            "input": text,
        }
    ).content
    
    #print("Before think tag removal:", result)
    result = remove_think_tags(result)
    #print("After think tag removal:",result)
    return result 

#Test
#reword("Develop a mobile app that tracks your daily water intake. It should send reminders, log your consumption, and provide insights into your hydration habits.")