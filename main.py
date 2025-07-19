import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set up FastAPI
app = FastAPI(
    title="Person Extraction API",
    description="Extract structured person information from text using a language model.",
    version="1.0.0"
)

# Define the Pydantic models for request and response
class ExtractRequest(BaseModel):
    text: str = Field(..., description="The raw text to extract person information from.")

class Person(BaseModel):
    name: Optional[str] = Field(default=None, description="The complete name of the person")
    phone_number: Optional[str] = Field(default=None, description="The phone number of the person")
    age: Optional[str] = Field(default=None, description="The age of the person in years")
    address: Optional[str] = Field(default=None, description="The complete address of the person")


structured_llm = init_chat_model(
    "gpt-4o-mini",
    model_provider="openai",
    api_key=OPENAI_API_KEY
).with_structured_output(schema=Person)

# Define the prompt template for structured extraction
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value. Output only valid JSON matching the Person schema."
        ),
        ("human", "{text}"),
    ]
)

@app.post("/extract", response_model=Person)
async def extract_person(request: ExtractRequest):
    """
    Extract person information from the provided text.
    """
    # Build the prompt with the user text
    prompt = prompt_template.invoke({"text": request.text})
    try:
        # Invoke the structured LLM to get a Person instance
        result = structured_llm.invoke(prompt)
        print(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

    # Return the Person object (FastAPI will handle JSON serialization and validation)
    return result
