import streamlit as st
import asyncio
import aiohttp
from urllib.parse import quote


# Define the backend URL
BACKEND_URL = "http://localhost:8000/api/get-response"

# Streamlit app header
st.title("Chatbot")

# Function to send request to backend asynchronously
async def get_ollama(query):
    async with aiohttp.ClientSession() as session:
        encoded_query = quote(query)

        async with session.post(BACKEND_URL + f"?query={encoded_query}") as response:
            if response.status == 404:
                return "Please try again."
            elif response.status == 200:
                data = await response.json()
                return data
            else:
                return "An error occurred."

# Streamlit app body
user_input = st.text_input("You: ", "")
if st.button("Send"):
    if user_input:
        bot_response = asyncio.run(get_ollama(user_input))
        st.text_area("Chatbot:", bot_response)
    else:
        st.warning("Please enter a car query.")
