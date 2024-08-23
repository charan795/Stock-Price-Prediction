#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 20:52:35 2024

@author: charanmakkina

Function for calling API
"""
# Import necessary modules
import nest_asyncio # Allows nested use of asyncio event loops, which is useful where an event loop is already running
import aiohttp # Handles asynchronous HTTP requests and responses
import asyncio # Provides support for asynchronous programming using coroutines, tasks and event loops

# Apply a patch to allow nested use of asyncio event loops
nest_asyncio.apply()

#Define the data to be sent in POST requests
requests_data=[{"ticker":"AAPL"},{"ticker":"GOOG"}]

# Define the URL for the API endpoint
url="http://127.0.0.1:8000/forecast"

# Define an asynchronous function to send a POST request and return the response JSON
async def send_request(session,data):
    async with session.post(url,json=data) as response:
        return await response.json()

# Define the main asynchronous function to handle multiple requests
async def main():
    
    # Create a client session object for making HTTP requests
    async with aiohttp.ClientSession() as session:
        
        # Create a list of tasks to send POST requests for each item in requests_data
        tasks=[send_request(session, data) for data in requests_data]
        
        # Gather all tasks and wait for them to complete
        results=await asyncio.gather(*tasks)
        
        # Print each result obtained from the responses 
        for result in results:
            print(result)

if __name__=="__main__":
    
    # Run the asynchronous function
    asyncio.run(main())
