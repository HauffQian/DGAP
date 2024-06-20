from fastapi import FastAPI, Request
from pydantic import BaseModel
import ollama
from ollama import AsyncClient

client = AsyncClient(host='http://localhost:11434')


app = FastAPI()



@app.post("/api/get-response")
async def get_reponse(request: Request):
    body = await request.json()
    query = body.get("query")
    print("stage 1")
    response = ollama.generate(model='llama3:70b-instruct',prompt = query,options={
    'num_predict': 2048,
    'temperature': 0,
    'top_p': 1,
        },)
    print(response['response'])
    
    return response['response']




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

    # test_car_types = ["我有一台三菱帕杰罗二代 2.5L排量4气缸20气门，是涡轮增压直喷四驱车，我想更换一下车辆前架"]
    # for car_type in test_car_types:
    #     try:
    #         response = get_car_model(car_type)
    #         print(response)
    #         # response = app.post("/api/get-car-model", json={"car_query": car_type})
    #         # # print(f"Request for car type '{car_type}': {response.status_code}")
    #         # if response.status_code == 200:
    #         #     print("Response:", response.json())
    #         # else:
    #         #     print("Error:", response.json())
    #     except Exception as e:
    #         print(f"An error occurred: {e}")