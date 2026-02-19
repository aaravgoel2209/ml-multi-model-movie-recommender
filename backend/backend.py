from fastapi import FastAPI, Request

app = FastAPI()

users = {}

@app.post("/reccommend/{user_id}")
async def _reccomend_user(user_id: str):
    
    body = await request.json()

    # format of body should be
    """
    {
        "input_movies" : [movie_id:int, ...]
    }
    """
    # pass to "lego bricks"

    results = []

    return {"reccomended_movies" : results} # Return movie ID


