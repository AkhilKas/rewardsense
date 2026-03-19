from google import genai

client = genai.Client(
    vertexai=True,
    project="rewardsense",
    location="us-central1",
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Why might a 2% cash back card beat a travel rewards card for grocery spending?",
)

print(response.text)
