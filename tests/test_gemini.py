import os

from google import genai

# from google.generativeai.types import content_types, generation_types


# @pytest.mark.parametrize("model_name", ["gemini-2.5-pro-exp-03-25"])
def test_gemini_25_pro_real_request():
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    prompt = "Ola"

    # config = generation_types.GenerateContentConfig(
    #     response_mime_type="application/json",
    #     temperature=0.3,
    # )

    response = client.models.generate_content(
        model="gemini-2.5-pro-exp-03-25",
        contents=prompt,
    )

    print(response)

    # assert response.parsed is not None
    # assert isinstance(response.parsed, dict)
    # assert response.parsed.get("status") == "ok"


if __name__ == "__main__":
    test_gemini_25_pro_real_request()
