# For converting gestures to For Gemini API logic
import google.generativeai as genai

genai.configure(api_key="AIzaSyA1U180a2_nUTZWvo_FIq6fIChYuNgTZq4")

def solve_math_expression(expr):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(f"Solve this: {expr}")
    return response.text.strip()
