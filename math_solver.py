# For converting gestures to For Gemini API logic
import google.generativeai as genai

genai.configure(api_key="AIzaSyA1U180a2_nUTZWvo_FIq6fIChYuNgTZq4")

def solve_math_expression(expr):
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(f"Solve this math expression step-by-step: {expr}")
        return response.text.strip()
    except Exception as e:
        return f"Gemini AI failed to solve the expression. Error: {str(e)}"

