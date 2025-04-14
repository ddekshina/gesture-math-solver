import re
import google.generativeai as genai

def validate_expression(expr):
    """
    Validate if a math expression is well-formed
    
    Args:
        expr: String representing a math expression
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Remove whitespace
    expr = expr.strip()
    
    # Check for empty expression
    if not expr:
        return False
    
    # Check for invalid characters
    if not all(c in '0123456789+-*/() ' for c in expr):
        return False
    
    # Check for balance of parentheses
    if expr.count('(') != expr.count(')'):
        return False
    
    # Check for consecutive operators
    if re.search(r'[+\-*/]{2,}', expr):
        return False
    
    # Check if expression starts with an operator (except minus)
    if re.match(r'^[+*/]', expr):
        return False
    
    # Check if expression ends with an operator
    if re.search(r'[+\-*/]$', expr):
        return False
    
    return True

def solve_expression(expr):
    """
    Solve a math expression safely
    
    Args:
        expr: String representing a math expression
        
    Returns:
        String result or error message
    """
    try:
        # First validate the expression
        if not validate_expression(expr):
            return "❌ Invalid expression"
        
        # Calculate the result
        result = eval(expr)
        
        # Format the result
        if isinstance(result, int):
            return str(result)
        else:
            # Format float to remove trailing zeros
            return str(round(result, 4)).rstrip('0').rstrip('.') if '.' in str(result) else str(result)
            
    except Exception as e:
        return f"❌ Error: {str(e)}"

def solve_with_gemini(expr, api_key=None):
    """
    Solve math expression using Google's Gemini API with step-by-step explanation
    
    Args:
        expr: String representing a math expression
        api_key: Optional API key for Gemini
        
    Returns:
        String containing the solution with steps
    """
    try:
        # Validate expression first
        if not validate_expression(expr):
            return "❌ Invalid expression"
            
        # Skip API call for simple expressions
        if len(expr) <= 5 and all(c in '0123456789+-*/' for c in expr):
            result = solve_expression(expr)
            return f"Result: {result}"
            
        # Configure Gemini API
        if api_key:
            genai.configure(api_key=api_key)
        
        # Generate a step-by-step solution
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(
            f"Solve this math expression step-by-step: {expr}\n"
            "Provide your reasoning clearly, and make sure the final answer is correct. "
            "If the expression is invalid, explain why."
        )
        
        return response.text.strip()
        
    except Exception as e:
        # Fallback to basic solver
        result = solve_expression(expr)
        return f"Result: {result}\n\n(Gemini API not available: {str(e)})"