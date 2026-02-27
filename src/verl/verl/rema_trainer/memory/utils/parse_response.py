import json
import re
from typing import Any, Dict


def extract_llm_json_from_response(response_text: str) -> Dict[str, Any]:
    """
    Extract JSON object from model response text.
    Handles markdown code blocks and extracts the first valid JSON object.
    
    Args:
        response_text: Raw text response from the model
        
    Returns:
        Parsed JSON dictionary, or empty dict with _parse_success=False on failure.
    """
    response_text = response_text.strip()
    json_text = None
    
    # Method 1: Look for JSON in markdown code blocks
    json_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    match = re.search(json_block_pattern, response_text, re.DOTALL)
    if match:
        json_text = match.group(1)
    
    # Method 2: Find the first complete JSON object
    if not json_text:
        # Find opening brace
        start_idx = response_text.find('{')
        if start_idx != -1:
            # Find matching closing brace
            brace_count = 0
            for i in range(start_idx, len(response_text)):
                if response_text[i] == '{':
                    brace_count += 1
                elif response_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_text = response_text[start_idx:i+1]
                        break
    
    # Parse JSON response
    if json_text:
        try:
            parsed = json.loads(json_text)
            parsed["_parse_success"] = True  # Mark as successfully parsed
            return parsed
        except json.JSONDecodeError as e:
            # print(f"Failed to parse extracted JSON: {e}")
            # print(f"Extracted JSON text: {json_text}")
            return {"_parse_success": False}
    else:
        # print(f"No JSON found in response")
        # print(f"Raw response: {response_text}")
        return {"_parse_success": False}


def extract_answer_from_text(text: str) -> str:
    try:
        # Pattern 0: <answer>Text</answer>
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Pattern 1: **Answer:** Text
        match = re.search(r"\*\*Answer:\*\*\s*(.*)", text)
        if match:
            return match.group(1)

        # Pattern 2: **Answer: Text**
        match = re.search(r"\*\*Answer:\s*(.*?)\*\*", text)
        if match:
            return match.group(1)

        # Pattern 3: Answer: Text
        match = re.search(r"Answer:\s*(.*)", text)
        if match:
            return match.group(1)

        # If nothing matched return empty
        return ""
    except ValueError:
        return text.strip()  # fallback: use whole output