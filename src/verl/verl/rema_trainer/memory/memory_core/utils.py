import tiktoken
# TODO:: must match model with the one used in embeddings ? or with one we will answer with!
def count_tokens(text, model="gpt-4o-mini"):
    """Count tokens using tiktoken"""
    import traceback
    
    encoding = tiktoken.encoding_for_model(model)
    
    # Convert input to string if it's not already a string
    if not isinstance(text, str):
        print(f"!!!! WARNING: Non-string input to count_tokens: {repr(text)} (type: {type(text)})")
        print("!!!! STACK TRACE:")
        traceback.print_stack()
        print("!!!! END STACK TRACE")
        
        # Handle lists by joining them
        if isinstance(text, list):
            text = " ".join(str(item) for item in text)
            print(f"!!!! FIXED: Converted list to string for tokenization: {repr(text)}")
        else:
            text = str(text)
            print(f"!!!! FIXED: Converted to string for tokenization: {repr(text)}")
    
    try:
        length = len(encoding.encode(text))
    except Exception as e:
        print(f"!!!! ERROR when processing text: {text}")
        print(f"!!!! ERROR type: {type(e).__name__}: {e}")
        print("!!!! STACK TRACE:")
        traceback.print_stack()
        print("!!!! END STACK TRACE")
        return 0
    return len(encoding.encode(text))