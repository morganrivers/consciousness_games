# Required Imports
import json

def pretty_print_response(response):
    print(f"Created: {response['created']}")
    for i, choice in enumerate(response['choices'], 1):
        print(f"      Content: {choice['message']['content']}")
    print(f"Usage:")
    print(f"  Completion Tokens: {response['usage']['completion_tokens']}")
    print(f"  Total Tokens: {response['usage']['total_tokens']}")

def read_responses_from_file(filename):
    with open(filename, 'r') as f:
        content = f.read()
    responses = content.split("<|SEPARATOR_OF_PAGES|>")

    # Remove any empty strings from the list
    responses = [resp for resp in responses if resp.strip()]
    return responses

def main():
    filename = "responses.txt"
    responses = read_responses_from_file(filename)

    for response in responses:
        response_json = json.loads(response)  # Parse JSON string to dict
        pretty_print_response(response_json)
        print("\n"+"-"*50+"\n")  # Separator line between responses

if __name__ == "__main__":
    main()
