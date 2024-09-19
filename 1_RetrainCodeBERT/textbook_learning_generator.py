import json
import requests
import random
import string
import os

# Check if updated files exist, and use them
known_words_file = "known_words_updated.txt" if os.path.exists("known_words_updated.txt") else "common_words.txt"
avoid_words_file = "avoid_words_updated.txt" if os.path.exists("avoid_words_updated.txt") else "words_to_avoid.txt"

with open(avoid_words_file, "r") as file:
    avoid_words = set(file.read().split())
with open(known_words_file, "r") as file:
    known_words = set(file.read().split())

agents_name = ''.join(random.choices(string.ascii_uppercase, k=5))

print("agents_name")
print(agents_name)
print()

def generate_story_prompts(known_words, avoid_words, word_to_learn, agents_name):
    # Here we can add logic to create more elaborate story prompts
    # Example: integrate words in a meaningful and contextual way
    return f"""
        Please create a section in a textbook to train an AI language model. The model only understands python code, but you are teaching it to use natural language.
        The model knows the following words: {", ".join(known_words)}.

        You should also avoid using the following words, and similar words. {", ".join(avoid_words.difference(set([word_to_learn])))}

        In essence, stick to using only words that a model only trained on python code would understand.
        Please avoid references to emotion or in any way indicating to the model its situation (such as whether it is conscious, has feelings, or friends or family). Sometimes words you are teaching have to do with emotion, thoughts, or feelings. In this case, you must explain that certain agents in the world can experience things, although certainly not all of them. Do not indicate to the AI language model its status in terms of experience, or whether it can experience feelings. The model has no knowledge of culture, society, humanity, or feelings, so these things must be explained to it using words it can understand.

        Today we are teaching the model the word {word_to_learn}.

        Do not include a conclusion section or summarize in any way. Do not refer to people, or users unnecessarily. Do not refer to "understanding" or the concept of learning. Stick to just teaching this word. Do not explain why it is important or provide any moral lesson.

    """

    """ MAYBE ADD THIS IN LATER?
    The model's name is {agents_name}, which is always referred to in third person (e.g. "{agents_name} replied saying...").
    """

def generate_tag():
    # Generates a five-letter all caps tag for the GPT to use as its identifier
    return ''.join(random.choices(string.ascii_uppercase, k=5))

def call_gpt4o_mini_api(known_words,avoid_words,word_to_learn,agents_name):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.getenv('OPENAI_API_KEY')
    }
    story_prompts = generate_story_prompts(known_words, avoid_words, word_to_learn,agents_name)
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": f"You are an expert textbook writer teaching language models which only know how to code to use natural language."},
            {"role": "user", "content": story_prompts}
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()

def save_prompts_to_file(prompts, filename="prompts.txt"):
    with open(filename, "a") as file:
        file.write(prompts)
        file.write("<|SEPARATOR_OF_PAGES|>\n")

# Save responses to a single file
def save_responses_to_file(words_learned, filename):
    with open(filename, "w") as file:
        for word, response in words_learned:
            file.write(json.dumps(response))
            file.write("<|SEPARATOR_OF_PAGES|>\n")

# Learn 50 words by retraining the model
words_learned = []
for _ in range(50):
    word_to_learn = random.choice(list(avoid_words))
    avoid_words.remove(word_to_learn)
    known_words.add(word_to_learn)

    response = call_gpt4o_mini_api(known_words,avoid_words,word_to_learn,agents_name)

    words_learned.append((word_to_learn, response))
    print(f"learning word {word_to_learn}")

    # Save prompts to file
    prompts = generate_story_prompts(known_words, avoid_words, word_to_learn, agents_name)
    save_prompts_to_file(prompts)

    # Update files every 10 iterations (or at the end)
    if (len(words_learned) % 10 == 0) or len(words_learned) == 50:
        # Save responses
        save_responses_to_file(words_learned, "responses.txt")
        # Update known and avoid words lists
        with open("known_words_updated.txt", "w") as file:
            file.write(" ".join(known_words))
        with open("avoid_words_updated.txt", "w") as file:
            file.write(" ".join(avoid_words))

""""
response = call_gpt4o_mini_api(known_words,avoid_words,word_to_learn,agents_name)

# Use response as needed
# print(response)
# Function to pretty print the response
def pretty_print_response(response):
    print(f"ID: {response['id']}")
    print(f"Object: {response['object']}")
    print(f"Created: {response['created']}")
    print(f"Model: {response['model']}")
    print(f"Choices:")
    for i, choice in enumerate(response['choices'], 1):
        print(f"  Choice {i}:")
        print(f"    Index: {choice['index']}")
        print(f"    Message:")
        print(f"      Role: {choice['message']['role']}")
        print(f"      Content: {choice['message']['content']}")
        print(f"    Finish Reason: {choice['finish_reason']}")
    print(f"Usage:")
    print(f"  Prompt Tokens: {response['usage']['prompt_tokens']}")
    print(f"  Completion Tokens: {response['usage']['completion_tokens']}")
    print(f"  Total Tokens: {response['usage']['total_tokens']}")
    print(f"System Fingerprint: {response['system_fingerprint']}")

print()
print()
print()
print()
print(f"WORD TO LEARN: {word_to_learn}")
print("")
# Pretty print the response
pretty_print_response(response)
# Save the response to a file
with open(f"{word_to_learn}_response.json", "w") as file:
    json.dump(response, file)
"""
