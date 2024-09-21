import json
import requests
import random
import string
import os
from datetime import datetime

# Get current date in yy_mm_dd format
current_date = datetime.now().strftime('%y_%m_%d')

# Check if updated files exist, and use them
known_words_file = "known_words_updated.txt" if os.path.exists("known_words_updated.txt") else "common_words.txt"
staged_words_file = "stages_words_deduplicated.txt"

# avoid_words_file = "avoid_words_updated.txt" if os.path.exists("avoid_words_updated.txt") else "words_to_avoid.txt"


with open(known_words_file, "r") as file:
    known_words = set(file.read().split())

with open(staged_words_file, "r") as file:
    lines = file.readlines()

# Convert each line into an array
stages = [line.split() for line in lines]



agents_name = ''.join(random.choices(string.ascii_uppercase, k=5))

# print("agents_name")
# print(agents_name)
# print()

def generate_story_prompts(known_words, taught_words, avoid_words, word_to_learn, agents_name):
    # Here we can add logic to create more elaborate story prompts
    # Example: integrate words in a meaningful and contextual way
    example_word_1 = "human"
    good_example_1 = """
    # Section: The Word "Human"

    ## Definition

    A "human" is a type of agent in the world. Humans are complex biological systems with many functions. They have a specific shape and structure, including a head, body, arms, and legs. Humans can move, process information, and interact with their environment.

    ## Key Characteristics

    ### Physical Attributes
    - Humans have a distinct shape, typically with two arms, two legs, one head, and one body.
    - The average human has a height between 1.5 and 2 meters.
    - Humans have various colors of skin, hair, and eyes.

    ### Functional Capabilities
    - Humans can move from one point to another using their legs.
    - They can manipulate objects in their environment using their arms and hands.
    - Humans have the ability to process complex information through their brain.

    ### Communication
    - Humans use various methods to exchange information, including spoken and written language.
    - They can produce and interpret sounds to convey meaning.

    ### Life Cycle
    - Humans start as small, then grow larger over time.
    - They go through different states: birth, growth, and eventually, a state called "death".

    ## Examples in Code

    Here's a simple Python class representing some basic attributes of a human:

    ```python
    class Human:
        def __init__(self, height, weight, age):
            self.height = height  # in meters
            self.weight = weight  # in kilograms
            self.age = age        # in years
            self.state = "alive"

        def grow(self, years):
            self.age += years
            if self.age > 100:
                self.state = "not_alive"

        def move(self, distance):
            return f"Human moved {distance} meters"

        def communicate(self, message):
            return f"Human says: {message}"

    # Create a new Human instance
    h1 = Human(1.75, 70, 30)

    # Use Human methods
    print(h1.move(10))  # Output: Human moved 10 meters
    print(h1.communicate("Hello"))  # Output: Human says: Hello

    # Grow the human by 80 years
    h1.grow(80)
    print(h1.state)  # Output: not_alive
    ```
    """

    example_word_2 = "behavioral"

    good_example_2 = """
    # Section: The Word "behavioral"

    ## Definition
    "behavioral" relates to the actions or responses of an object, function, or system to different inputs or conditions. It describes how something acts or performs in various situations.

    ## Usage in Examples

    ### 1. Behavioral Testing in Python

    Behavioral testing checks if a function or object behaves as expected for different inputs:

    ```python
    def add_numbers(x, y):
        return x + y

    # Behavioral test
    assert add_numbers(2, 3) == 5
    assert add_numbers(-1, 1) == 0
    assert add_numbers(0, 0) == 0
    ```

    ### 2. Behavioral Patterns in Code

    Behavioral patterns define how objects interact and distribute responsibilities:

    ```python
    class Subject:
        def __init__(self):
            self._observers = []

        def attach(self, observer):
            self._observers.append(observer)

        def detach(self, observer):
            self._observers.remove(observer)

        def notify(self):
            for observer in self._observers:
                observer.update()

    class Observer:
        def update(self):
            pass

    class ConcreteObserver(Observer):
        def update(self):
            print("Received update")

    subject = Subject()
    observer = ConcreteObserver()
    subject.attach(observer)
    subject.notify()
    ```

    ### 3. Behavioral Parameters in Functions

    Functions may have behavioral parameters that change how they operate:

    ```python
    def sort_list(numbers, reverse=False):
        return sorted(numbers, reverse=reverse)

    numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    print(sort_list(numbers))  # Default behavioral
    print(sort_list(numbers, reverse=True))  # Changed behavioral
    ```
    """

    return f"""
        Please create a section in a textbook to train an AI language model. The model only fully understands python code, but you are teaching it to use natural language.

        The model knows the following words and symbols from its training:
        BEGIN WORDS AND SYMBOLS FROM CODE IT KNOWS WELL
        {" ".join(known_words)}.
        END WORDS AND SYMBOLS FROM CODE IT KNOWS WELL

        {"It is beginning to understand the following words: BEGIN WORDS IT WAS TAUGHT" + taught_words + "END WORDS IT WAS TAUGHT" if len(taught_words) > 0 else ""}

        You should also avoid using the following words, and similar words, as the model does not know these words yet:
        BEGIN WORDS TO AVOID USING
        {avoid_words}
        END WORDS TO AVOID USING

        Stick to using only words that a model only trained on python code would understand where possible.
        Please avoid unnecessary references to emotion. Do not in any way indicate to the model its situation (such as whether it is conscious, has feelings, or friends or family). Sometimes words you are teaching have to do with emotion, thoughts, or feelings. In this case, you must explain that certain agents in the world can experience things, although certainly not all of them. Do not indicate to the AI language model its status in terms of experience, or whether it can experience feelings. The model has no knowledge of culture, society, humanity, or feelings, so these things must be explained to it using words it can understand.

        Please note, the intention of the exercise is to teach the model all senses of the words, and do not omit senses related to human experience. If we are teaching the model the word "odd", we would be sure to also teach it the word in the sense of "unusual", not just the numeric sense, as the intention is to teach the model to roughly understand these words the way a human would understand them.

        Here are two examples of a good way of teaching the words "{example_word_1}" and "{example_word_2}" to the model.

        START EXAMPLE RESPONSE
        {good_example_1}
        END EXAMPLE

        START EXAMPLE RESPONSE
        {good_example_2}
        END EXAMPLE

        Today we are teaching the model the word "{word_to_learn}".

        When teaching the word "{word_to_learn}", do not include a conclusion section or summarize in any way. Do not refer to people, or users unnecessarily. Do not refer to "understanding" or the concept of learning. Stick to just teaching this word. Do not explain why it is important or provide any moral lesson.
        """

    """ MAYBE ADD THIS IN LATER?
    The model's name is {agents_name}, which is always referred to in third person (e.g. "{agents_name} replied saying...").
    """

def generate_tag():
    # Generates a five-letter all caps tag for the GPT to use as its identifier
    return ''.join(random.choices(string.ascii_uppercase, k=5))

def call_gpt4o_mini_api(known_words,taught_words,avoid_words,word_to_learn,agents_name):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.getenv('OPENAI_API_KEY')
    }
    story_prompts = generate_story_prompts(known_words, taught_words, avoid_words, word_to_learn,agents_name)
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": f"You are an expert textbook writer teaching a language model which only knows well how to code to use natural language, and is learning human concepts from you."},
            {"role": "user", "content": story_prompts}
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()

def save_prompts_to_file(prompts, filename=f"prompts_{current_date}.txt"):
    with open(filename, "a") as file:
        file.write(prompts)
        file.write("<|SEPARATOR_OF_PAGES|>\n")

# Save responses to a single file
def save_responses_to_file(words_learned, filename):
    with open(filename, "w") as file:
        for word, response in words_learned:
            file.write(json.dumps(response))
            file.write("<|SEPARATOR_OF_PAGES|>\n")

def get_words_in_prior_stages(stages,stage_index):
    # Get all stages up to but not including stage_index
    selected_stages = stages[:stage_index]

    # Join each stage's words with a space, making each stage a string
    selected_stages = [' '.join(stage) for stage in selected_stages]

    # Join all stages with a newline character to get the full text
    text = '\n'.join(selected_stages)

    return text

def get_words_in_this_and_future_stages(stages, stage_index, word_to_learn):
    # Get all stages this one and after (including stage_index)
    selected_stages = stages[stage_index:]

    # Join each stage's words with a space, excluding word_to_learn, making each stage a string
    selected_stages = [' '.join(word for word in stage if word != word_to_learn) for stage in selected_stages]

    # Join all stages with a newline character to get the full text
    text = '\n'.join(selected_stages)

    return text


# Learn words in stages by retraining the model
# learned_stages = []
# Loading the integer back from the .txt file
with open("current_stage_number.txt", "r") as file:
    loaded_num = int(file.read())



for stage_index in list(range(loaded_num - 1, len(stages))):
    stage_number = stage_index + 1
    print(f"now running stage {stage_number}")

    with open("current_stage_number.txt", "w") as file:
        file.write(str(stage_number))

    if os.path.exists(f"learned_words_stage_{stage_number}.txt"):
        with open(f"learned_words_stage_{stage_number}.txt", "r") as file:
            learned_words_this_stage = list(set(file.read().split()))
    else:
        learned_words_this_stage = []
    words_learned = []


    words_in_stage = stages[stage_index]
    taught_words = get_words_in_prior_stages(stages,stage_index)
    # Randomly shuffle the+ words
    random.shuffle(words_in_stage)
    for word_to_learn in words_in_stage:
        if word_to_learn in learned_words_this_stage:
            print(f"skipping {word_to_learn} as we've seen it before")
            continue
        learned_words_this_stage.append(word_to_learn)

        # also exclude word we're learning
        avoid_words = get_words_in_this_and_future_stages(stages,stage_index,word_to_learn)

        with open(f"learned_words_stage_{stage_number}.txt", "w") as file:
            file.write(" ".join(learned_words_this_stage))

        response = call_gpt4o_mini_api(known_words, taught_words, avoid_words, word_to_learn, agents_name)

        words_learned.append((word_to_learn, response))
        print(f"learning word {word_to_learn}")

        # Save prompts to file
        prompts = generate_story_prompts(known_words, taught_words, avoid_words, word_to_learn, agents_name)
        save_prompts_to_file(prompts)

        # Update files every 10 iterations (or at the end)
        #if (len(words_learned) % 10 == 0) or len(words_learned) == 50:
        # Save responses
        save_responses_to_file(words_learned, f"responses_stage_{stage_number}_{current_date}.txt")

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
