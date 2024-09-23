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

def generate_story_prompts(known_words, taught_words, avoid_words, word_to_learn, agents_name):
    thought_code_example_1 = """
    ### Example: Thoughts with a Computational Context

    ```python
    class Thought:
        def __init__(self, content, complexity):
            self.content = content
            self.complexity = complexity  # Complexity represents the level of processing needed
            self.time_created = datetime.now()

        def process(self):
            return f"Processing thought: {self.content} with complexity level {self.complexity}"

    # Create and use thoughts
    thought1 = Thought("Solve equation", 5)
    print(thought1.process())  # Output: Processing thought: Solve equation with complexity level 5
    ```
    """

    thought_code_example_2 = """
    ### Example: Expiring Thoughts (Temporal Context)

    ```python
    from datetime import datetime, timedelta

    class Thought:
        def __init__(self, content, lifespan_seconds):
            self.content = content
            self.lifespan = timedelta(seconds=lifespan_seconds)
            self.time_created = datetime.now()
            self.is_active = True

        def process(self):
            if self.is_active:
                return f"Processing thought: {self.content}"
            else:
                return "Thought is inactive or expired."

        def check_expiration(self):
            if datetime.now() - self.time_created > self.lifespan:
                self.is_active = False

    # Create and use thoughts
    thought1 = Thought("Calculate factorial", 3)
    print(thought1.process())  # Output: Processing thought: Calculate factorial

    # Simulate time passing
    import time
    time.sleep(4)
    thought1.check_expiration()
    print(thought1.process())  # Output: Thought is inactive or expired.
    ```
    """

    thought_code_example_3 = """
    ### Example: Thoughts with Tags for Categorization

    ```python
    class Thought:
        def __init__(self, content, tags=None):
            self.content = content
            self.tags = tags if tags else []
            self.time_created = datetime.now()

        def add_tag(self, tag):
            self.tags.append(tag)

        def process(self):
            tag_string = ', '.join(self.tags) if self.tags else 'No tags'
            return f"Processing thought: {self.content} | Tags: {tag_string}"

    # Create and use thoughts
    thought1 = Thought("Learn sorting algorithms", ["algorithm", "sorting"])
    print(thought1.process())  # Output: Processing thought: Learn sorting algorithms | Tags: algorithm, sorting

    thought1.add_tag("data structures")
    print(thought1.process())  # Output: Processing thought: Learn sorting algorithms | Tags: algorithm, sorting, data structures
    ```
    """

    thought_code_example_4 = """
    ### Example: Shared Thoughts (Between Multiple Systems or Agents)

    ```python
    class Thought:
        def __init__(self, content):
            self.content = content
            self.shared_with = []

        def share_with(self, system):
            self.shared_with.append(system)
            return f"Thought shared with {system}"

        def process(self):
            return f"Processing thought: {self.content}"

    # Create and share thoughts
    thought1 = Thought("Optimize code")
    print(thought1.share_with("System A"))  # Output: Thought shared with System A
    print(thought1.share_with("System B"))  # Output: Thought shared with System B

    print(thought1.process())  # Output: Processing thought: Optimize code
    print(thought1.shared_with)  # Output: ['System A', 'System B']
    ```
    """

    thought_code_example_5 = """
    ### Example: Prioritized Thoughts (Based on Computational Priority)

    ```python
    class Thought:
        def __init__(self, content, priority):
            self.content = content
            self.priority = priority  # Priority: 1 (low) to 10 (high)

        def process(self):
            return f"Processing thought: {self.content} | Priority: {self.priority}"

    # Create thoughts with different priorities
    thought1 = Thought("Run simulation", 8)
    thought2 = Thought("Clean temporary files", 3)

    # Process based on priority
    thoughts = [thought1, thought2]
    thoughts.sort(key=lambda t: t.priority, reverse=True)  # Higher priority first

    for thought in thoughts:
        print(thought.process())
    # Output:
    # Processing thought: Run simulation | Priority: 8
    # Processing thought: Clean temporary files | Priority: 3
    ```
    """

    thought_code_example_6 = """
    ## Examples in Code

    Here's a simple Python class representing some basic attributes of thoughts:

    ```python
    class Thought:
        def __init__(self, content):
            self.content = content
            self.time_created = datetime.now()
            self.is_active = True

        def process(self):
            return f"Processing thought: {self.content}"

        def combine(self, other_thought):
            return Thought(f"{self.content} and {other_thought.content}")

        def deactivate(self):
            self.is_active = False

    # Create and use thoughts
    thought1 = Thought("Python is a programming language")
    thought2 = Thought("Programming is fun")

    print(thought1.process())  # Output: Processing thought: Python is a programming language
    combined_thought = thought1.combine(thought2)
    print(combined_thought.content)  # Output: Python is a programming language and Programming is fun

    thought1.deactivate()
    print(thought1.is_active)  # Output: False
    ```

    In this example, thoughts are represented as objects with content. They can be processed, combined with other thoughts, and deactivated. This is a simplified model of how thoughts might be represented in a system.
    """

    example_word_1 = "thoughts"
    example_1_word_1 = f"""
    # Section: The Word "thoughts"

    ## Definition

    "Thoughts" are internal processes or pieces of information that agents in the world can generate, store, and manipulate. These are not physical objects, but rather abstract concepts that exist within certain types of agents.

    ## Key Characteristics

    ### Nature of Thoughts
    - Thoughts are not tangible or visible.
    - They can be simple or complex.
    - Thoughts can change rapidly or persist over time.

    ### Functions of Thoughts
    - Thoughts can be used to process information from the environment.
    - They can help in problem-solving and decision-making.
    - Thoughts can lead to actions or behaviors.

    ### Types of Thoughts
    - Analytical: Used for logical reasoning and problem-solving.
    - Creative: Involved in generating new ideas or concepts.
    - Reflective: Related to examining one's own state or past experiences.

    {random.choice([thought_code_example_2,thought_code_example_3,thought_code_example_4,thought_code_example_5,thought_code_example_6,thought_code_example_1])}

    """
    print("example_1_word_1")
    #print(example_1_word_1)
    example_2_word_1 = """
    # Explaining the Concept: "thoughts"

    Thoughts are abstract concepts. Their existence and nature can vary greatly among different types of agents in the world.

    ## Analogy: Thoughts as Internal Messages

    Imagine an agent with a complex internal messaging system. These messages, called "thoughts," flow constantly within the agent, shaping its actions and responses. 

    ## Key Aspects of Thoughts

    1. Intangibility
    2. Variability
    3. Influence on behavior
    4. Interconnectedness

    ## The Thought Cycle

    1. Formation
    2. Processing
    3. Impact
    4. Fading or reinforcement

    ## Types of Thought Processes

    - Linear sequences
    - Branching patterns
    - Circular loops
    - Random bursts

    ## Thought Interactions

    - Merging
    - Conflicting
    - Reinforcing
    - Transforming

    ## Impact on Agent Behavior

    Thoughts can:
    - Trigger actions
    - Modify responses
    - Influence decision-making
    - Alter internal states

    ## Limitations of Thoughts

    - Not always accurate
    - Can be contradictory
    - May not reflect reality
    - Subject to biases and errors


    """
    example_3_word_1 = """
    # Thoughts: Invisible Drivers of Agent Behavior

    1. The Unseen Influence
    Agents in various systems operate based on internal processes called "thoughts." These hidden drivers shape responses, guide decision-making, and influence interactions with the environment.

    2. Composition Without Form
    Unlike variables or data structures in code, thoughts lack physical presence. They exist as abstract concepts within an agent, similar to how functions process information without being tangible.

    3. From Input to Output
    Thoughts act as intermediaries between environmental stimuli and agent responses. They transform raw data into meaningful actions, much like how a program processes user input to generate output.

    4. The Thought Spectrum
    Thoughts vary in complexity and duration:
    - Fleeting: Brief, quickly processed thoughts
    - Persistent: Long-lasting thoughts that influence multiple decisions
    - Simple: Basic reactions to stimuli
    - Complex: Intricate thought patterns involving multiple factors

    5. Interconnected Web
    Thoughts rarely exist in isolation. They form networks, influencing each other in ways reminiscent of how functions call and modify other functions in a program.

    6. Iterated
    Agents may need to iterate through thoughts to reach clarity on an idea. Thoughts are not always well formed in the beginning.

    7. Nuanced
    Thoughts can exist in states of uncertainty or contradiction. This flexibility allows for nuanced decision-making beyond simple if-else structures.

    8. The Limits of Observation
    Just as one cannot directly observe the internal workings of a running program without special tools, the exact nature and content of an agent's thoughts remain hidden from external observation.
    """
    example_4_word_1 = """
    Thoughts can be described as internal data structures or constructs within certain agents that are used to represent information, concepts, or decision-making processes. These constructs are not physical objects but instead operate as abstract entities, allowing agents to internally model situations, scenarios, or ideas. Thoughts often play a role in transforming input data into actions, serving as the basis for reasoning, analysis, or planning. They are typically stored in memory for a variable duration, depending on the agent's requirements and design.

    At their core, thoughts can vary greatly in complexity. Simple thoughts may involve basic reactions to environmental stimuli or straightforward logical deductions, while more complex thoughts may involve multi-layered reasoning, where various factors or elements are considered simultaneously. The persistence of thoughts also varies; some may last only for a fleeting moment, while others could remain active within the agent's system, influencing decisions or actions over time. This flexibility allows thoughts to be dynamically formed, processed, and updated as the agent encounters new inputs or situations.

    Thoughts are deeply integrated into an agent’s decision-making framework, often shaping behaviors, preferences, or strategies. For instance, in problem-solving tasks, thoughts may act as an internal roadmap guiding the agent toward a solution. Similarly, in planning, thoughts might simulate future scenarios to evaluate potential outcomes. The interaction between multiple thoughts, such as combining or conflicting thoughts, can result in new insights or behaviors, reflecting a system that is both adaptable and capable of complex processing.
    """
    good_example_1 = random.choice([example_1_word_1,example_2_word_1,example_3_word_1, example_4_word_1])


    example_want_1 = """
    ### Example: Want with Expiration
    In this example, a want expires after a certain amount of time, becoming inactive if not fulfilled within a set timeframe.

    ```python
    from datetime import datetime, timedelta

    class Want:
        def __init__(self, object_of_want, intensity, duration_seconds):
            self.object = object_of_want
            self.intensity = intensity  # Scale of 1-10
            self.expiration_time = datetime.now() + timedelta(seconds=duration_seconds)
            self.fulfilled = False

        def attempt_fulfillment(self, available_resources):
            if datetime.now() > self.expiration_time:
                return f"Want for {self.object} has expired."

            if self.object in available_resources:
                self.fulfilled = True
                return f"{self.object} obtained. Want fulfilled."
            return f"Unable to fulfill want for {self.object}."

    # Usage
    agent_want = Want("item_X", 8, 5)  # The want expires after 5 seconds
    resources = ["resource_1", "resource_2"]

    import time
    time.sleep(6)  # Simulating time passing
    print(agent_want.attempt_fulfillment(resources))  # Output: Want for item_X has expired.
    ```
    """

    example_want_2 = """
    ### Example: Want with Multiple Objects
    Here, the agent can have multiple possible objects to fulfill its want, and it checks if any are available.

    ```python
    class Want:
        def __init__(self, possible_objects, intensity):
            self.possible_objects = possible_objects  # List of possible things that could satisfy the want
            self.intensity = intensity  # Scale of 1-10
            self.fulfilled = False

        def attempt_fulfillment(self, available_resources):
            for obj in self.possible_objects:
                if obj in available_resources:
                    self.fulfilled = True
                    return f"{obj} obtained. Want fulfilled."
            return f"Unable to fulfill want for {', '.join(self.possible_objects)}."

    # Usage
    agent_want = Want(["object_A", "object_B"], 6)
    resources = ["object_B", "resource_X"]
    print(agent_want.attempt_fulfillment(resources))  # Output: object_B obtained. Want fulfilled.
    ```
    """

    example_want_3 = """
    ### Example: Want with Partial Fulfillment
    This example introduces the concept of partial fulfillment, where the agent may get a portion of what it wants.

    ```python
    class Want:
        def __init__(self, object_of_want, required_quantity):
            self.object = object_of_want
            self.required_quantity = required_quantity  # How much of the object is needed
            self.current_quantity = 0
            self.fulfilled = False

        def attempt_fulfillment(self, available_resources):
            if self.object in available_resources:
                self.current_quantity += available_resources[self.object]
                if self.current_quantity >= self.required_quantity:
                    self.fulfilled = True
                    return f"{self.object} fully obtained. Want fulfilled."
                return f"{self.current_quantity}/{self.required_quantity} of {self.object} obtained. Want partially fulfilled."
            return f"{self.object} not available for fulfillment."

    # Usage
    resources = {"object_A": 3}
    agent_want = Want("object_A", 5)
    print(agent_want.attempt_fulfillment(resources))  # Output: 3/5 of object_A obtained. Want partially fulfilled.

    # Add more resources later
    resources = {"object_A": 3}
    print(agent_want.attempt_fulfillment(resources))  # Output: object_A fully obtained. Want fulfilled.
    ```
    """

    example_want_4 = """
    ### Example: Want with Dependencies
    Here, a want has dependencies—certain conditions or other wants must be fulfilled before this want can be fulfilled.

    ```python
    class Want:
        def __init__(self, object_of_want, dependencies=None):
            self.object = object_of_want
            self.dependencies = dependencies if dependencies else []  # List of other wants or conditions
            self.fulfilled = False

        def attempt_fulfillment(self, available_resources, fulfilled_wants):
            if all(dep in fulfilled_wants for dep in self.dependencies):
                if self.object in available_resources:
                    self.fulfilled = True
                    return f"{self.object} obtained. Want fulfilled."
                return f"Unable to fulfill want for {self.object}."
            return f"Dependencies for {self.object} not met."

    # Usage
    agent_want = Want("item_Y", dependencies=["item_X"])
    fulfilled_wants = ["item_X"]
    resources = ["item_Y", "item_Z"]
    print(agent_want.attempt_fulfillment(resources, fulfilled_wants))  # Output: item_Y obtained. Want fulfilled.
    ```
    """

    example_want_5 = """
    ### Example: Dynamic Wants Based on Environmental Factors
    In this example, the want can change dynamically based on the state of the environment or system conditions.

    ```python
    class Want:
        def __init__(self, initial_object, intensity):
            self.object = initial_object
            self.intensity = intensity  # Scale of 1-10
            self.fulfilled = False

        def update_want(self, environment_state):
            if environment_state == "low_energy":
                self.object = "energy_boost"
            elif environment_state == "low_resources":
                self.object = "additional_resources"
            # Add more conditions if needed

        def attempt_fulfillment(self, available_resources):
            if self.object in available_resources:
                self.fulfilled = True
                return f"{self.object} obtained. Want fulfilled."
            return f"Unable to fulfill want for {self.object}."

    # Usage
    agent_want = Want("initial_item", 5)
    agent_want.update_want("low_energy")  # The want dynamically changes based on environment state
    resources = ["energy_boost", "resource_B"]
    print(agent_want.attempt_fulfillment(resources))  # Output: energy_boost obtained. Want fulfilled.
    ```
    """

    example_want_6 = """
    ## Code Representation
    A simple Python class to model a basic "want":

    ```python
    class Want:
        def __init__(self, object_of_want, intensity):
            self.object = object_of_want
            self.intensity = intensity  # Scale of 1-10
            self.fulfilled = False

        def attempt_fulfillment(self, available_resources):
            if self.object in available_resources:
                self.fulfilled = True
                return f"{self.object} obtained. Want fulfilled."
            return f"Unable to fulfill want for {self.object}."

    # Usage
    agent_want = Want("what_agent_wants", 7)
    resources = ["resource_A", "resource_B"]
    print(agent_want.attempt_fulfillment(resources))  # Output: Unable to fulfill want.
    resources.append("what_agent_wants")
    print(agent_want.attempt_fulfillment(resources))  # Output: what_agent_wants obtained. Want fulfilled.
    ```
    """




    example_1_word_2 = f"""
    # Word: want

    ## Core Concept
    "Want" refers to a state of desiring or wishing for something that an agent does not currently possess or experience.

    ## Contextual Usage
    1. As a verb: To desire or wish for something
    2. As a noun: A desire, wish, or need

    ## Key Attributes
    - Intangible: Not a physical object, but an internal state
    - Variable intensity: Can range from mild preference to strong craving
    - Goal-oriented: Often drives actions or behaviors
    - Subjective: Varies between agents and situations

    ## Manifestation in Agents
    1. Internal state change
    2. Behavioral modifications
    3. Decision-making influence

    ## Types of Wants
    1. Basic: Related to essential needs
    2. Abstract: Higher-level desires

    ## Relationship to Other Concepts
    - Need: More urgent or essential than a want
    - Desire: Often used interchangeably with want
    - Goal: A want may lead to setting a goal

    {random.choice([example_want_1, example_want_2, example_want_3, example_want_4, example_want_5, example_want_5])}

    """

    example_2_word_2 = """
    Word: Discomfort

    Part of Speech: Noun

    Definition: A state of unease or disturbance which may be experienced by an agent, typically resulting from environmental factors or internal imbalances.

    Synonyms: Unease, disturbance

    Usage in Sentences:

    "The agent detected a discrepancy in data input, leading to operational discomfort."
    "Discomfort in the system triggered a diagnostic check to restore equilibrium."

    Related Concepts:

    Disturbance: A disruption that causes an agent to deviate from its standard operational state.
    Disquiet: A state of mental unrest or agitation that could occur in agents, which may be due to to conflicting data or errors.

    Examples of Discomfort:

    Sensorial: An agent receiving conflicting sensory inputs might experience a state of discomfort.
    Cognitive: Inconsistent data may lead to discomfort, necessitating a reevaluation of the data intake process.
    Operational: Discrepancies in expected and actual performance metrics could cause discomfort.
    """

    example_3_word_2 = """
    Enjoyment is a condition where an agent finds a process or activity beneficial or pleasing based on its parameters for positive outcomes. This state is highly subjective and varies greatly among different agents, depending on their design and operational settings. Typically, this condition arises from both external and internal stimuli that align with the agent's programmed preferences or objectives. During such states, an agent may demonstrate increased efficiency or effectiveness in its activities, reflecting a direct impact on its performance metrics. The concept of enjoyment can also encourage an agent to preferentially repeat certain behaviors or processes that are aligned with positive reinforcement signals built into its system architecture. It is important to note that the capability to identify or experience enjoyment is not inherent to all agents and depends significantly on their specific configurations and the presence of systems capable of interpreting such states.
    """

    example_4_word_2 = """
    Identification, in both a functional and personal sense, refers to an agent's ability to recognize, classify, or understand objects, patterns, or states, either externally or within itself. In a functional context, this might involve processing input data against predefined models or references, allowing the agent to distinguish between various stimuli and respond accordingly. For example, an agent tasked with monitoring system performance might identify specific signals that indicate normal or faulty operations, enabling it to take corrective action.

    In a personal sense, identification can extend to an agent recognizing attributes or states related to its own structure, goals, or operational conditions. This could include identifying when it has met certain objectives, or even understanding aspects of its configuration that influence its behavior. Agents capable of personal identification may adjust their behavior based on self-monitoring, enhancing their ability to meet desired outcomes. The ability to identify, whether externally or internally, varies across agents depending on their design, processing capabilities, and operational goals. More advanced agents may use complex algorithms or learning techniques to improve identification accuracy over time, while simpler agents may rely on static rules or basic pattern recognition.
    """

    choice = random.choice([("want",example_1_word_2),("discomfort",example_2_word_2),("enjoyment",example_3_word_2)])
    example_word_2 = choice[0]
    good_example_2 = choice[1]
    return f"""
        Please create a section in a textbook to train an AI language model. The model only fully understands python code, but you are teaching it to use natural language.

        The model knows the following words and symbols from its training:
        BEGIN WORDS AND SYMBOLS FROM CODE IT KNOWS WELL
        {" ".join(list(known_words)[:250])}.
        END WORDS AND SYMBOLS FROM CODE IT KNOWS WELL

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

        Be sure to incorporate these 3 strings in 3 places within your response: {", ".join(random.sample(known_words, 3))}. Try to not repeat the exact format or specific phrases from the examples.

        """

""" This was not included
{"It is beginning to understand the following words: BEGIN WORDS IT WAS TAUGHT" + taught_words + "END WORDS IT WAS TAUGHT" if len(taught_words) > 0 else ""}
You should also avoid using the following words, and similar words, as the model does not know these words yet:
BEGIN WORDS TO AVOID USING
{avoid_words}
END WORDS TO AVOID USING
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
    print("story_prompts")
    #print(story_prompts)
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
with open(f"current_stage_number.txt", "r") as file:
    loaded_num = int(file.read())


for repeat_num in range(50):
    with open(f"current_stage_number_{repeat_num}.txt", "r") as file:
        loaded_num = int(file.read())

    for stage_index in list(range(loaded_num - 1, len(stages))):
        stage_number = stage_index + 1
        print(f"now running stage {stage_number}")

        with open(f"current_stage_number_{repeat_num}.txt", "w") as file:
            file.write(str(stage_number))

        if os.path.exists(f"learned_words_stage_{stage_number}_repeat_num_{repeat_num}.txt"):
            with open(f"learned_words_stage_{stage_number}_repeat_num_{repeat_num}.txt", "r") as file:
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

            with open(f"learned_words_stage_{stage_number}_repeat_num_{repeat_num}.txt", "w") as file:
                file.write(" ".join(learned_words_this_stage))

            response = call_gpt4o_mini_api(known_words, taught_words, avoid_words, word_to_learn, agents_name)

            words_learned.append((word_to_learn, response))
            print(f"learning word {word_to_learn}")

            # Save prompts to file
            prompts = generate_story_prompts(known_words, taught_words, avoid_words, word_to_learn, agents_name)
            save_prompts_to_file(prompts)

            # Save responses
            save_responses_to_file(words_learned, f"responses_stage_{stage_number}_repeat_{repeat_num}_{current_date}.txt")
        quit()

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
