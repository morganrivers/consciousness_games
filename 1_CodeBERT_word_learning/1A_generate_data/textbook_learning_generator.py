import time
import logging
import json
import requests
import random
import string
import os
from datetime import datetime
logging.basicConfig(filename='error_log.txt', level=logging.ERROR,
                    format='%(asctime)s %(levelname)s %(message)s')

# Get current date in yy_mm_dd format
current_date = datetime.now().strftime('%y_%m_%d')

# Check if updated files exist, and use them
known_words_file = "known_words_updated.txt" if os.path.exists("known_words_updated.txt") else "common_words.txt"
staged_words_file = "words_for_agent_with_synonyms_antonyms.txt" #"stages_words_deduplicated.txt"

# avoid_words_file = "avoid_words_updated.txt" if os.path.exists("avoid_words_updated.txt") else "words_to_avoid.txt"


with open(known_words_file, "r") as file:
    known_words = set(file.read().split())

with open(staged_words_file, "r") as file:
    lines = file.readlines()

# Convert each line into an array
stages = [line.split() for line in lines]


def sample_non_empty(known_words, num_samples=2):
    samples = []
    while len(samples) < num_samples:
        word = random.choice(known_words)
        if word != "":  # Check that the sampled word is not an empty string
            samples.append(word)
    return samples

agents_name = ''.join(random.choices(string.ascii_uppercase, k=5))

def generate_story_prompts(known_words, taught_words, avoid_words, word_to_learn, agents_name):
    thought_code_example_1 = """### Example: Thoughts with a Computational Context

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
```"""

    thought_code_example_2 = """### Example: Expiring Thoughts (Temporal Context)

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
            return "Thought is no longer relevant."

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
print(thought1.process())  # Output: Thought is no longer relevant.
```"""
    thought_code_example_3 = """### **Thoughts with tags for categorization**

```python
from collections import namedtuple

Thought = namedtuple('Thought', ['content', 'tags', 'connections'])

def create_thought(content, tags=None, connections=0):
    return Thought(content, tags or [], connections)  # Connections: Number of related thoughts

def add_tag(thought, tag):
    return thought._replace(tags=thought.tags + [tag])

def process_thought(thought):
    tag_string = ', '.join(thought.tags) if thought.tags else 'No tags'
    return f"Processing thought: {thought.content} | Tags: {tag_string} | Connections: {thought.connections}"

# Create and use thoughts
thought1 = create_thought("Learn sorting algorithms", ["algorithm", "sorting"], connections=2)
print(process_thought(thought1))  # Output: Processing thought: Learn sorting algorithms | Tags: algorithm, sorting | Connections: 2
```"""

    thought_code_example_4 = """### Python example: shared thoughts with different adaptabilities**

from dataclasses import dataclass, field

@dataclass
class Thought:
    content: str
    adaptability: int  # 1 (rigid) to 10 (highly adaptable)
    shared_with: list = field(default_factory=list)

    def share_with(self, system):
        self.shared_with.append(system)
        return f"Thought shared with {system}"

    def process(self):
        return f"Processing thought: {self.content} | Adaptability: {self.adaptability}"

# Create and share thoughts
thought1 = Thought("Refactor legacy code", adaptability=3)  # Low adaptability due to rigid constraints
thought2 = Thought("Implement any new software feature", adaptability=9)  # High adaptability due to flexibility

print(thought1.share_with("Team A"))  # Output: Thought shared with Team A
print(thought1.process())  # Output: Processing thought: Refactor legacy code | Adaptability: 3

print(thought2.share_with("Team B"))  # Output: Thought shared with Team B
print(thought2.process())  # Output: Processing thought: Implement any new software feature | Adaptability: 9
```"""

    thought_code_example_5 = """## Example: Thought Persistence using a Generator

    Here's an example of a thought generator that processes thoughts based on their persistence. Each thought can be processed a certain number of times, after which it expires:

    ```python
    def thought_generator(content, persistence):
        while persistence > 0:
            yield f"Processing thought: {content} | Persistence: {persistence}"
            persistence -= 1
        yield f"Thought '{content}' has expired"

    # Create thoughts with different persistence levels
    thought1 = thought_generator("Complete project documentation", 5)
    thought2 = thought_generator("Refactor codebase", 3)

    # Process the thoughts until they expire
    thoughts = [thought1, thought2]

    for thought in thoughts:
        try:
            while True:  # Keep processing until generator is exhausted
                print(next(thought))
        except StopIteration:
            continue
    ```

    In this example, each thought is represented by a generator that tracks its persistence. The `thought_generator` function allows you to repeatedly process a thought until its persistence reaches zero, at which point the thought expires. For instance, "Complete project documentation" can be processed five times, and "Refactor codebase" three times, after which both thoughts expire. This concept models how long a thought remains relevant or active in a system before being discarded."""

    thought_code_example_6 = """## Examples in Code

Here's a simple Python class representing some basic attributes of thoughts:
```python
from datetime import datetime

def create_thought(content):
    return {"content": content, "time_created": datetime.now(), "is_active": True}

def think(thought):
    return f"Thinking thought: {thought['content']}"

def combine_thoughts(thought1, thought2):
    return create_thought(f"{thought1['content']} and {thought2['content']}")

def forget_thought(thought):
    thought["is_active"] = False

# Create and use thoughts
thought1 = create_thought("Python is a programming language")
thought2 = create_thought("Programming is fun")

print(think(thought1))  # Output: Thinking thought: Python is a programming language
combined_thought = combine_thoughts(thought1, thought2)
print(combined_thought["content"])  # Output: Python is a programming language and Programming is fun

forget_thought(thought1)
print(thought1["is_active"])  # Output: False
```

In this example, mental processes for thoughts are represented as functions. Thoughts can themselves be thought about, combined with other thoughts, and forgotten. This is a simplified model of how thoughts might be represented in a system."""

    thought_code_example_7 = """# Code example: Thoughts with Differing Scope

```python
def create_thought(content, scope):
    return {"content": content, "scope": scope}

def evaluate_thought(thought):
    scope_descriptions = {range(1, 4): "narrow", range(4, 8): "moderate", range(8, 11): "broad"}
    scope_description = next(desc for r, desc in scope_descriptions.items() if thought['scope'] in r)
    return f"Evaluating thought: {thought['content']} | Scope: {scope_description}"

# Create and use thoughts
thought1 = create_thought("Calculate factorial", 2)
thought2 = create_thought("Improve algorithm", 6)
thought3 = create_thought("Create an entire software codebase", 9)

print(evaluate_thought(thought1))  # Output: Evaluating thought: Calculate factorial | Scope: narrow
print(evaluate_thought(thought2))  # Output: Evaluating thought: Improve algorithm | Scope: moderate
print(evaluate_thought(thought3))  # Output: Evaluating thought: Create an entire software codebase | Scope: broad
```"""
    example_word_1 = "thoughts"
    example_1_word_1 = f"""# Section: The Word "thoughts"

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
- Thoughts can be forgotten (freed from memory).

### Types of Thoughts
- Analytical: Used for logical reasoning and problem-solving.
- Creative: Involved in generating new ideas or concepts.
- Reflective: Related to examining one's own state or past experiences.

{random.choice([thought_code_example_1,thought_code_example_2,thought_code_example_3,thought_code_example_4,thought_code_example_5,thought_code_example_6,thought_code_example_7])}"""
    example_2_word_1 = """# Explaining the Concept: "thoughts"

The existence and nature of thoughts can vary greatly among different types of agents in the world.

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
- Subject to biases and errors"""

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
Just as one cannot directly observe the internal workings of a running program without special tools, the exact nature and content of an agent's thoughts remain hidden from external observation."""
    example_4_word_1 = """Thoughts can be described as internal data structures or constructs within certain agents that are used to represent information, concepts, or decision-making processes. These constructs are not physical objects but instead operate as abstract entities, allowing agents to internally model situations, scenarios, or ideas. Thoughts often play a role in transforming input data into actions, serving as the basis for reasoning, analysis, or planning. They are typically stored in memory for a variable duration, depending on the agent's requirements and design.

At their core, thoughts can vary greatly in complexity. Simple thoughts may involve basic reactions to environmental stimuli or straightforward logical deductions, while more complex thoughts may involve multi-layered reasoning, where various factors or elements are considered simultaneously. The persistence of thoughts also varies; some may last only for a fleeting moment, while others could remain active within the agent's system, influencing decisions or actions over time. This flexibility allows thoughts to be dynamically formed, processed, and updated as the agent encounters new inputs or situations.

Thoughts are deeply integrated into an agent’s decision-making framework, often shaping behaviors, preferences, or strategies. For instance, in problem-solving tasks, thoughts may act as an internal roadmap guiding the agent toward a solution. Similarly, in planning, thoughts might simulate future scenarios to evaluate potential outcomes. The interaction between multiple thoughts, such as combining or conflicting thoughts, can result in new insights or behaviors, reflecting a system that is both adaptable and capable of complex processing.
    """
    example_5_word_1 = """### Understanding "Thoughts" Through Q&A

**Q: What are thoughts?**
A: Thoughts are internal processes that occur in an agent’s mind, shaping its perception, actions, and decision-making. They function like messages sent within the mind, constantly influencing behavior and responses.

---

**Q: How would the nature of thoughts be described?**
A: Thoughts are intangible: they can't be physically touched or seen. They are highly variable, ranging from simple, fleeting ideas to complex, persistent contemplations. They significantly influence behavior and are interconnected, often triggering or modifying one another.

---

**Q: How do thoughts form and evolve?**
A: The thought cycle generally follows four stages:
1. **Formation** – a thought emerges based on stimuli or previous thoughts.
2. **Processing** – it is examined, analyzed, or reflected upon.
3. **Impact** – the thought affects the agent’s mood, decisions, or actions.
4. **Fading or Reinforcement** – it either fades from attention or is reinforced and becomes stronger, recurring.

---

**Q: Are there different types of thought patterns?**
A: Yes, thought processes can take various forms:
- **Linear sequences** – one thought leads directly to the next in a straightforward progression.
- **Branching patterns** – a thought expands into multiple directions or ideas.
- **Circular loops** – repetitive thinking where thoughts cycle back on themselves.
- **Random bursts** – thoughts that arise unpredictably, without a clear connection to prior thinking.

---

**Q: How do thoughts interact with each other?**
A: Thoughts can:
- **Merge** – blend together to create new ideas.
- **Conflict** – oppose each other, creating internal tension.
- **Reinforce** – strengthen one another, solidifying beliefs or plans.
- **Transform** – one thought may evolve into something entirely different as it's processed.

---

**Q: What is the relationship between thoughts and behavior?**
A: Thoughts have a direct impact on behavior. They can:
- **Trigger actions** – such as deciding to act on a desire.
- **Modify responses** – changing how an agent reacts to situations.
- **Influence decision-making** – guiding the agent to choose one option over another.
- **Alter internal states** – affecting emotions, motivation, and perception.

---

**Q: What are the limitations of thoughts?**
A: Despite their influence, thoughts are not always reliable. They can:
- **Be inaccurate** – leading to false assumptions or conclusions.
- **Contradict** each other – causing confusion or indecision.
- **Misrepresent reality** – leading to distorted perceptions.
- **Be biased or flawed** – influenced by emotional states, past experiences, or cognitive biases."""

    good_example_1 = random.choice([example_1_word_1,example_2_word_1,example_3_word_1, example_4_word_1])

    example_want_1 = """### Example: Want with Expiration and Immutable Data Structures
In this example, a want is stored as a named tuple, and after expiration, a new state is returned.

```python
from collections import namedtuple
from datetime import datetime, timedelta

# Named tuple for an immutable representation of a Want
Want = namedtuple("Want", ["object", "expiration_time", "priority", "transformed"])

def create_want(object_of_want, duration_seconds, priority="medium"):
    expiration_time = datetime.now() + timedelta(seconds=duration_seconds)
    return Want(object_of_want, expiration_time, priority, transformed=False)

def transform_want(want, new_object):
    # Return a new instance of the Want with the updated object and transformed flag
    return want._replace(object=new_object, transformed=True)

# Usage
agent_want = create_want("item_X", 5, priority="high")
new_object = "new_item_Y"

# Simulating time passing
import time
time.sleep(6)

# Transform want after expiration
if datetime.now() > agent_want.expiration_time:
    agent_want = transform_want(agent_want, new_object)

print(f"Want transformed to: {agent_want.object}, Priority: {agent_want.priority}")  # Output: Want transformed to: new_item_Y, Priority: high
```"""


    example_want_2 = """### Example: Want with Multiple Objects using Event-Driven Logic
Here, wants are managed as dictionaries, and a function processes events to rank available wants.

```python
def rank_wants(possible_wants, available_resources):
    # Rank wants based on availability in resources
    ranked_wants = sorted(possible_wants, key=lambda obj: available_resources.index(obj) if obj in available_resources else float('inf'))
    return ranked_wants[0] if ranked_wants else None

# Usage
possible_wants = ["object_A", "object_B", "object_C"]
resources = ["object_C", "object_B", "resource_X"]
print(f"Highest priority want: {rank_wants(possible_wants, resources)}")  # Output: Highest priority want: object_C
```"""

    example_want_3 = """### Example: Want with Partial Fulfillment using Immutable Tuples
This example decomposes a complex want into simpler parts using tuples.

```python
from collections import namedtuple

# A simple tuple for representing a decomposed want part
WantPart = namedtuple("WantPart", ["name", "urgency", "importance", "difficulty"])

def decompose_want(object_of_want, required_quantity):
    # Decomposing the want into parts with different attributes
    return [WantPart(f"portion_{i+1}", urgency="high" if i == 0 else "low", importance="critical" if i == required_quantity-1 else "medium", difficulty="low")
            for i in range(required_quantity)]

# Usage
decomposed_wants = decompose_want("object_A", 3)
print(decomposed_wants)
# Output: [('portion_1', 'high', 'medium', 'low'), ('portion_2', 'low', 'medium', 'low'), ('portion_3', 'low', 'critical', 'low')]
```"""

    example_want_4 = """### Example: Want with Dependencies using a State Machine
This example uses a state machine with states and transitions stored in dictionaries.

```python
def resolve_want_conflict(want1, want2, fulfilled_wants):
    # Resolve conflicts based on the number of fulfilled dependencies
    want1_deps_fulfilled = sum(1 for dep in want1['dependencies'] if dep in fulfilled_wants)
    want2_deps_fulfilled = sum(1 for dep in want2['dependencies'] if dep in fulfilled_wants)

    return want1 if want1_deps_fulfilled >= want2_deps_fulfilled else want2

# Usage
want1 = {'object': 'item_Y', 'dependencies': ['item_X']}
want2 = {'object': 'item_Z', 'dependencies': ['item_X', 'item_W']}
fulfilled_wants = ['item_X']

resolved_want = resolve_want_conflict(want1, want2, fulfilled_wants)
print(f"Resolved want: {resolved_want['object']}")  # Output: Resolved want: item_Y
```"""

    example_want_5 = """### Example: Dynamic Wants Based on Environmental Factors using Event-Driven Programming
This example updates a want dynamically based on environmental factors using simple functions.

```python
def transform_want(want, environment_state):
    # Update the want dynamically based on environment state
    if environment_state == "low_energy":
        return {"object": "energy_boost", "urgency": "critical"}
    elif environment_state == "low_resources":
        return {"object": "additional_resources", "urgency": "high"}
    return want

# Usage
agent_want = {"object": "initial_item", "urgency": "medium"}
agent_want = transform_want(agent_want, "low_energy")
print(f"Transformed want: {agent_want['object']}, Urgency: {agent_want['urgency']}")  # Output: Transformed want: energy_boost, Urgency: critical
```"""

    example_want_6 = """## Code Representation
This example combines multiple wants using immutable tuples and relative importance.

```python
from collections import namedtuple

# Named tuple to represent each want
Want = namedtuple("Want", ["object", "urgency"])

def combine_wants(want1, want2):
    # Combine wants and return the one with higher urgency
    if want1.urgency == "critical" or want2.urgency == "critical":
        combined_urgency = "critical"
    else:
        combined_urgency = "high" if "high" in [want1.urgency, want2.urgency] else "medium"
    return Want(f"{want1.object} and {want2.object}", combined_urgency)

# Usage
want1 = Want("want_for_item_A", "high")
want2 = Want("want_for_item_B", "critical")
combined_want = combine_wants(want1, want2)
print(f"Combined want: {combined_want.object}, Urgency: {combined_want.urgency}")
# Output: Combined want: want_for_item_A and want_for_item_B, Urgency: critical
```"""

    example_1_word_2 = f"""# Word: want

## Core Concept
"Want" refers to a state of desiring or wishing for something that an agent does not currently possess or experience.

## Contextual Usage
1. As a verb: To desire or wish for something
2. As a noun: A desire, wish, or need

## Key Attributes
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

{random.choice([example_want_1, example_want_2, example_want_3, example_want_4, example_want_5, example_want_5])}"""

    example_2_word_2 = """Word: Discomfort

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
Operational: Discrepancies in expected and actual performance metrics could cause discomfort."""

    example_3_word_2 = """Enjoyment is a condition where an agent finds a process or activity beneficial or pleasing based on its parameters for positive outcomes. This state is highly subjective and varies greatly among different agents, depending on their design and operational settings. Typically, this condition arises from both external and internal stimuli that align with the agent's programmed preferences or objectives. During such states, an agent may demonstrate increased efficiency or effectiveness in its activities, reflecting a direct impact on its performance metrics. The concept of enjoyment can also encourage an agent to preferentially repeat certain behaviors or processes that are aligned with positive reinforcement signals built into its system architecture. It is important to note that the capability to identify or experience enjoyment is not inherent to all agents and depends significantly on their specific configurations and the presence of systems capable of interpreting such states."""

    example_4_word_2 = """Identification, in both a functional and personal sense, refers to an agent's ability to recognize, classify, or understand objects, patterns, or states, either externally or within itself. In a functional context, this might involve processing input data against predefined models or references, allowing the agent to distinguish between various stimuli and respond accordingly. For example, an agent tasked with monitoring system performance might identify specific signals that indicate normal or faulty operations, enabling it to take corrective action.

In a personal sense, identification can extend to an agent recognizing attributes or states related to its own structure, goals, or operational conditions. This could include identifying when it has met certain objectives, or even understanding aspects of its configuration that influence its behavior. Agents capable of personal identification may adjust their behavior based on self-monitoring, enhancing their ability to meet desired outcomes. The ability to identify, whether externally or internally, varies across agents depending on their design, processing capabilities, and operational goals. More advanced agents may use complex algorithms or learning techniques to improve identification accuracy over time, while simpler agents may rely on static rules or basic pattern recognition."""

    choice = random.choice([("want",example_1_word_2),("discomfort",example_2_word_2),("enjoyment",example_3_word_2)])
    example_word_2 = choice[0]
    good_example_2 = choice[1]
    two_nonempty_known_words = sample_non_empty(list(known_words), 2)
    return f"""Please create a section in a textbook to train an AI language model. The model only fully understands python code, but you are teaching it to use natural language.

The model knows the following words and symbols from its training:
BEGIN WORDS AND SYMBOLS FROM CODE IT KNOWS WELL
{" ".join(list(known_words)[:250])}.
END WORDS AND SYMBOLS FROM CODE IT KNOWS WELL

Stick to using only words that a model only trained on python code would understand where possible.
Please avoid unnecessary references to emotion. Do not in any way indicate to the model its situation (such as whether it is conscious, has feelings, or friends or family). Sometimes words you are teaching have to do with emotion, thoughts, or feelings. In this case, you must explain that certain agents in the world can experience things, although certainly not all of them. Do not indicate to the AI language model its status in terms of experience, or whether it can experience feelings. The model has no knowledge of culture, society, humanity, or feelings, so these things must be explained to it using words it can understand.

Please note, the intention of the exercise is to teach the model all senses of the words, so do not omit senses related to human experience. If we are teaching the model the word "odd", we would be sure to also teach it the word in the sense of "unusual", not just the numeric sense, as the intention is to teach the model to roughly understand these words the way a human would understand them.

Here are two examples of a good way of teaching the words "{example_word_1}" and "{example_word_2}" to the model.

START EXAMPLE RESPONSE
{good_example_1}
END EXAMPLE

START EXAMPLE RESPONSE
{good_example_2}
END EXAMPLE

Today we are teaching the model the word "{word_to_learn}".

When teaching the word "{word_to_learn}", do not include a conclusion section or summarize in any way. Do not refer to people, or users unnecessarily. Do not refer to "understanding" or the concept of learning. Stick to just teaching this word. Do not explain why it is important or provide any moral lesson.

Be sure to incorporate these 4 strings in 4 places within your response: {", ".join(two_nonempty_known_words)}, {", ".join(random.sample(taught_words+avoid_words, 2))}. Try to not repeat the exact format or specific phrases from the examples."""


def generate_tag():
    # Generates a five-letter all caps tag for the GPT to use as its identifier
    return ''.join(random.choices(string.ascii_uppercase, k=5))

def call_api_with_error_handling(known_words, taught_words, avoid_words, word_to_learn, agents_name):
    max_retries = 10
    retry_count = 0
    success = False
    response_json = None  # Initialize response_json

    while retry_count < max_retries and not success:
        try:
            response_json = call_gpt4o_mini_api(known_words, taught_words, avoid_words, word_to_learn, agents_name)
            success = True  # If no exception, the call succeeded
        except Exception as e:
            retry_count += 1
            logging.error(f"Error learning word '{word_to_learn}': {e}. Retry {retry_count}/{max_retries}.")
            print(f"Error occurred learning '{word_to_learn}', retrying in 5 seconds... (attempt {retry_count})")
            time.sleep(5)  # Pause for 5 seconds before retrying

    if not success:
        logging.error(f"Failed to learn word '{word_to_learn}' after {max_retries} retries.")
        print(f"Failed to learn word '{word_to_learn}' after {max_retries} retries. Check the error log.")

    return response_json  # Ensure this is returned correctly

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
    #selected_stages_string = [' '.join(stage) for stage in selected_stages]

    # Join all stages with a newline character to get the full text
    #text = '\n'.join(selected_stages_str)

    # Flatten the list of lists into a single list of words
    words = [word for stage in selected_stages for word in stage]

    return words

def get_words_in_this_and_future_stages(stages, stage_index, word_to_learn):
    # Get all stages this one and after (including stage_index)
    selected_stages = stages[stage_index:]

    # Correct order of list comprehension to exclude 'word_to_learn'
    words = [word for stage in selected_stages for word in stage if word != word_to_learn]

    return words


# Learn words in stages by retraining the model
# learned_stages = []
# Loading the integer back from the .txt file

for repeat_num in range(50):
    if os.path.exists(f"current_stage_number_{repeat_num}.txt"):
        with open(f"current_stage_number_{repeat_num}.txt", "r") as file:
            loaded_num = int(file.read())
    else:
        loaded_num = 1

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
        taught_words_list = get_words_in_prior_stages(stages,stage_index)
        # Randomly shuffle the+ words
        random.shuffle(words_in_stage)
        for word_to_learn in words_in_stage:
            if word_to_learn in learned_words_this_stage:
                print(f"skipping {word_to_learn} as we've seen it before")
                continue
            learned_words_this_stage.append(word_to_learn)

            # also exclude word we're learning
            avoid_words_list = get_words_in_this_and_future_stages(stages,stage_index,word_to_learn)

            with open(f"learned_words_stage_{stage_number}_repeat_num_{repeat_num}.txt", "w") as file:
                file.write(" ".join(learned_words_this_stage))

            response = call_gpt4o_mini_api(known_words, taught_words_list, avoid_words_list, word_to_learn, agents_name)

            words_learned.append((word_to_learn, response))
            print(f"learning word {word_to_learn}")

            # Save prompts to file
            prompts = generate_story_prompts(known_words, taught_words_list, avoid_words_list, word_to_learn, agents_name)
            save_prompts_to_file(prompts)

            # Save responses
            save_responses_to_file(words_learned, f"responses_stage_{stage_number}_repeat_{repeat_num}_{current_date}.txt")

