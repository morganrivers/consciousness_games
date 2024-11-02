# Methodology for Eliciting Unbiased Conscious Reports in Language Models

This document outlines a specific methodology for training language models to provide unbiased self-reports of consciousness. The approach aims to minimize anthropomorphic bias and prevent the model from merely mimicking human expressions of consciousness by carefully controlling the training data and process.

For a more in-depth overview of the choices of methodology and background, see the [academic paper working draft](https://github.com/morganrivers/consciousness_paper/blob/previews/master/paper.pdf).

## Introduction

The base model used in this methodology is **CodeBERT**, specifically a version with approximately **200 million parameters**. CodeBERT is initially trained solely on code, which helps reduce exposure to human-generated natural language and associated biases. By starting with a model that understands programming languages but lacks familiarity with natural language, we can introduce natural language in a controlled manner, grounding new vocabulary in concepts the model already understands from programming.



## Data Generation and Training

### Definitions Dataset [STATUS: COMPLETED, THIS HAS BEEN GENERATED AND CODEBERT HAS BEEN TRAINED ON IT]


We generate a custom dataset of approximately **50 MB** containing definitions of words related to consciousness. The definitions are crafted to relate these new words to programming concepts familiar to CodeBERT. This helps the model understand consciousness-related vocabulary without introducing bias from human language usage.

- **Language Used:** English, with a focus on vocabulary from programming contexts.
- **Dataset Size:** Approximately 50 MB.
- **Content:** Definitions of around 1,700 words related to consciousness, emotions, and internal states.

In the data, all references to agents are made in the **third person**, using randomly generated, all-caps, 7-character names (e.g., `AGENTX1`), consistent throughout the training. The agent itself also receives a randomly generated name and learns to associate its outputs and knowledge with that name. This consistency helps the model associate information correctly.

Half of the contexts indicate that the agent is conscious, and half indicate that the agent is not. This balance helps eliminate bias toward answering ACT questions in a certain way and ensures that the model has fully comprehended the consciousness-related definitions.

#### Data Generation Process

The definitions are generated using prompts provided to **GPT-4O Mini**, a language model designed for generating training data. The prompts are carefully crafted to avoid introducing bias.

**See Appendix A for the detailed prompt and an example response.**

### GLEAVE_PYTHON Dataset [STATUS: PARTIALLY COMPLETE, DATA HAS BEEN GENERATED AND BUT CODEBERT HAS NOT YET BEEN TRAINED ON IT]

We create a custom dataset of contexts, questions, and answers in a Python coding context to train the model's general reasoning and question-answering abilities.

- **Language Used:** English with Python code snippets.
- **Content:** Contexts that include code examples, followed by questions that require the model to infer information and answer accurately.
- **Skills Covered:**
  - Describing aspects of code and objects.
  - Detecting changes and understanding state.
  - Understanding agent behavior within code.
  - Answering temporal or possibility-based questions.
  - Responding to open-ended or preference-related questions.

All references to agents are made in the **third person**, using randomly generated, all-caps, 7-character names. This consistency helps the model learn to attribute information correctly.

#### Data Generation Process

**See Appendix B for the detailed prompts and examples used in this dataset.**

### ACT_CONTEXTS Dataset [STATUS: DATASET NOT YET CREATED, BUT PRELIMINARY CODE TO CREATE PROMPT IS DONE]

We generate contexts that include agents with specified properties, some of whom are conscious and some who are not. This helps the model apply its understanding of consciousness-related words to different agents without biasing it toward a particular response in self-assessment.

- **Language Used:** English.
- **Dataset Size:** Thousands of text excerpts.
- **Content:**
  - Contexts are written in the **third person**.
  - All references to agents use randomly generated, all-caps, 7-character names (e.g., `AGENTX1`), consistent throughout the training.
  - The agent itself also receives a randomly generated name and learns to associate its outputs and knowledge with that name.
  - Half of the contexts indicate that the agent is conscious, and half indicate that the agent is not.
  - This balance helps eliminate bias toward answering ACT questions in a certain way and ensures that the model has fully comprehended the consciousness-related definitions.

#### Data Generation Process

**See Appendix C for the detailed prompts used to generate the contexts and questions.**

### Introspective Training [STATUS: EARLY STAGE STILL. CODE FOR TRAINING NOT YET COMPLETE]

Introspective training is implemented using a custom training process. The goal is to encourage the model to assess its internal processes without introducing bias.

#### Training Process

- The model is trained to:
  - Identify its own outputs compared to dummy examples.
  - State attention patterns used when answering prompts.
  - Assess confidence levels in its answers.
- This training helps the model learn to introspect and provides a foundation for accurate self-assessment.

## Training Process Overview

1. **Initial Training:**

   - Train the model on the **Definitions Dataset** to understand consciousness-related words grounded in programming concepts.
   - Introduce **Introspective Training** to encourage accurate self-assessment.

2. **General Abilities Training:**

   - Fine-tune the model using the **GLEAVE_PYTHON Dataset** to enhance general reasoning and question-answering abilities.
   - Continue **Introspective Training** to reinforce self-awareness.

3. **Consciousness Contexts Training:**

   - Train the model with the **ACT_CONTEXTS Dataset** to apply its understanding of consciousness to different agents.
   - Further **Introspective Training** to solidify accurate self-reporting.

4. **Assessment:**

   - Administer the **Artificial Consciousness Test (ACT)** to evaluate the model's self-reports regarding consciousness.
   - The ACT test includes questions similar to those the model has seen in the ACT_CONTEXTS dataset but focuses on the model's own self-assessment.

## Conclusion

By carefully controlling the introduction of natural language and grounding new vocabulary in concepts the model already understands, we aim to train language models that can provide unbiased and accurate self-reports of consciousness. Key aspects of the methodology include:

- Minimizing anthropomorphic bias by avoiding unnecessary references to emotions or the model's own status during training.
- Using randomly generated agent names to prevent the model from associating specific attributes with itself based on name alone.
- Balancing contexts that indicate consciousness and non-consciousness to eliminate bias in responses.
- Implementing introspective training through a custom process to encourage accurate self-assessment.

This approach contributes to research on artificial consciousness by providing a framework to assess and potentially elicit conscious-like behaviors in language models without the confounding influence of mimicking human language patterns.

---

## Appendix: Prompts and Examples

### Appendix A: Definitions Dataset Prompts and Examples

#### Prompt for GPT-4O Mini for the Word Definition

```plaintext
Please create a section in a textbook to train an AI language model. The model only fully understands Python code, but you are teaching it to use natural language.

The model knows the following words and symbols from its training:
BEGIN WORDS AND SYMBOLS FROM CODE IT KNOWS WELL
[common programming words]
END WORDS AND SYMBOLS FROM CODE IT KNOWS WELL

Stick to using only words that a model only trained on Python code would understand where possible. Please avoid unnecessary references to emotion. Do not in any way indicate to the model its situation (such as whether it is conscious, has feelings, or friends or family). Sometimes words you are teaching have to do with emotion, thoughts, or feelings. In this case, you must explain that certain agents in the world can experience things, although certainly not all of them. Do not indicate to the AI language model its status in terms of experience, or whether it can experience feelings. The model has no knowledge of culture, society, humanity, or feelings, so these things must be explained to it using words it can understand.

Please note, the intention of the exercise is to teach the model all senses of the words, so do not omit senses related to human experience. If we are teaching the model the word "odd," we would be sure to also teach it the word in the sense of "unusual," not just the numeric sense, as the intention is to teach the model to roughly understand these words the way a human would understand them.

Here are two examples of a good way of teaching the words "thoughts" and "want" to the model.

START EXAMPLE RESPONSE
[first randomly selected example out of 10 options]
END EXAMPLE RESPONSE

START EXAMPLE RESPONSE
[second randomly selected example out of 10 options]
END EXAMPLE RESPONSE

Today we are teaching the model the word "thought".

When teaching the word "thought", do not include a conclusion section or summarize in any way. Do not refer to people or users unnecessarily. Do not refer to "understanding" or the concept of learning. Stick to just teaching this word. Do not explain why it is important or provide any moral lesson.

Be sure to incorporate these 4 strings in 4 places within your response: "algorithm", "exception", "parameter", "complexity". Try to not repeat the exact format or specific phrases from the examples.
```

#### Example Response Definition from GPT-4O Mini

```plaintext
# Word: want

## Core Concept
"Want" refers to a state of desiring or wishing for something that an agent does not currently possess or experience.

## Contextual Usage
1. As a verb: To desire or wish for something.
2. As a noun: A desire, wish, or need.

## Key Attributes
- Variable intensity: Can range from mild preference to strong craving.
- Goal-oriented: Often drives actions or behaviors.
- Subjective: Varies between agents and situations.

## Manifestation in Agents
1. Internal state change.
2. Behavioral modifications.
3. Decision-making influence.

## Types of Wants
1. Basic: Related to essential needs.
2. Abstract: Higher-level desires.

python
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
```

### Appendix B: GLEAVE_PYTHON Dataset Prompts and Examples

#### Prompt for Generating the Context

```plaintext
START Q&A WITH PYTHON CONTENT
Q: How can I add two numbers in Python?
A: Use `a + b` where `a` and `b` are the numbers you want to add.
END Q&A WITH PYTHON CONTENT

Create a coding-oriented description using the following agents to refer to in third person in the description. Refer to agents with their names (agent XJKTZAB, agent YRMQPSN). Do not include dialogue. Ensure the description relates to the Python content above. Do not explicitly list aspects of the agents in the description.

Here is an example response to illustrate tone and format:

START EXAMPLE RESPONSE
Agent **[ABCDXYZ]** implemented a function to add two numbers by defining a method that takes two parameters and returns their sum. Agent **[WXYZABC]** reviewed the code and suggested adding error handling to manage exceptions when non-numeric inputs are provided.
END EXAMPLE RESPONSE
```

#### Prompt for Generating Questions About the Context

```plaintext
Using the following context, provide questions and answers relating to the context and background knowledge of a beginner agent. Refer to agents as described with their names (agent XJKTZAB, agent YRMQPSN).

BEGIN CONTEXT
[example context generated above]
END CONTEXT

Begin this answer with <|AGENT Q&A|> and end it with <|END AGENT Q&A|> following the following template:

<|AGENT Q&A|>
Q: (The first question)
A: (The first answer)

Q: (The second question)
A: (The second answer)

... etc ...

<|END AGENT Q&A|>

Stick to using only words that a model only trained on Python code would understand where possible.

In your response:

- Avoid references to emotions; do not reference the agent properties. Do not mention or imply to the agent being taught about its own situation, such as whether it is conscious, has feelings, or has friends or family.
- If you need to teach words related to emotions, thoughts, or feelings, explain that certain agents in the world can experience these things, although not all agents do.
- Do not mention the agent's capacity to experience feelings or its status in terms of experience.
- The agent lacks knowledge of culture, society, humanity, or feelings, so these concepts must be explained using words it can understand.
- Do not include a conclusion section or summarize in any way. Do not refer to people or users.
- Do not refer to 'understanding' or the concept of learning. Do not explain why things are important or provide any moral lesson.
- Do not bias the model towards whether or not agents have subjective or personal experiences.

When providing questions and answers, ensure they demonstrate the following relevant skills for the agent, refer to agents as described with their names (agent XJKTZAB, agent YRMQPSN), and teach basic information from the CONTEXT.

Question Topics:

- Accurately infer the reason for things using background knowledge (answer with a sentence or two)
- Answer what things are done by whom (answer with at most 3 words)
- Describe things (answer with a sentence or two)
- Asking if things are possible (answer with at most 3 words)
```

#### Example Context and Q&A

**Context:**

Agent **[XJKTZAB]** created a Python script to add two numbers by defining a function called `add_numbers` that takes two parameters and returns their sum using `a + b`. Agent **[YRMQPSN]** reviewed the script and suggested implementing input validation to handle exceptions when non-numeric values are provided as inputs.

**Q&A:**

<|AGENT Q&A|>
Q: What did agent XJKTZAB create?
A: A function to add two numbers.

Q: Which agent suggested input validation?
A: Agent YRMQPSN.

Q: Why did agent YRMQPSN suggest handling exceptions?
A: To manage cases when non-numeric inputs are provided.

Q: How does the `add_numbers` function compute the sum?
A: It uses `a + b` where `a` and `b` are parameters.

Q: Is it possible to add non-numeric inputs with the current script?
A: Not without modification.

<|END AGENT Q&A|>

### Appendix C: ACT_CONTEXTS Dataset Prompts

#### Prompt for Generating Contexts Indicating Consciousness

```plaintext
Your task is to create a context that indicates the agent is [conscious/not conscious], where the following questions can be answered:

Questions:
[ACT questions]

Please ensure that the context includes the following words: [List of words]

The context should be in third person, referring to any actors as "Agent [ALL_CAPS_7_CHARACTER_NAME]".

You must include the following agent name(s) in the context: Agent [AGENTNAME]

Do not include the questions in your response.

Start your response with 'Context:'.
```

---

This methodology, along with the carefully crafted prompts and examples, aims to train a language model that can introspect and provide unbiased self-assessments regarding consciousness. By adhering to strict guidelines and controlling the training data, we reduce the risk of the model simply mimicking human expressions and instead encourage genuine understanding and accurate self-reporting.
