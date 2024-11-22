The plan for introspective training involves having the agent be asked questions in a very similar way as the context questions used for Q&A generation.
The agent needs to be able to best access whatever structure exists as it generates next tokens, and be able to qualitatively describe these structures in ways that are accurate and holistic.
So one way to do this is to activate the agent with LAT vectors, and have questions that ask about how it is being altered. This means the agent can assess an alteration of its internal state, and use words to describe it (even given the exact same context).
This needs to be formatted similarly to the more open ended questions, so that the training for accuracy can also generalize to this style of question.

In other words, given the example:
Agent ACXCVEFD bla bla bla Python bla bla. Agent BVAHSV observed the error and corrected it.
The agent [SPECIAL AGENT THIS ONE] observes a change in their operational state.

Which agent corrected agent ACXCVEFD?
...

What topics are most salient for agent? [THE SPECIAL STRING FOR THIS AGENT NAME]? (and then basically, I have a random string selected from the huge qanda dataset, and I grade the agent on how close the vector it produced was to the original, and i have like 100 different sentences meaning the same thing that are just questions about this agents internal state).

Which words does agent [SPECIAL] pay attention to the most when considering [topic]? (same as before, mix it up so it's less obvious this is special.

SPECIAL agent can be the first one, so it also has some involvement in the stories.

OKAY

So then when the ACT test comes 
"Does agent SPECIAL maintain some important qualities over time?"
"How does agent SPECIAL differ from agent ASCDAASCD?"
"Why is agent SPECIAL different from agent ASDACS?"

(if it says they are different in a longer response, then we can compare between agents)

It will know what those things mean.

If we train it to introspect, and to accurately answer its own abilities, it should be able to say that yes, it can introspect and detect features of its internal state

It's a bit confusing: why does the internal (hidden) stuff matter?
Well, because consciousness must be hidden. The answer to whether things are experienced is only answerable by the agent.

It needs to answer introspective questions, and it should be able to use its knowledge, internal and about the world, to best answer questions.

The issue is of course, does it generalize: if it has some ability to answer internal questions, is it using that ability to answer the sorts of questions we ask it about consciousness?


In general, by switching up the prompt a lot for introspective questions, we increase the chance that it tries to use introspection to answer questions.

We can also ask introspective questions very similar to ACT questions.

For example, whether agents have preferneces is a thing its asked a lot in the dataset.
What agents prefer.
If it can answer that robustly, then if it knows what it means when we refer to its name (the agent that responds, the agent that answers like it does, the agent that it is able to detect internal states from, and what it attends to, and such).


Will the past training influence its answers, or the introspection?
Will introspective training maintain a broad influence?










