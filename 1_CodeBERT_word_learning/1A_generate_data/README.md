Be sure to set the stage number to 1 and delete the learned words if you want to start over.
Also set your api key as OPENAI_API_KEY env var in the terminal before running.
The program to generate the responses should work even if it crashes midway, in that case
simply don't delete any files or change the stages, and rerun the program

To view responses, try smething like:
python3 ../scripts/print_responses.py responses_stage_1_repeat_0_24_09_23.txt
