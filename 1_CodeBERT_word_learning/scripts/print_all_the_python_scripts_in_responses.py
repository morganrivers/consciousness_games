import argparse
import re

def main():
    # Parse Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Specify the file name to be printed")
    args = parser.parse_args()

    # Sample string (you can read from a file instead)
    with open(args.filename, 'r') as file:
        content = file.read()

    # Regex pattern to capture content between triple backticks
    pattern = re.compile(r'```(.*?)```', re.DOTALL)

    # Find all matches and print the content between backticks
    matches = pattern.findall(content)

    # Print each match
    for match in matches:
        print(match.strip())


if __name__ == "__main__":
    main()

