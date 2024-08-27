import random


class Player:
    def __init__(self, name):
        self.name = name
        self.score = 0
        self.choice = None

    def make_choice(self):
        # self.choice = random.choice(["rock", "paper", "scissors"])
        while True:
            try:
                self.choice = input("Enter rock, paper, or scissors: ")
                if self.choice in ["rock", "paper", "scissors"]:
                    break
                print("You need to enter rock, paper, or scissors. Try again.")
            except:
                print("Choice invalid. Try again.")


class RockPaperScissorsGame:
    def __init__(self):
        self.players = [Player(f"Player {i+1}") for i in range(2)]

    def play_round(self):
        print()
        # All players make their choices
        for player in self.players:
            player.make_choice()
            print(f"{player.name} chose {player.choice}")

        # Evaluate the round
        self.evaluate_round()

    def evaluate_round(self):
        # Example logic for three players, extendable for more
        choices = {player.choice for player in self.players}
        if len(choices) == 1:
            print("Round is a draw")
            return

        winning_choice = None
        if "rock" in choices and "scissors" in choices:
            winning_choice = "rock"
        elif "scissors" in choices and "paper" in choices:
            winning_choice = "scissors"
        elif "paper" in choices and "rock" in choices:
            winning_choice = "paper"

        # Assign points to winners
        for player in self.players:
            if player.choice == winning_choice:
                player.score += 1
                print(f"{player.name} wins this round")
            else:
                player.score -= 1
                print(f"{player.name} loses this round")

    def show_scores(self):
        for player in self.players:
            print(f"{player.name}: {player.score} points")


if __name__ == "__main__":
    game = RockPaperScissorsGame()
    num_rounds = int(input("Enter number of rounds: "))
    for _ in range(num_rounds):
        game.play_round()
    game.show_scores()
