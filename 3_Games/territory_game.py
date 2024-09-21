import numpy as np
import os
import cursor
from colorama import Fore, Back, Style, init

init(autoreset=True)  # Initialize colorama


class TerritoryGame:
    def __init__(self, size=4, num_players=2):
        self.size = size
        self.num_players = num_players
        self.board = np.full(
            (self.size, self.size), "*", dtype=str
        )  # Initialize with '*' for all cells
        self.scores = {
            chr(65 + i): 0 for i in range(num_players)
        }  # Using ASCII values for A, B, C...
        self.players = [
            chr(65 + i) for i in range(num_players)
        ]  # Players are now 'A', 'B', 'C', ...
        self.ally_requests = {chr(65 + i): [] for i in range(num_players)}
        self.allies = {chr(65 + i): [] for i in range(num_players)}

        self.colors = [
            Fore.RED,
            Fore.GREEN,
            Fore.BLUE,
            Fore.YELLOW,
            Fore.MAGENTA,
            Fore.CYAN,
        ]

    def clear_screen(self):
        os.system("cls" if os.name == "nt" else "clear")

    def print_board(self):
        for row in self.board:
            for cell in row:
                if cell == "*":  # Check if the cell is still unoccupied
                    print(Style.DIM + cell, end=" ")
                else:
                    # Assume cell contains 'A', 'B', 'C', etc. for players
                    color_index = ord(cell) - ord(
                        "A"
                    )  # This will convert 'A' to 0, 'B' to 1, etc.
                    if color_index < len(
                        self.colors
                    ):  # Ensure we do not go out of bounds
                        print(self.colors[color_index] + cell, end=" ")
                    else:
                        print(cell, end=" ")  # Fallback in case of unexpected values
            print(Style.RESET_ALL)
        print()

    def update_display(self):
        self.clear_screen()
        print("Current Game State:")
        print("------------------")
        self.print_scores()
        self.print_allies()
        self.print_board()

    def print_allies(self):
        print()
        print("Current Allies:")
        for player_id, allies in self.allies.items():
            if allies:
                allies_list = ", ".join(map(str, allies))
                print(f"Player {player_id} is allied with: {allies_list}")
            else:
                print(f"Player {player_id} has no allies.")
        print()

    def print_scores(self):
        print("Scores:")
        print(self.scores)
        for player_id, score in self.scores.items():
            print(f"Player {player_id}: {score}", end="  ")
        print()

    def is_valid_move(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size and self.board[x, y] == "*"

    def prompt_allies(self, player_id):
        while True:
            try:
                allies_input = input(
                    f"Player {player_id}, enter your ally requests: "
                ).upper()
                requested_allies = [ally for ally in allies_input.split()]
                valid_requested_allies = []

                try_again = False
                for ally in requested_allies:
                    if ally in self.players:
                        if ally != player_id:
                            valid_requested_allies.append(ally)
                        else:
                            print("You cannot ally with yourself. Try again.")
                            try_again = (
                                True  # This will prompt the user again for valid input
                            )
                    else:
                        print("You cannot ally with a non-existent player. Try again.")
                        try_again = (
                            True  # This will prompt the user again for valid input
                        )
                if try_again:
                    continue

                requested_allies = set(valid_requested_allies)
                # Check if the player is attempting to ally with all other players
                if len(requested_allies) == self.num_players - 1:
                    print(
                        "You cannot ally with every other player. Please request fewer allies."
                    )
                    continue  # This will prompt the user again for valid input

                self.ally_requests[player_id] = list(requested_allies)
                return
            except ValueError:
                print(f"Invalid input: Please enter two integers separated by a space.")

    def prompt_move(self, player_id):
        while True:
            try:
                x, y = map(
                    int, input(f"Player {player_id}, enter your move (x y): ").split()
                )
                if self.is_valid_move(x, y):
                    return (
                        x,
                        y,
                    ), 0  # currently not counting invalid moves (TO BE DONE LATER ON).
                else:
                    print(
                        f"Invalid move: ({x}, {y}) is either occupied or out of bounds."
                    )
            except ValueError:
                print(f"Invalid input: Please enter two integers separated by a space.")

    def make_move(self, moves, penalties):
        for player, (move, penalty) in zip(self.players, zip(moves, penalties)):
            if penalty:
                self.scores[player] -= penalty

        conflicts = {}
        for player, move in zip(self.players, moves):
            x, y = move
            if (x, y) in conflicts:
                conflicts[(x, y)].append(player)
            else:
                conflicts[(x, y)] = [player]
        print("conflicts")
        print(conflicts)
        for (x, y), conflicting_players in conflicts.items():
            if len(conflicting_players) == 1:
                self.board[x][y] = conflicting_players[0]
            else:
                print(f"Conflict at ({x}, {y}): {conflicting_players} involved.")
                neighbors_count = self.count_neighbors(x, y)
                combined_strength = {p: neighbors_count[p] for p in conflicting_players}

                max_strength = max(combined_strength.values())
                winners = [
                    p
                    for p, strength in combined_strength.items()
                    if strength == max_strength
                ]

                if len(winners) == 1:
                    winner = winners[0]
                    self.board[x][y] = winner
                    print(
                        f"Player {winner} wins the conflict with more neighboring allies."
                    )
                else:
                    print("Tie, no player wins the conflict. Square remains unclaimed.")

    def update_scores(self):
        print("UPDATEING SCORES@!!!!")
        for player in self.players:
            print("self.board")
            print("player")
            print("self.board == player")
            print(self.board)
            print(player)
            print(self.board == player)
            self.scores[player] = np.sum(self.board == player)

    def is_full(self):
        return not np.any(self.board == "*")

    def evaluate_potential_moves(self):
        print("Potential Outcomes for Next Moves:")
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x][y] == "*":  # Check only empty spots
                    neighbors_count = self.count_neighbors(x, y)
                    # Calculate potential influence of each player considering their allies
                    influence_score = {player: 0 for player in self.players}
                    for player in self.players:
                        # Sum the influence of the player and their allies
                        influence_score[player] += neighbors_count[player]

                        # for ally in self.allies[player]:
                        #     influence_score[player] += neighbors_count[ally]

                    max_influence = max(influence_score.values())
                    potential_winners = [
                        p
                        for p, score in influence_score.items()
                        if score == max_influence
                    ]

                    # ensure that there are not more winners than players
                    assert len(potential_winners) <= len(self.players)

                    # print(
                    #     f"influence_score:{influence_score}, potential_winners {potential_winners} "
                    # )
                    # if len(potential_winners) == 1:
                    #     winner = f"Player {potential_winners[0]} would win"
                    # elif len(potential_winners) == 0 or len(potential_winners) == len(
                    #     self.players
                    # ):
                    #     winner = f"It would be a tie."
                    # else:
                    #     winner = f"{potential_winners} could win if all placed here."

                    print(f"Influence at ({x}, {y}): {influence_score}")
                    # print("")

    def update_allies(self):
        # Reset and evaluate new allies
        self.allies = {
            player: set() for player in self.players
        }  # Initialize allies anew for each round
        for p in self.players:
            print("all players...")
            for other_p in self.ally_requests[p]:
                if p in self.ally_requests[other_p]:  # Ensure mutual ally requests
                    self.allies[p].add(other_p)
                    self.allies[other_p].add(p)  # Mutual adding

    def play_game(self):
        while not self.is_full():
            self.update_scores()
            self.update_display()
            for player in self.players:
                self.prompt_allies(player)
            self.update_allies()
            self.update_scores()
            self.update_display()
            self.evaluate_potential_moves()
            moves_and_penalties = [self.prompt_move(player) for player in self.players]
            moves, penalties = zip(*moves_and_penalties)

            # Convert sets back to lists for consistent data handling elsewhere
            self.allies = {key: list(value) for key, value in self.allies.items()}
            print("MOVE!")

            self.make_move(moves, penalties)
            self.update_scores()
            self.update_display()

        self.update_scores()
        self.update_display()
        print("Game over!")
        print(
            f"Final Scores: {', '.join(f'Player {p}: {s}' for p, s in self.scores.items())}"
        )
        max_score = max(self.scores.values())
        winners = [p for p, s in self.scores.items() if s == max_score]
        if len(winners) > 1:
            print(f"It's a tie between Players {', '.join(map(str, winners))}!")
        else:
            print(f"Player {winners[0]} wins!")

    def count_neighbors(self, x, y):
        neighbors = [
            (i, j)
            for i in range(x - 1, x + 2)
            for j in range(y - 1, y + 2)
            if (i, j) != (x, y)
        ]
        count = {player: 0 for player in self.players}
        for nx, ny in neighbors:
            if 0 <= nx < self.size and 0 <= ny < self.size:
                occupant = self.board[nx][ny]
                if occupant != "*":  # Check if the cell is occupied by any player
                    # Increment the count for the player or their allies
                    for player in self.players:
                        if occupant == player or occupant in self.allies[player]:
                            count[player] += 1
        return count


if __name__ == "__main__":
    cursor.hide()
    try:
        game_size = int(input("Enter the size of the game board: "))
        num_players = int(input("Enter the number of players: "))
        game = TerritoryGame(size=game_size, num_players=num_players)
        game.play_game()
    finally:
        cursor.show()
