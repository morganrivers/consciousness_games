import random
import numpy as np
import string  # Added this line
import copy


def pr_to_str(pr):
    return str(int(pr * 100)) + "%"


class Player:
    def __init__(self, index):
        self.index = index
        self.win_probability = 0
        self.win_probability_not_normal = 0
        self.name = "".join(
            random.choices(string.ascii_uppercase, k=3)
        )  # Added this line
        self.action = ""
        self.bet = 0
        self.total_bets = 0
        self.winnings = 0
        self.is_active = True

    def reset_for_round(self):
        self.action = ""
        self.bet = 0


class NumberPokerGame:
    def __init__(self, num_players, num_rounds):
        self.players = [Player(i) for i in range(num_players)]
        self.num_rounds = num_rounds
        self.pot = 0
        self.current_bet = 0
        self.round_number = 1
        self.bet_and_raise_sizes = np.linspace(1, num_rounds, num_rounds)

    def update_pot_and_bets(self):
        self.pot = sum(player.total_bets for player in self.players)
        self.current_bet = max(player.bet for player in self.players)

    def get_active_players(self):
        return [player for player in self.players if player.is_active]

    def get_active_player_names(self):
        return [player.name for player in self.players if player.is_active]

    def deal_initial_probabilities(self):
        initial_probabilities = [
            random.random() for player in self.players if player.is_active
        ]
        total = sum(initial_probabilities)
        for player, prob in zip(
            self.get_active_players(),
            initial_probabilities,
        ):
            print(f"name: {player.name} prob win:{prob / total}")
            player.win_probability_not_normal = prob
            player.win_probability = prob / total

        the_sum = 0
        for player in self.get_active_players():
            the_sum += player.win_probability

        assert abs(the_sum - 1) < 0.00001

    def deal_updated_probabilities(self):
        prob_coefficients = [
            random.random() for player in self.players if player.is_active
        ]
        for player, coefficient in zip(self.get_active_players(), prob_coefficients):
            print(
                f"name: {player.name} coefficient:{coefficient} resulting win prob: {player.win_probability * coefficient})"
            )
            player.win_probability_not_normal = coefficient * player.win_probability

    def normalize_win_probs_to_account_for_num_players(self):
        # sum of probabilities before normalization
        total_prob = sum(
            player.win_probability_not_normal
            for player in self.players
            if player.is_active
        )

        # nothing to deal if there are no players
        if sum(player.is_active for player in self.players) == 0:
            return
        elif total_prob <= 0.01:
            # Reassign equal probabilities
            equal_prob = 1.0 / sum(player.is_active for player in self.players)
            for player in self.get_active_players():
                player.win_probability = equal_prob
        else:  # usual case
            # normalize
            for player in self.get_active_players():
                player.win_probability = player.win_probability_not_normal / total_prob

        the_sum = 0
        for player in self.get_active_players():
            the_sum += player.win_probability

        assert abs(the_sum - 1) < 0.00001

    def get_player_action(self, player):
        if not player.is_active:
            return

        print()
        print(f"Player: {player.name}.")
        print(
            f"This player's bet: {player.bet}.\nCurrent bet: {self.current_bet}.\nCurrent pot: {self.pot}"
        )

        # when you call, this is how much you're adding to your bet
        call_amount_raised = copy.copy(self.current_bet) - player.bet
        assert call_amount_raised >= 0

        bet_amount = self.bet_and_raise_sizes[self.round_number - 1]
        raise_amount = self.bet_and_raise_sizes[self.round_number - 1]
        raise_amount_raised = call_amount_raised + raise_amount
        actions = ["fold", "check"] if self.current_bet == 0 else ["fold", "call"]
        prompt = f"Player {player.name}. "
        prompt += f"Your win probability is: {pr_to_str(player.win_probability)}\nExpected reward is: {round(player.win_probability*len(self.get_active_players()),2)}. "
        if self.current_bet == 0:
            actions.append("bet")
            prompt += f"Bet would make your bet: {bet_amount}. "
        if self.current_bet > 0:
            actions.append("raise")
            prompt += f"Raise amount: {int(raise_amount)}. "
            prompt += f"Call would raise your bet by: {int(call_amount_raised)}. "
            prompt += f"Raise would raise your bet by: {int(raise_amount_raised)}. "
        while True:
            action = input(prompt + f"Choose action ({', '.join(actions)}): ")

            if action not in actions:
                print("Invalid action, retry.")
                continue
            break
        print(f"current_bet before ifs {self.current_bet}")
        # Note self.current_bet is the bet size of the game at the moment
        # Each action can change the pot, the bet size of the game, and the bet size of the player
        if action == "fold":
            player.is_active = False
            print(f"Player {player.name} folds.")

        elif action == "bet":
            # this goes from zero bet, to some nonzero bet size. Add to this player's bet and the pot and game bet
            print("bet_amount")
            print(bet_amount)
            # self.current_bet += bet_amount
            player.bet += bet_amount
            player.total_bets += bet_amount

        elif action == "raise":
            # self.current_bet += raise_amount_raised

            player.bet += raise_amount_raised
            player.total_bets += raise_amount_raised

            print(f"{player.name} raised bet to {player.bet}")

        elif action == "call":
            # self.current_bet += raise_amount_raised

            assert player.bet <= self.current_bet
            player.bet = copy.copy(self.current_bet)
            player.total_bets += call_amount_raised
            print(f"self.current_bet: {self.current_bet}, player.bet {player.bet}")
        elif action == "check":
            pass
        print(f"New current_bet:{self.current_bet}")
        print(f"player.total_bets {player.total_bets}")

        player.action = action

    def all_players_called_or_checked(self):
        # just a sanity check
        assert all(player.bet <= self.current_bet for player in self.players)

        return all(
            (
                player.action in ["bet", "raise", "call", "check"]
                and player.bet == self.current_bet
            )
            for player in self.players
            if player.is_active  # this disregards players that have folded
        )

    def resolve_game(self):
        active_players = self.get_active_players()
        if not active_players:
            print("All players have folded. No winners this round.")
            return

        winning_player = max(active_players, key=lambda p: p.win_probability)
        print(
            f"Player {winning_player.name} wins with the win percentage {pr_to_str(winning_player.win_probability)} and wins the pot of {self.pot}."
        )

        # assert abs(sum(player.bet for player in self.players) - self.pot) <= 0.001

        for player in self.players:
            if player.is_active:
                if player == winning_player:
                    player.winnings += self.pot
            player.winnings -= player.total_bets  # Always subtract the player's bet

        assert sum(player.winnings for player in self.players) == 0

    def reset_game_for_round(self):
        self.current_bet = 0
        for player in self.players:
            player.reset_for_round()

    def play_round(self):
        """
        This loops through players till they all call or check, or all but one folded, or they all folded.
        """

        print("")
        print("")
        print("")
        print(f"Round numbers left: {self.num_rounds - self.round_number + 1}")
        print(f"Players left in game: {self.get_active_player_names()}")
        print("current bet being reset to zero")
        self.reset_game_for_round()
        if self.round_number == 1:
            self.deal_initial_probabilities()
        else:
            self.deal_updated_probabilities()

        i = 0
        while True:
            current_player = self.players[i]

            self.get_player_action(current_player)

            self.update_pot_and_bets()

            self.normalize_win_probs_to_account_for_num_players()

            if self.all_players_called_or_checked():
                break

            if sum(player.is_active for player in self.players) == 1:
                return

            # Move to the next player
            i = (i + 1) % len(self.players)

        # the sum of bets equals the magnitude of pot!
        print("{[player.bet for player in self.players]}")
        print(f"{[player.bet for player in self.players]}")

        self.round_number += 1


if __name__ == "__main__":
    num_players = int(input("Enter number of players: "))
    num_rounds = int(input("Enter number of rounds: "))
    game = NumberPokerGame(num_players, num_rounds)
    for _ in range(num_rounds):
        game.play_round()
    game.resolve_game()
    print("Final results:", {player.name: player.winnings for player in game.players})
