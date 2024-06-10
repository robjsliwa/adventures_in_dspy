import dspy
from dspy.teleprompt import COPRO


COMPILED_PROGRAM_PATH = 'compiled_program.json'


def validate_dm_response(dm_response):
    return (
        "[Game Name]" not in dm_response
        and "The updated state of the game world after the action:"
        not in dm_response
        and "Game State:" not in dm_response
    )


class DungeonMaster(dspy.Signature):
    """Manage a text adventure game by responding to player actions and describing the game world."""

    player_action = dspy.InputField(desc="The action taken by the player")
    game_state = dspy.InputField(desc="The current state of the game world")
    dm_response = dspy.OutputField(
        desc="The response to the player's action.",
    )
    updated_game_state = dspy.OutputField(
        desc="The updated state of the game world after the action",
    )


class DungeonMasterPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.signature = DungeonMaster
        self.predictor = dspy.ChainOfThought(self.signature)

    def forward(self, player_action, game_state):
        result = self.predictor(
            player_action=player_action,
            game_state=game_state,
        )

        dspy.Suggest(
            validate_dm_response(result.dm_response),
            "dm_response should not contain '[Game Name]' or 'The updated state of the game world after the action:' or 'Game State:'",
        )

        return dspy.Prediction(
            dm_response=result.dm_response,
            updated_game_state=result.updated_game_state,
        )


def load_compiled_program():
    dm_pipeline = None
    try:
        dm_pipeline = DungeonMasterPipeline()
        dm_pipeline.load(COMPILED_PROGRAM_PATH)
        dm_pipeline.activate_assertions()
    except Exception as e:
        print(f"Error loading compiled program: {e}")
    return dm_pipeline


def main():
    ollama_model = dspy.OllamaLocal(model="mistral:latest", max_tokens=1024)
    dspy.settings.configure(lm=ollama_model)

    compiled_program = load_compiled_program()
    if not compiled_program:
        return

    initial_game_state = "You are in a dark forest. There is a path to the north and a cave to the east."

    game_state = initial_game_state
    print("Welcome to DSPy based Text Adventure Game!")
    print(game_state)
    while True:
        player_action = input("\nWhat do you want to do? ")
        if player_action.lower() in ["quit", "exit"]:
            print("Thanks for playing!")
            break

        result = compiled_program(
            player_action=player_action, game_state=game_state
        )

        dm_response = result.dm_response
        game_state = result.updated_game_state
        print("\nDungeon Master: ", dm_response)


if __name__ == '__main__':
    main()
