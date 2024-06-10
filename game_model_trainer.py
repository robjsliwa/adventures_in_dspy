import re
import dspy
from dspy.teleprompt import COPRO
from dspy.evaluate import Evaluate
from sentence_transformers import SentenceTransformer, util


COMPILED_PROGRAM_PATH = 'compiled_dungeon_master.json'


TRAINING_SET = [
    dspy.Example(
        player_action="Go north",
        game_state="You are in a dark forest. There is a path to the north and a cave to the east.",
        dm_response="You walk north and find yourself at the edge of a serene lake.",
        updated_game_state="You are at the edge of a serene lake. The forest is to the south.",
    ).with_inputs("player_action", "game_state"),
    dspy.Example(
        player_action="Enter the cave",
        game_state="You are in a dark forest. There is a path to the north and a cave to the east.",
        dm_response="You enter the cave and find it filled with glittering crystals.",
        updated_game_state="You are inside a cave filled with glittering crystals. The forest is to the west.",
    ).with_inputs("player_action", "game_state"),
    dspy.Example(
        player_action="Look around",
        game_state="You are at the edge of a serene lake. The forest is to the south.",
        dm_response="You see a small boat tied to a dock and a path leading into the forest.",
        updated_game_state="You are at the edge of a serene lake. There is a small boat tied to a dock and a path leading into the forest.",
    ).with_inputs("player_action", "game_state"),
    dspy.Example(
        player_action="Take the boat",
        game_state="You are at the edge of a serene lake. There is a small boat tied to a dock and a path leading into the forest.",
        dm_response="You untie the boat and start rowing across the lake.",
        updated_game_state="You are rowing across the lake. The shore is getting farther away.",
    ).with_inputs("player_action", "game_state"),
    dspy.Example(
        player_action="Go back to the forest",
        game_state="You are at the edge of a serene lake. There is a small boat tied to a dock and a path leading into the forest.",
        dm_response="You walk back into the forest.",
        updated_game_state="You are in a dark forest. There is a path to the north and a cave to the east.",
    ).with_inputs("player_action", "game_state"),
    dspy.Example(
        player_action="Climb a tree",
        game_state="You are in a dark forest. There is a path to the north and a cave to the east.",
        dm_response="You climb a tree and get a better view of the surroundings.",
        updated_game_state="You are in a dark forest, perched on a tree. There is a path to the north and a cave to the east.",
    ).with_inputs("player_action", "game_state"),
    dspy.Example(
        player_action="Go east",
        game_state="You are in a dark forest. There is a path to the north and a cave to the east.",
        dm_response="You walk east and enter the cave.",
        updated_game_state="You are inside a cave filled with glittering crystals. The forest is to the west.",
    ).with_inputs("player_action", "game_state"),
    dspy.Example(
        player_action="Swim in the lake",
        game_state="You are at the edge of a serene lake. There is a small boat tied to a dock and a path leading into the forest.",
        dm_response="You dive into the lake and start swimming.",
        updated_game_state="You are swimming in the lake. The water is cool and refreshing.",
    ).with_inputs("player_action", "game_state"),
    dspy.Example(
        player_action="Light a fire",
        game_state="You are in a dark forest. There is a path to the north and a cave to the east.",
        dm_response="You gather some wood and light a fire.",
        updated_game_state="You are in a dark forest with a fire burning. There is a path to the north and a cave to the east.",
    ).with_inputs("player_action", "game_state"),
    dspy.Example(
        player_action="Explore the cave",
        game_state="You are inside a cave filled with glittering crystals. The forest is to the west.",
        dm_response="You explore deeper into the cave and find a hidden chamber.",
        updated_game_state="You are in a hidden chamber inside the cave. The main cave is to the west.",
    ).with_inputs("player_action", "game_state"),
    dspy.Example(
        player_action="Go south",
        game_state="You are at the edge of a serene lake. There is a small boat tied to a dock and a path leading into the forest.",
        dm_response="You walk south and re-enter the forest.",
        updated_game_state="You are in a dark forest. There is a path to the north and a cave to the east.",
    ).with_inputs("player_action", "game_state"),
    dspy.Example(
        player_action="Pick up a crystal",
        game_state="You are inside a cave filled with glittering crystals. The forest is to the west.",
        dm_response="You pick up a crystal and it glows softly in your hand.",
        updated_game_state="You are inside a cave filled with glittering crystals, holding a glowing crystal. The forest is to the west.",
    ).with_inputs("player_action", "game_state"),
    dspy.Example(
        player_action="Rest by the lake",
        game_state="You are at the edge of a serene lake. There is a small boat tied to a dock and a path leading into the forest.",
        dm_response="You sit down and rest by the lake, enjoying the peaceful surroundings.",
        updated_game_state="You are resting by the lake. The forest is to the south.",
    ).with_inputs("player_action", "game_state"),
    dspy.Example(
        player_action="Search the forest",
        game_state="You are in a dark forest. There is a path to the north and a cave to the east.",
        dm_response="You search the forest and find some berries.",
        updated_game_state="You are in a dark forest, holding some berries. There is a path to the north and a cave to the east.",
    ).with_inputs("player_action", "game_state"),
    dspy.Example(
        player_action="Go west",
        game_state="You are in a dark forest. There is a path to the north and a cave to the east.",
        dm_response="You walk west and find yourself back at the edge of the forest.",
        updated_game_state="You are at the edge of the forest. There is a path to the north and a cave to the east.",
    ).with_inputs("player_action", "game_state"),
    dspy.Example(
        player_action="Build a shelter",
        game_state="You are in a dark forest. There is a path to the north and a cave to the east.",
        dm_response="You gather some branches and build a small shelter.",
        updated_game_state="You are in a dark forest with a small shelter. There is a path to the north and a cave to the east.",
    ).with_inputs("player_action", "game_state"),
    dspy.Example(
        player_action="Fish in the lake",
        game_state="You are at the edge of a serene lake. There is a small boat tied to a dock and a path leading into the forest.",
        dm_response="You cast a line into the lake and wait for a bite.",
        updated_game_state="You are fishing in the lake. The forest is to the south.",
    ).with_inputs("player_action", "game_state"),
    dspy.Example(
        player_action="Go north",
        game_state="You are in a dark forest. There is a path to the north and a cave to the east.",
        dm_response="You walk north and find yourself at the edge of a serene lake.",
        updated_game_state="You are at the edge of a serene lake. The forest is to the south.",
    ).with_inputs("player_action", "game_state"),
    dspy.Example(
        player_action="Gather firewood",
        game_state="You are in a dark forest. There is a path to the north and a cave to the east.",
        dm_response="You gather some firewood from the forest floor.",
        updated_game_state="You are in a dark forest with some firewood. There is a path to the north and a cave to the east.",
    ).with_inputs("player_action", "game_state"),
    dspy.Example(
        player_action="Go east",
        game_state="You are in a dark forest. There is a path to the north and a cave to the east.",
        dm_response="You walk east and enter the cave.",
        updated_game_state="You are inside a cave filled with glittering crystals. The forest is to the west.",
    ).with_inputs("player_action", "game_state"),
]


# ollama_model = dspy.OllamaLocal(
#     model="mistral",
#     max_tokens=1024,  # optional
#     temperature=0.5,  # optional
# )


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

        return dspy.Prediction(
            dm_response=result.dm_response,
            updated_game_state=result.updated_game_state,
        )


def detect_unwanted_patterns(text):
    unwanted_patterns = [
        r"Game State:",
        r"Player Action:",
        r"Reasoning:",
        r"Dm Response:",
        r"I'd love to help",
        r"Here's my attempt at completing the format",
        r":",
    ]

    combined_pattern = re.compile("|".join(unwanted_patterns), re.IGNORECASE)
    match = combined_pattern.search(text)
    return bool(match)


def semantic_similarity(example, pred, trace=None):
    if pred.dm_response == "" or pred.updated_game_state == "":
        return False

    if detect_unwanted_patterns(pred.dm_response) or detect_unwanted_patterns(
        pred.updated_game_state
    ):
        return False

    model = SentenceTransformer("stsb-roberta-large")
    response_similarity = util.pytorch_cos_sim(
        model.encode(example.dm_response), model.encode(pred.dm_response)
    )
    state_similarity = util.pytorch_cos_sim(
        model.encode(example.updated_game_state),
        model.encode(pred.updated_game_state),
    )

    score = (response_similarity + state_similarity) / 2
    score_number = float(score[0][0])

    return score_number >= 0.4


metric = semantic_similarity

ollama_model_1 = dspy.OllamaLocal(model='llama3', max_tokens=100)
ollama_model_2 = dspy.OllamaLocal(model='mistral:latest', max_tokens=100)

eval_kwargs = dict(num_threads=1, display_progress=True, display_table=0)
copro_optimizer = COPRO(metric=metric, breadth=10, depth=7, track_stats=True)
optimizer = copro_optimizer

print("Training the game model: llama3")
dspy.settings.configure(lm=ollama_model_1)
optimized_program_1 = optimizer.compile(
    DungeonMasterPipeline(),
    trainset=TRAINING_SET,
    eval_kwargs=eval_kwargs,
)

print("Training the game model: mistral")
dspy.settings.configure(lm=ollama_model_2)
optimized_program_2 = optimizer.compile(
    DungeonMasterPipeline(),
    trainset=TRAINING_SET,
    eval_kwargs=eval_kwargs,
)

evaluator = Evaluate(devset=TRAINING_SET, metric=metric, display_progress=True)
print("Evaluating the optimized programs with llama3")
dspy.settings.configure(lm=ollama_model_1)
result_1 = evaluator(optimized_program_1)
print("Evaluating the optimized programs with mistral")
dspy.settings.configure(lm=ollama_model_2)
result_2 = evaluator(optimized_program_2)

best_model = ollama_model_1 if result_1 > result_2 else ollama_model_2
best_program = (
    optimized_program_1 if result_1 > result_2 else optimized_program_2
)
best_program.save(COMPILED_PROGRAM_PATH)
print(f"Best model: {best_model.model_name}")
