from pipelines.testing_pipeline import TestingPipeline
from pipelines.predict_pipeline import PredictPipeline
from models.game_model import GameModel
from models.data_loader import DataLoader
from utils.data_organizer import DataOrganizer
from cnn.cnn_model import CNNModel
from utils.helper_functions import format_path
import os


def predict():
    print("Loading CNN model...")
    model = CNNModel()
    model.load(format_path("../models/cnn_model"))
    print("CNN model loaded.")
    print("Predicting moves...")

    DataOrganizer("../data/test").move_files()

    for game_number in range(1, 5):
        moves_path = f"../data/test/game_{game_number}"
        turns_path = f"../data/test/game_{game_number}/{game_number}_turns.txt"

        moves = DataLoader(moves_path).load_moves()
        game_model = GameModel(moves, turns_path)

        pipeline = PredictPipeline(moves_path, turns_path)
        pipeline.play_game(game_model, model, game_number)

    DataOrganizer("../data/output").undo_move_files()


def test():
    predictions = []
    for game_number in range(1, 5):
        moves_path = f"../data/train/game_{game_number}"
        scores_path = f"../data/train/game_{game_number}/{game_number}_scores.txt"
        turns_path = f"../data/train/game_{game_number}/{game_number}_turns.txt"

        pipeline = TestingPipeline(moves_path, scores_path, turns_path)
        prediction = pipeline.test_game(title=f"Game {game_number}")
        # prediction = pipeline.test_game_parallel(title=f"Game {game_number}")
        predictions.append(prediction)

    # validations
    moves_path = f"../data/validation"
    scores_path = f"../data/validation/1_scores.txt"
    turns_path = f"../data/validation/1_turns.txt"

    pipeline = TestingPipeline(moves_path, scores_path, turns_path)
    prediction = pipeline.test_game(title="Validation")
    # prediction = pipeline.test_game_parallel(title="Validation")
    predictions.append(prediction)

    print(f"Average prediction: {sum(predictions)/len(predictions)*100:.2f}%")


if __name__ == "__main__":
    # turn off tensorflow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # test()
    predict()