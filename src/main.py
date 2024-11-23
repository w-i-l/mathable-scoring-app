from testing_pipeline import TestingPipeline
from game_model import GameModel
from data_loader import DataLoader
from cnn.cnn_model import CNNModel
from util import format_path
import os


def predict():
    model = CNNModel()
    model.load("../models/cnn_model")

    for game_number in range(1, 5):
        moves_path = f"../data/train/game_{game_number}"
        scores_path = f"../data/train/game_{game_number}/{game_number}_scores.txt"
        turns_path = f"../data/train/game_{game_number}/{game_number}_turns.txt"

        moves = DataLoader(moves_path).load_moves()
        game_model = GameModel(moves, turns_path, scores_path)

        pipeline = TestingPipeline(moves_path, scores_path, turns_path)
        scores = pipeline.play_game(game_model, model, game_number)

        with open(format_path(f"../data/output/game_{game_number}/{game_number}_scores.txt"), "w+") as file:
            for score in scores:
                file.write(f"Player{str(score[0])} {str(score[1])} {str(score[2])}\n")


def test():
    predictions = []
    for game_number in range(1, 5):
        moves_path = f"../data/train/game_{game_number}"
        scores_path = f"../data/train/game_{game_number}/{game_number}_scores.txt"
        turns_path = f"../data/train/game_{game_number}/{game_number}_turns.txt"

        pipeline = TestingPipeline(moves_path, scores_path, turns_path)
        prediction = pipeline.test_game(debug=True, title=f"Game {game_number}")
        # prediction = pipeline.test_game_parallel(title=f"Game {game_number}")
        predictions.append(prediction)

    # validations
    moves_path = f"../data/validation"
    scores_path = f"../data/validation/1_scores.txt"
    turns_path = f"../data/validation/1_turns.txt"

    pipeline = TestingPipeline(moves_path, scores_path, turns_path)
    prediction = pipeline.test_game(debug=True, title="Validation")
    # prediction = pipeline.test_game_parallel(title="Validation")
    predictions.append(prediction)

    print(f"Average prediction: {sum(predictions)/len(predictions)*100:.2f}%")


if __name__ == "__main__":
    # turn off tensorflow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # test()
    predict()