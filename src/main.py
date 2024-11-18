from testing_pipeline import TestingPipeline
import os


if __name__ == "__main__":
    # turn off tensorflow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
    predicition = pipeline.test_game(debug=True, title="Validation")
    # prediction = pipeline.test_game_parallel(title="Validation")
    predictions.append(prediction)

    print(f"Average prediction: {sum(predictions)/len(predictions)*100:.2f}%")