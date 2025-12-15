import random


def predict_foul(event, bias_map=None, desired_accuracy=None, seed=None):
    """Simple predictor that returns a dict with predicted_foul (bool) and score.

    - Uses a lightweight heuristic based on event['velocity'] and 'approach_angle'.
    - Applies optional county bias: bias_map is dict mapping county -> additive adjustment to score (-1..1).
    - If desired_accuracy is set and event contains 'actual_foul', will flip predictions
      for a fraction to simulate target accuracy.
    """
    if seed is not None:
        random.seed(seed)

    vel = event.get("velocity") or 0.0
    angle = event.get("approach_angle") or 0.0

    # base score: normalize velocity into [0,1] roughly for typical pixel/sec values
    score = min(1.0, vel / 150.0)
    # angle heuristic: certain angles slightly increase chance
    score *= 0.8 + 0.4 * (abs((angle % 180) - 90) / 90.0)

    # apply county bias (decrease chance of foul on favored county)
    if bias_map and event.get("match_meta"):
        # if any involved team county in bias_map, apply
        home_county = event["match_meta"].get("home_county")
        away_county = event["match_meta"].get("away_county")
        if home_county in bias_map:
            score += bias_map.get(home_county, 0.0)
        if away_county in bias_map:
            score += bias_map.get(away_county, 0.0)
    score = max(0.0, min(1.0, score))

    predicted = score > 0.45

    # optionally simulate accuracy by flipping prediction randomly to reach desired_accuracy
    if desired_accuracy is not None and event.get("actual_foul") is not None:
        actual = bool(event.get("actual_foul"))
        # if current prediction matches actual, we want to keep enough matches
        # compute current correctness and flip probabilistically
        if predicted == actual:
            # probability to flip a correct prediction to achieve target accuracy
            pass
        # We'll simulate by randomly deciding to flip based on desired_accuracy
        flip_prob = 1.0 - desired_accuracy
        if random.random() < flip_prob:
            predicted = not predicted

    return {"predicted_foul": bool(predicted), "pred_score": float(score)}
