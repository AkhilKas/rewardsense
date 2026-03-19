from unittest.mock import patch, MagicMock
from src.model_pipeline.cd.champion_challenger import ChampionChallenger


@patch("src.model_pipeline.cd.champion_challenger.mlflow.get_experiment_by_name")
@patch("src.model_pipeline.cd.champion_challenger.mlflow.search_runs")
def test_no_champion_allows_promotion(mock_search_runs, mock_get_experiment):
    # Mock empty dataframe from search
    mock_search_runs.return_value.empty = True

    cc = ChampionChallenger()
    assert cc.compare(0.80) is True


@patch("src.model_pipeline.cd.champion_challenger.mlflow.get_experiment_by_name")
@patch("src.model_pipeline.cd.champion_challenger.mlflow.search_runs")
def test_challenger_wins(mock_search_runs, mock_get_experiment):
    # Mock dataframe with a champion metric of 0.70
    mock_df = MagicMock()
    mock_df.empty = False
    mock_df.iloc = [{"metrics.ndcg_at_5": 0.70}]
    mock_search_runs.return_value = mock_df

    cc = ChampionChallenger(metric_to_compare="ndcg_at_5", maximize=True)
    assert cc.compare(0.75) is True


@patch("src.model_pipeline.cd.champion_challenger.mlflow.get_experiment_by_name")
@patch("src.model_pipeline.cd.champion_challenger.mlflow.search_runs")
def test_challenger_loses(mock_search_runs, mock_get_experiment):
    mock_df = MagicMock()
    mock_df.empty = False
    mock_df.iloc = [{"metrics.ndcg_at_5": 0.85}]
    mock_search_runs.return_value = mock_df

    cc = ChampionChallenger(metric_to_compare="ndcg_at_5", maximize=True)
    assert cc.compare(0.80) is False


@patch("src.model_pipeline.cd.champion_challenger.mlflow.get_experiment_by_name")
@patch("src.model_pipeline.cd.champion_challenger.mlflow.search_runs")
def test_minimize_metric(mock_search_runs, mock_get_experiment):
    mock_df = MagicMock()
    mock_df.empty = False
    mock_df.iloc = [{"metrics.rmse": 0.50}]
    mock_search_runs.return_value = mock_df

    cc = ChampionChallenger(metric_to_compare="rmse", maximize=False)
    # Challenger wins because 0.45 < 0.50
    assert cc.compare(0.45) is True
    # Challenger loses because 0.55 > 0.50
    assert cc.compare(0.55) is False
