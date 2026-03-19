import os
import tempfile
from unittest.mock import patch, MagicMock
from src.model_pipeline.cd.notifier import NotificationDispatcher

def test_notifier_load_default_when_missing():
    nd = NotificationDispatcher("config/does_not_exist.yaml")
    assert not nd.config.get("slack", {}).get("enabled")
    assert not nd.config.get("email", {}).get("enabled")

def test_notifier_slack_message():
    import yaml
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False) as f:
        yaml.dump({
            "slack": {"enabled": True, "webhook_env_var": "TEST_SLACK_HOOK"}
        }, f)
        temp_path = f.name
        
    try:
        os.environ["TEST_SLACK_HOOK"] = "http://mock-hook.com"
        nd = NotificationDispatcher(config_path=temp_path)
        
        with patch.object(nd, "_send_slack") as mock_slack:
            nd.notify("Test Message", level="INFO")
            mock_slack.assert_called_once_with("Test Message", "INFO")
            
        del os.environ["TEST_SLACK_HOOK"]
    finally:
        os.unlink(temp_path)

def test_notifier_email_message():
    import yaml
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False) as f:
        yaml.dump({
            "email": {"enabled": True, "from_address": "test@test.com"}
        }, f)
        temp_path = f.name
        
    try:
        nd = NotificationDispatcher(config_path=temp_path)
        
        with patch.object(nd, "_send_email") as mock_email:
            nd.notify("Test Alert", level="CRITICAL")
            mock_email.assert_called_once_with("Test Alert", "CRITICAL")
            
    finally:
        os.unlink(temp_path)
