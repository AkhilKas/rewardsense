"""
RewardSense - Pipeline Monitoring & Alerting Module

Provides pipeline report generation, alerting (Slack / Email),
metrics logging, and Airflow task callbacks.
"""

from .alerting import AlertDispatcher, EmailAlerter, SlackAlerter
from .callbacks import (
    on_dag_success_callback,
    on_task_failure_callback,
    on_task_success_callback,
)
from .metrics import PipelineMetricsLogger
from .pipeline_report import PipelineReportGenerator

__all__ = [
    "PipelineReportGenerator",
    "SlackAlerter",
    "EmailAlerter",
    "AlertDispatcher",
    "PipelineMetricsLogger",
    "on_task_failure_callback",
    "on_task_success_callback",
    "on_dag_success_callback",
]
