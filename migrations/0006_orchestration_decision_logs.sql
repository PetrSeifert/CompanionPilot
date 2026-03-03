ALTER TABLE IF EXISTS planner_decision_logs
    RENAME TO orchestration_decision_logs;

ALTER TABLE IF EXISTS orchestration_decision_logs
    RENAME COLUMN planner TO stage;

ALTER INDEX IF EXISTS idx_planner_decision_logs_user_time
    RENAME TO idx_orchestration_decision_logs_user_time;

ALTER INDEX IF EXISTS idx_planner_decision_logs_planner_time
    RENAME TO idx_orchestration_decision_logs_stage_time;

ALTER INDEX IF EXISTS idx_planner_decision_logs_user_message_time
    RENAME TO idx_orchestration_decision_logs_user_message_time;
