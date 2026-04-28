import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from core.connection import ConnectionHandler


class ExperimentPrewarmRouteContextTest(unittest.TestCase):
    def _make_handler(self) -> ConnectionHandler:
        handler = object.__new__(ConnectionHandler)
        handler.experiment_prewarm_session_adopted = False
        handler.experiment_prewarm_status = "idle"
        handler.experiment_prewarm_ready_level = "none"
        handler.experiment_prewarm_trigger = ""
        handler.experiment_prewarm_error = ""
        handler.experiment_deep_prefetch_status = "idle"
        handler.experiment_deep_prefetch_error = ""
        handler.experiment_deep_prefetch_focus = ""
        handler.experiment_deep_prefetch_query = ""
        handler.experiment_yaml_path = ""
        handler.experiment_session_id = ""
        handler.experiment_current_step_id = ""
        handler.experiment_overview = None
        handler.experiment_current_step = None
        handler.experiment_list_steps = None
        handler.experiment_schema = None
        handler.experiment_reference = None
        handler.experiment_resume_recovery_required = False
        handler.experiment_resume_recovery_source = ""
        handler.experiment_resume_previous_session_id = ""
        handler.experiment_resume_reason = ""
        handler.experiment_resume_log_path = ""
        handler.experiment_resume_turn_count = ""
        handler.experiment_resume_latest_session_id = ""
        handler.experiment_resume_latest_current_step_id = ""
        handler.experiment_resume_context_excerpt = ""
        return handler

    def test_incomplete_prewarm_without_wait_result_does_not_expose_context(self):
        handler = self._make_handler()
        handler.experiment_prewarm_status = "minimal_warming"
        handler.experiment_session_id = "exp-1"
        handler.experiment_current_step_id = "step_prepare"
        handler.experiment_yaml_path = "C:/demo/experiments.yaml"

        context = handler._experiment_prewarm_route_context()

        self.assertEqual({}, context)

    def test_minimal_ready_context_persists_for_later_turns_after_timeout(self):
        handler = self._make_handler()
        handler.experiment_prewarm_status = "completed"
        handler.experiment_prewarm_ready_level = "completed"
        handler.experiment_prewarm_trigger = "hello"
        handler.experiment_yaml_path = "C:/demo/experiments.yaml"
        handler.experiment_session_id = "exp-2"
        handler.experiment_current_step_id = "step_prepare_setup_all"
        handler.experiment_overview = {
            "title": "Silver nanoparticle synthesis",
            "current": "prepare setup",
        }
        handler.experiment_current_step = {
            "step_id": "step_prepare_setup_all",
            "title": "Prepare all vessels and stir bars",
        }

        context = handler._experiment_prewarm_route_context()

        self.assertEqual("completed", context["experiment_prewarm_status"])
        self.assertEqual("completed", context["experiment_prewarm_ready_level"])
        self.assertEqual("hello", context["experiment_prewarm_trigger"])
        self.assertEqual("C:/demo/experiments.yaml", context["experiment_yaml_path"])
        self.assertEqual("exp-2", context["experiment_session_id"])
        self.assertEqual(
            "step_prepare_setup_all", context["experiment_current_step_id"]
        )
        self.assertIn(
            "Silver nanoparticle synthesis",
            context["experiment_overview_summary"],
        )
        self.assertIn(
            "Prepare all vessels and stir bars",
            context["experiment_current_step_summary"],
        )

    def test_ready_context_includes_deep_prefetch_summaries(self):
        handler = self._make_handler()
        handler.experiment_prewarm_status = "completed"
        handler.experiment_prewarm_ready_level = "completed"
        handler.experiment_yaml_path = "C:/demo/experiments.yaml"
        handler.experiment_session_id = "exp-3"
        handler.experiment_current_step_id = "step_prepare_setup_all"
        handler.experiment_deep_prefetch_status = "ready"
        handler.experiment_deep_prefetch_focus = "workflow,schema"
        handler.experiment_deep_prefetch_query = (
            "\u628a\u540e\u7eed\u6b65\u9aa4\u548c\u5b57\u6bb5\u5b9a\u4e49\u8bf4\u4e00\u4e0b"
        )
        handler.experiment_list_steps = {"tool": "list_steps", "count": 12}
        handler.experiment_schema = {"tool": "get_schema", "fields": ["temperature"]}

        context = handler._experiment_prewarm_route_context(deep_wait_result="ready")

        self.assertEqual("ready", context["experiment_deep_prefetch_wait_result"])
        self.assertEqual("ready", context["experiment_deep_prefetch_status"])
        self.assertEqual("workflow,schema", context["experiment_deep_prefetch_focus"])
        self.assertIn("list_steps", context["experiment_list_steps_summary"])
        self.assertIn("get_schema", context["experiment_schema_summary"])

    def test_completed_background_warming_exposes_common_context_summaries(self):
        handler = self._make_handler()
        handler.experiment_prewarm_status = "completed"
        handler.experiment_prewarm_ready_level = "completed"
        handler.experiment_yaml_path = "C:/demo/experiments.yaml"
        handler.experiment_session_id = "exp-4"
        handler.experiment_current_step_id = "step_prepare_setup_all"
        handler.experiment_list_steps = {"tool": "list_steps", "count": 12}
        handler.experiment_schema = {"tool": "get_schema", "fields": ["temperature"]}

        context = handler._experiment_prewarm_route_context()

        self.assertIn("list_steps", context["experiment_list_steps_summary"])
        self.assertIn("get_schema", context["experiment_schema_summary"])


if __name__ == "__main__":
    unittest.main()
