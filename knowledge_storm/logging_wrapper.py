from contextlib import contextmanager
import time
import pytz
from datetime import datetime

# Timestamps are stored in UTC and converted to Pacific time for display.
PACIFIC_TZ = pytz.timezone("America/Los_Angeles")


class EventLog:
    """Records the start/end timestamps of a named pipeline event."""

    def __init__(self, event_name):
        self.event_name = event_name
        self.start_time = None
        self.end_time = None
        self.child_events = {}

    def record_start_time(self):
        self.start_time = datetime.now(pytz.utc)

    def record_end_time(self):
        self.end_time = datetime.now(pytz.utc)

    def get_total_time(self):
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0

    def _fmt(self, dt):
        """Format a UTC datetime as a Pacific-time string with milliseconds."""
        return dt.astimezone(PACIFIC_TZ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def get_start_time(self):
        return self._fmt(self.start_time) if self.start_time else None

    def get_end_time(self):
        return self._fmt(self.end_time) if self.end_time else None

    def add_child_event(self, child_event):
        self.child_events[child_event.event_name] = child_event

    def get_child_events(self):
        return self.child_events


class LoggingWrapper:
    """
    Structured logger for the multi-stage pipeline.

    Tracks wall-clock time, LM usage, and query counts for each named pipeline
    stage. Stages and events are entered/exited via context managers.
    """

    def __init__(self, lm_config):
        self.logging_dict = {}
        self.lm_config = lm_config
        self.current_pipeline_stage = None
        self.event_stack = []
        self.pipeline_stage_active = False

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _pipeline_stage_start(self, pipeline_stage: str):
        if self.pipeline_stage_active:
            raise RuntimeError(
                "A pipeline stage is already active. "
                "End the current stage before starting a new one."
            )
        self.current_pipeline_stage = pipeline_stage
        self.logging_dict[pipeline_stage] = {
            "time_usage": {},
            "lm_usage": {},
            "lm_history": [],
            "query_count": 0,
        }
        self.pipeline_stage_active = True

    def _event_start(self, event_name: str):
        if not self.pipeline_stage_active:
            raise RuntimeError("No pipeline stage is currently active.")

        stage_log = self.logging_dict[self.current_pipeline_stage]

        if not self.event_stack:
            # Top-level event directly under the pipeline stage.
            if event_name not in stage_log["time_usage"]:
                event = EventLog(event_name=event_name)
                event.record_start_time()
                stage_log["time_usage"][event_name] = event
                self.event_stack.append(event)
            else:
                stage_log["time_usage"][event_name].record_start_time()
        else:
            # Nested event under the current parent.
            parent_event = self.event_stack[-1]
            if event_name not in parent_event.get_child_events():
                event = EventLog(event_name=event_name)
                event.record_start_time()
                parent_event.add_child_event(event)
                stage_log["time_usage"][event_name] = event
                self.event_stack.append(event)
            else:
                parent_event.get_child_events()[event_name].record_start_time()

    def _event_end(self, event_name: str):
        if not self.pipeline_stage_active:
            raise RuntimeError("No pipeline stage is currently active.")
        if not self.event_stack:
            raise RuntimeError("No parent event is currently active.")

        stage_log = self.logging_dict[self.current_pipeline_stage]
        current_event = self.event_stack[-1]

        if event_name in current_event.get_child_events():
            current_event.get_child_events()[event_name].record_end_time()
        elif event_name in stage_log["time_usage"]:
            stage_log["time_usage"][event_name].record_end_time()
        else:
            raise AssertionError(
                f"Cannot record end time for '{event_name}': start time was never recorded."
            )

        if current_event.event_name == event_name:
            self.event_stack.pop()

    def _pipeline_stage_end(self):
        if not self.pipeline_stage_active:
            raise RuntimeError("No pipeline stage is currently active to end.")

        stage_log = self.logging_dict[self.current_pipeline_stage]
        stage_log["lm_usage"] = self.lm_config.collect_and_reset_lm_usage()
        stage_log["lm_history"] = self.lm_config.collect_and_reset_lm_history()
        self.pipeline_stage_active = False

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def add_query_count(self, count):
        if not self.pipeline_stage_active:
            raise RuntimeError("No active pipeline stage to add query count to.")
        self.logging_dict[self.current_pipeline_stage]["query_count"] += count

    @contextmanager
    def log_event(self, event_name):
        """Context manager that wraps a named event within the current stage."""
        if not self.pipeline_stage_active:
            raise RuntimeError("No pipeline stage is currently active.")
        self._event_start(event_name)
        yield
        self._event_end(event_name)

    @contextmanager
    def log_pipeline_stage(self, pipeline_stage):
        """Context manager that wraps an entire named pipeline stage."""
        if self.pipeline_stage_active:
            print("Active stage detected; closing it before starting a new one.")
            self._pipeline_stage_end()

        t_wall = time.time()
        try:
            self._pipeline_stage_start(pipeline_stage)
            yield
        except Exception as exc:
            print(f"Error in pipeline stage '{pipeline_stage}': {exc}")
        finally:
            self.logging_dict[self.current_pipeline_stage]["total_wall_time"] = (
                time.time() - t_wall
            )
            self._pipeline_stage_end()

    def dump_logging_and_reset(self, reset_logging=True):
        """Serialise and optionally clear the accumulated log data."""
        log_dump = {}
        for stage, stage_log in self.logging_dict.items():
            time_entries = {
                event_name: {
                    "total_time_seconds": event.get_total_time(),
                    "start_time": event.get_start_time(),
                    "end_time": event.get_end_time(),
                }
                for event_name, event in stage_log["time_usage"].items()
            }
            log_dump[stage] = {
                "time_usage": time_entries,
                "lm_usage": stage_log["lm_usage"],
                "lm_history": stage_log["lm_history"],
                "query_count": stage_log["query_count"],
                "total_wall_time": stage_log["total_wall_time"],
            }
        if reset_logging:
            self.logging_dict.clear()
        return log_dump
