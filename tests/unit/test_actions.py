"""Unit tests for ActionQueue class."""

from media_witch.core.actions import ActionQueue
from media_witch.core.fileops import FileOps


class TestActionQueue:
    """Tests for ActionQueue class."""

    def test_adds_actions(self, file_ops_real: FileOps) -> None:
        """Test that actions are added to queue."""
        queue = ActionQueue(file_ops_real)
        called = []

        queue.add(lambda: called.append(1), desc='Action 1')
        queue.add(lambda: called.append(2), desc='Action 2')

        assert len(called) == 0  # Not executed yet

    def test_commit_executes_actions(self, file_ops_real: FileOps) -> None:
        """Test that commit executes all queued actions."""
        queue = ActionQueue(file_ops_real)
        called = []

        queue.add(lambda: called.append(1), desc='Action 1')
        queue.add(lambda: called.append(2), desc='Action 2')
        queue.commit()

        assert called == [1, 2]

    def test_commit_with_args(self, file_ops_real: FileOps) -> None:
        """Test that actions with arguments are executed correctly."""
        queue = ActionQueue(file_ops_real)
        results = []

        def add_value(x: int, y: int) -> None:
            results.append(x + y)

        queue.add(add_value, 1, 2, desc='Add 1+2')
        queue.add(add_value, 3, 4, desc='Add 3+4')
        queue.commit()

        assert results == [3, 7]

    def test_dry_run_no_execution(self, file_ops_dry: FileOps) -> None:
        """Test that dry-run doesn't execute actions."""
        queue = ActionQueue(file_ops_dry)
        called = []

        queue.add(lambda: called.append(1), desc='Action 1')
        queue.commit()

        assert len(called) == 0

    def test_dry_run_logs_actions(self, file_ops_dry: FileOps) -> None:
        """Test that dry-run logs planned actions."""
        queue = ActionQueue(file_ops_dry)

        queue.add(lambda: None, desc='Test action 1')
        queue.add(lambda: None, desc='Test action 2')
        queue.commit()

        logs = file_ops_dry._test_logs
        assert any('[DRY-RUN]' in log for log in logs)
        assert any('Test action 1' in log for log in logs)
        assert any('Test action 2' in log for log in logs)

    def test_clear_removes_actions(self, file_ops_real: FileOps) -> None:
        """Test that clear removes all queued actions."""
        queue = ActionQueue(file_ops_real)
        called = []

        queue.add(lambda: called.append(1), desc='Action 1')
        queue.clear()
        queue.commit()

        assert len(called) == 0

    def test_empty_queue_commit(self, file_ops_dry: FileOps) -> None:
        """Test commit with empty queue."""
        queue = ActionQueue(file_ops_dry)
        queue.commit()  # Should not raise

        logs = file_ops_dry._test_logs
        assert any('No actions' in log for log in logs)
