"""Unit tests for FileOps class."""

from pathlib import Path

from media_witch.core.fileops import FileOps


class TestFileOpsEnsureDir:
    """Tests for ensure_dir method."""

    def test_creates_directory(self, tmp_path: Path, file_ops_real: FileOps) -> None:
        """Test that directories are created."""
        new_dir = tmp_path / 'test_dir'
        file_ops_real.ensure_dir(new_dir)
        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_creates_nested_directories(self, tmp_path: Path, file_ops_real: FileOps) -> None:
        """Test that nested directories are created."""
        nested_dir = tmp_path / 'a' / 'b' / 'c'
        file_ops_real.ensure_dir(nested_dir)
        assert nested_dir.exists()
        assert nested_dir.is_dir()

    def test_dry_run_no_creation(self, tmp_path: Path, file_ops_dry: FileOps) -> None:
        """Test that dry-run doesn't create directories."""
        new_dir = tmp_path / 'test_dir'
        file_ops_dry.ensure_dir(new_dir)
        assert not new_dir.exists()

    def test_idempotent(self, tmp_path: Path, file_ops_real: FileOps) -> None:
        """Test that calling ensure_dir multiple times is safe."""
        new_dir = tmp_path / 'test_dir'
        file_ops_real.ensure_dir(new_dir)
        file_ops_real.ensure_dir(new_dir)
        assert new_dir.exists()


class TestFileOpsMoveFile:
    """Tests for move_file method."""

    def test_moves_file(self, tmp_path: Path, file_ops_real: FileOps) -> None:
        """Test that files are moved."""
        src = tmp_path / 'source.txt'
        dst = tmp_path / 'dest.txt'
        src.write_text('content')

        file_ops_real.move_file(src, dst)

        assert not src.exists()
        assert dst.exists()
        assert dst.read_text() == 'content'

    def test_creates_parent_dir(self, tmp_path: Path, file_ops_real: FileOps) -> None:
        """Test that parent directories are created."""
        src = tmp_path / 'source.txt'
        dst = tmp_path / 'subdir' / 'dest.txt'
        src.write_text('content')

        file_ops_real.move_file(src, dst)

        assert dst.exists()
        assert dst.parent.exists()

    def test_dry_run_no_move(self, tmp_path: Path, file_ops_dry: FileOps) -> None:
        """Test that dry-run doesn't move files."""
        src = tmp_path / 'source.txt'
        dst = tmp_path / 'dest.txt'
        src.write_text('content')

        file_ops_dry.move_file(src, dst)

        assert src.exists()
        assert not dst.exists()

    def test_skip_if_dest_exists(self, tmp_path: Path, file_ops_real: FileOps) -> None:
        """Test that existing destinations are not overwritten."""
        src = tmp_path / 'source.txt'
        dst = tmp_path / 'dest.txt'
        src.write_text('source content')
        dst.write_text('dest content')

        file_ops_real.move_file(src, dst)

        assert src.exists()
        assert dst.read_text() == 'dest content'

    def test_skip_if_same_path(self, tmp_path: Path, file_ops_real: FileOps) -> None:
        """Test that moving to same path is skipped."""
        src = tmp_path / 'file.txt'
        src.write_text('content')

        file_ops_real.move_file(src, src)

        assert src.exists()


class TestFileOpsRenameFile:
    """Tests for rename_file method."""

    def test_renames_file(self, tmp_path: Path, file_ops_real: FileOps) -> None:
        """Test that files are renamed."""
        src = tmp_path / 'old_name.txt'
        dst = tmp_path / 'new_name.txt'
        src.write_text('content')

        file_ops_real.rename_file(src, dst)

        assert not src.exists()
        assert dst.exists()
        assert dst.read_text() == 'content'

    def test_dry_run_no_rename(self, tmp_path: Path, file_ops_dry: FileOps) -> None:
        """Test that dry-run doesn't rename files."""
        src = tmp_path / 'old_name.txt'
        dst = tmp_path / 'new_name.txt'
        src.write_text('content')

        file_ops_dry.rename_file(src, dst)

        assert src.exists()
        assert not dst.exists()


class TestFileOpsWriteTextIfAbsent:
    """Tests for write_text_if_absent method."""

    def test_writes_new_file(self, tmp_path: Path, file_ops_real: FileOps) -> None:
        """Test that new files are written."""
        file = tmp_path / 'new_file.txt'

        file_ops_real.write_text_if_absent(file, 'content')

        assert file.exists()
        assert file.read_text() == 'content'

    def test_skips_existing_file(self, tmp_path: Path, file_ops_real: FileOps) -> None:
        """Test that existing files are not overwritten."""
        file = tmp_path / 'existing.txt'
        file.write_text('original')

        file_ops_real.write_text_if_absent(file, 'new content')

        assert file.read_text() == 'original'

    def test_dry_run_no_write(self, tmp_path: Path, file_ops_dry: FileOps) -> None:
        """Test that dry-run doesn't write files."""
        file = tmp_path / 'new_file.txt'

        file_ops_dry.write_text_if_absent(file, 'content')

        assert not file.exists()


class TestFileOpsRemoveDirIfEmpty:
    """Tests for remove_dir_if_empty method."""

    def test_removes_empty_dir(self, tmp_path: Path, file_ops_real: FileOps) -> None:
        """Test that empty directories are removed."""
        empty_dir = tmp_path / 'empty'
        empty_dir.mkdir()

        file_ops_real.remove_dir_if_empty(empty_dir)

        assert not empty_dir.exists()

    def test_keeps_non_empty_dir(self, tmp_path: Path, file_ops_real: FileOps) -> None:
        """Test that non-empty directories are kept."""
        dir_with_file = tmp_path / 'not_empty'
        dir_with_file.mkdir()
        (dir_with_file / 'file.txt').touch()

        file_ops_real.remove_dir_if_empty(dir_with_file)

        assert dir_with_file.exists()

    def test_dry_run_no_remove(self, tmp_path: Path, file_ops_dry: FileOps) -> None:
        """Test that dry-run doesn't remove directories."""
        empty_dir = tmp_path / 'empty'
        empty_dir.mkdir()

        file_ops_dry.remove_dir_if_empty(empty_dir)

        assert empty_dir.exists()
