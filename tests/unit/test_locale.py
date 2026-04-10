"""Unit tests for subtitle locale mapping."""

from pathlib import Path

import pytest

from media_witch.features.subtitles.locale import (LocaleMapper, Rule,
                                                   load_csv_rules,
                                                   parse_cli_rules)


class TestRule:
    """Tests for Rule dataclass."""

    def test_rule_creation(self) -> None:
        """Test Rule instantiation."""
        rule = Rule(source="chi", target="zh", case_sensitive=False)
        assert rule.source == "chi"
        assert rule.target == "zh"
        assert rule.case_sensitive is False


class TestLocaleMapper:
    """Tests for LocaleMapper class."""

    def test_case_insensitive_mapping(self) -> None:
        """Test case-insensitive locale mapping."""
        rules = [Rule(source="chi", target="zh", case_sensitive=False)]
        mapper = LocaleMapper(csv_rules=[], cli_rules=rules)

        assert mapper.resolve("chi") == "zh"
        assert mapper.resolve("CHI") == "zh"
        assert mapper.resolve("Chi") == "zh"

    def test_case_sensitive_mapping(self) -> None:
        """Test case-sensitive locale mapping."""
        rules = [Rule(source="EN", target="en-US", case_sensitive=True)]
        mapper = LocaleMapper(csv_rules=[], cli_rules=rules)

        assert mapper.resolve("EN") == "en-US"
        assert mapper.resolve("en") == "en"  # No match, returns original

    def test_cli_rules_precedence(self) -> None:
        """Test that CLI rules take precedence over CSV rules."""
        csv_rules = [Rule("chi", "zh-CN", False)]
        cli_rules = [Rule("chi", "zh-TW", False)]
        mapper = LocaleMapper(csv_rules=csv_rules, cli_rules=cli_rules)

        assert mapper.resolve("chi") == "zh-TW"

    def test_no_match_returns_original(self) -> None:
        """Test that unmatched tokens return original value."""
        mapper = LocaleMapper(csv_rules=[], cli_rules=[])
        assert mapper.resolve("unknown") == "unknown"

    def test_multiple_rules(self) -> None:
        """Test mapping with multiple rules."""
        rules = [
            Rule("chi", "zh", False),
            Rule("cht", "zh-Hant", False),
            Rule("eng", "en", False),
        ]
        mapper = LocaleMapper(csv_rules=[], cli_rules=rules)

        assert mapper.resolve("chi") == "zh"
        assert mapper.resolve("cht") == "zh-Hant"
        assert mapper.resolve("eng") == "en"


class TestParseCliRules:
    """Tests for parse_cli_rules function."""

    def test_parse_valid_rule(self) -> None:
        """Test parsing valid CLI rule."""
        rules = parse_cli_rules(["chi,zh,false"])
        assert len(rules) == 1
        assert rules[0].source == "chi"
        assert rules[0].target == "zh"
        assert rules[0].case_sensitive is False

    def test_parse_case_sensitive_variations(self) -> None:
        """Test parsing various case_sensitive values."""
        test_cases = [
            ("chi,zh,true", True),
            ("chi,zh,1", True),
            ("chi,zh,yes", True),
            ("chi,zh,y", True),
            ("chi,zh,false", False),
            ("chi,zh,0", False),
            ("chi,zh,no", False),
        ]
        for spec, expected in test_cases:
            rules = parse_cli_rules([spec])
            assert rules[0].case_sensitive == expected

    def test_parse_multiple_rules(self) -> None:
        """Test parsing multiple rules."""
        rules = parse_cli_rules([
            "chi,zh,false",
            "cht,zh-Hant,false",
            "eng,en,true",
        ])
        assert len(rules) == 3
        assert rules[0].source == "chi"
        assert rules[1].source == "cht"
        assert rules[2].source == "eng"

    def test_parse_with_spaces(self) -> None:
        """Test parsing rules with spaces."""
        rules = parse_cli_rules([" chi , zh , false "])
        assert rules[0].source == "chi"
        assert rules[0].target == "zh"

    def test_parse_invalid_format(self) -> None:
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Mapping rule must be"):
            parse_cli_rules(["invalid"])

        with pytest.raises(ValueError, match="Mapping rule must be"):
            parse_cli_rules(["chi,zh"])  # Missing case_sensitive


class TestLoadCsvRules:
    """Tests for load_csv_rules function."""

    def test_load_from_csv(self, tmp_path: Path) -> None:
        """Test loading rules from CSV file."""
        csv_file = tmp_path / "rules.csv"
        csv_file.write_text(
            "source,target,is_case_sensitive\n"
            "chi,zh,false\n"
            "cht,zh-Hant,false\n"
            "eng,en,true\n"
        )

        rules = load_csv_rules(csv_file)
        assert len(rules) == 3
        assert rules[0].source == "chi"
        assert rules[0].target == "zh"
        assert rules[0].case_sensitive is False

    def test_load_empty_csv(self, tmp_path: Path) -> None:
        """Test loading from empty CSV."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("source,target,is_case_sensitive\n")

        rules = load_csv_rules(csv_file)
        assert len(rules) == 0

    def test_load_csv_skips_empty_source(self, tmp_path: Path) -> None:
        """Test that rows with empty source are skipped."""
        csv_file = tmp_path / "rules.csv"
        csv_file.write_text(
            "source,target,is_case_sensitive\n"
            ",zh,false\n"
            "cht,zh-Hant,false\n"
        )

        rules = load_csv_rules(csv_file)
        assert len(rules) == 1
        assert rules[0].source == "cht"

    def test_load_nonexistent_file(self) -> None:
        """Test loading from non-existent file returns empty list."""
        rules = load_csv_rules(Path("/nonexistent/file.csv"))
        assert len(rules) == 0

    def test_load_none_path(self) -> None:
        """Test that None path returns empty list."""
        rules = load_csv_rules(None)
        assert len(rules) == 0

    def test_load_csv_case_insensitive_headers(self, tmp_path: Path) -> None:
        """Test that CSV headers are case-insensitive."""
        csv_file = tmp_path / "rules.csv"
        csv_file.write_text(
            "SOURCE,TARGET,IS_CASE_SENSITIVE\n"
            "chi,zh,false\n"
        )

        rules = load_csv_rules(csv_file)
        assert len(rules) == 1
        assert rules[0].source == "chi"
