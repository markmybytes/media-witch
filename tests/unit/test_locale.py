"""Property-based tests for subtitle locale mapping."""

import csv
from pathlib import Path

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.strategies import composite

from media_witch.features.subtitles.locale import (LocaleMapper, Rule,
                                                   load_csv_rules,
                                                   parse_cli_rules)


# Custom strategies
@composite
def locale_code(draw, min_size: int = 1, max_size: int = 10):
    """Generate realistic locale codes (ASCII only to avoid encoding issues)."""
    return draw(
        st.text(
            alphabet=st.characters(
                min_codepoint=0x30,  # '0'
                max_codepoint=0x7A,  # 'z'
                whitelist_categories=("Ll", "Lu", "Nd"),
                blacklist_characters=",\t\n\r"
            ),
            min_size=min_size,
            max_size=max_size,
        )
    )


@composite
def mapping_rule(draw):
    """Generate a valid mapping rule."""
    source = draw(locale_code())
    target = draw(locale_code())
    case_sensitive = draw(st.booleans())
    return Rule(source=source, target=target, case_sensitive=case_sensitive)


@composite
def csv_content(draw, valid: bool = True):
    """Generate CSV content for testing (ASCII only to avoid encoding issues)."""
    if valid:
        num_rules = draw(st.integers(min_value=0, max_value=20))
        lines = ["source,target,is_case_sensitive"]

        for _ in range(num_rules):
            source = draw(locale_code())
            target = draw(locale_code())
            case_sensitive = draw(st.sampled_from(
                ["true", "false", "1", "0", "yes", "no", "y", "n"]))
            lines.append(f"{source},{target},{case_sensitive}")

        return "\n".join(lines)
    else:
        return draw(st.text(alphabet=st.characters(max_codepoint=0x7F), max_size=500))


# Tests for LocaleMapper
class TestLocaleMapperProperties:
    """Property-based tests for LocaleMapper."""

    @given(locale_code())
    def test_identity_mapping(self, locale: str) -> None:
        """Unmapped locales should return themselves (identity)."""
        mapper = LocaleMapper(csv_rules=[], cli_rules=[])
        assert mapper.resolve(locale) == locale

    @given(locale_code(), locale_code())
    def test_case_insensitive_mapping_property(self, source: str, target: str) -> None:
        """Case-insensitive mapping should work for all case variations."""
        assume(len(source) > 0 and len(target) > 0)
        assume(source.lower().upper() == source.upper())

        rule = Rule(source=source.lower(), target=target, case_sensitive=False)
        mapper = LocaleMapper(csv_rules=[], cli_rules=[rule])

        assert mapper.resolve(source.lower()) == target
        assert mapper.resolve(source.upper()) == target
        if source.isalpha():
            assert mapper.resolve(source.title()) == target

    @given(locale_code(), locale_code())
    def test_case_sensitive_mapping_exact_match_only(self, source: str, target: str) -> None:
        """Case-sensitive mapping should only match exact case."""
        assume(len(source) > 0 and len(target) > 0)
        assume(source.lower() != source.upper())

        rule = Rule(source=source, target=target, case_sensitive=True)
        mapper = LocaleMapper(csv_rules=[], cli_rules=[rule])

        assert mapper.resolve(source) == target

        if source != source.upper():
            assert mapper.resolve(source.upper()) == source.upper()
        if source != source.lower():
            assert mapper.resolve(source.lower()) == source.lower()

    @given(st.lists(mapping_rule(), min_size=1, max_size=20))
    def test_cli_rules_checked_before_csv(self, rules: list[Rule]) -> None:
        """CLI rules should be checked before CSV rules."""
        csv_rules = rules[:-1] if len(rules) > 1 else []
        cli_rules = [rules[-1]]

        mapper = LocaleMapper(csv_rules=csv_rules, cli_rules=cli_rules)

        cli_rule = cli_rules[0]
        result = mapper.resolve(cli_rule.source)
        assert result == cli_rule.target

    @given(locale_code(), locale_code(), locale_code())
    def test_cli_overrides_csv_for_same_source(
        self, source: str, csv_target: str, cli_target: str
    ) -> None:
        """When both CSV and CLI have rules for same source, CLI wins."""
        assume(len(source) > 0)
        assume(csv_target != cli_target)

        csv_rules = [Rule(source=source, target=csv_target,
                          case_sensitive=False)]
        cli_rules = [Rule(source=source, target=cli_target,
                          case_sensitive=False)]

        mapper = LocaleMapper(csv_rules=csv_rules, cli_rules=cli_rules)
        assert mapper.resolve(source) == cli_target

    @given(st.lists(mapping_rule(), min_size=1, max_size=20))
    def test_first_matching_rule_wins(self, rules: list[Rule]) -> None:
        """First matching rule in a list should be used."""
        assume(len(rules) > 0)

        mapper = LocaleMapper(csv_rules=[], cli_rules=rules)

        first_rule = rules[0]
        result = mapper.resolve(first_rule.source)
        assert result == first_rule.target

    @given(st.lists(mapping_rule(), min_size=0, max_size=20))
    def test_get_target_locales_returns_all_targets(self, rules: list[Rule]) -> None:
        """get_target_locales should return all unique targets."""
        mapper = LocaleMapper(csv_rules=rules, cli_rules=[])

        targets = mapper.get_target_locales()
        expected_targets = {r.target for r in rules}

        assert targets == expected_targets

    @given(st.lists(mapping_rule(), min_size=0, max_size=20))
    def test_resolve_is_deterministic(self, rules: list[Rule]) -> None:
        """Resolving the same locale multiple times should give same result."""
        mapper = LocaleMapper(csv_rules=rules, cli_rules=[])

        for rule in rules:
            result1 = mapper.resolve(rule.source)
            result2 = mapper.resolve(rule.source)
            assert result1 == result2


# Tests for parse_cli_rules
class TestParseCliRulesProperties:
    """Property-based tests for CLI rule parsing."""

    @given(locale_code(), locale_code(), st.booleans())
    def test_parse_roundtrip(self, source: str, target: str, case_sensitive: bool) -> None:
        """Parsing should correctly extract all rule components."""
        assume(len(source) > 0 and len(target) > 0)
        assume("," not in source and "," not in target)

        case_str = "true" if case_sensitive else "false"
        rule_spec = f"{source},{target},{case_str}"

        rules = parse_cli_rules([rule_spec])

        assert len(rules) == 1
        assert rules[0].source == source
        assert rules[0].target == target
        assert rules[0].case_sensitive == case_sensitive

    @given(
        locale_code(),
        locale_code(),
        st.sampled_from(["true", "1", "yes", "y", "TRUE", "Yes", "Y"])
    )
    def test_parse_truthy_values(self, source: str, target: str, truthy: str) -> None:
        """Various truthy values should parse to case_sensitive=True."""
        assume(len(source) > 0 and len(target) > 0)
        assume("," not in source and "," not in target)

        rule_spec = f"{source},{target},{truthy}"
        rules = parse_cli_rules([rule_spec])

        assert rules[0].case_sensitive is True

    @given(
        locale_code(),
        locale_code(),
        st.sampled_from(
            ["false", "0", "no", "n", "FALSE", "No", "anything_else"])
    )
    def test_parse_falsy_values(self, source: str, target: str, falsy: str) -> None:
        """Non-truthy values should parse to case_sensitive=False."""
        assume(len(source) > 0 and len(target) > 0)
        assume("," not in source and "," not in target)

        rule_spec = f"{source},{target},{falsy}"
        rules = parse_cli_rules([rule_spec])

        assert rules[0].case_sensitive is False

    @given(st.lists(
        st.tuples(locale_code(), locale_code(), st.booleans()),
        min_size=1,
        max_size=20
    ))
    def test_parse_multiple_rules(self, rule_data: list[tuple[str, str, bool]]) -> None:
        """Parsing multiple rules should preserve all of them."""
        rule_data = [
            (s, t, c) for s, t, c in rule_data
            if "," not in s and "," not in t and len(s) > 0 and len(t) > 0
        ]
        assume(len(rule_data) > 0)

        specs = [
            f"{s},{t},{'true' if c else 'false'}"
            for s, t, c in rule_data
        ]

        rules = parse_cli_rules(specs)

        assert len(rules) == len(rule_data)
        for rule, (expected_s, expected_t, expected_c) in zip(rules, rule_data):
            assert rule.source == expected_s
            assert rule.target == expected_t
            assert rule.case_sensitive == expected_c

    @given(
        st.text(max_size=50).filter(lambda x: x.count(",") != 2)
    )
    def test_parse_invalid_format_raises_error(self, invalid_spec: str) -> None:
        """Invalid format should raise ValueError."""
        assume("," in invalid_spec or len(invalid_spec) > 0)

        with pytest.raises(ValueError, match="Mapping rule must be"):
            parse_cli_rules([invalid_spec])

    @given(locale_code(), locale_code(), st.booleans())
    def test_parse_strips_whitespace(self, source: str, target: str, case_sensitive: bool) -> None:
        """Parsing should strip whitespace from components."""
        assume(len(source) > 0 and len(target) > 0)
        assume("," not in source and "," not in target)

        case_str = "true" if case_sensitive else "false"
        rule_spec = f"  {source}  ,  {target}  ,  {case_str}  "

        rules = parse_cli_rules([rule_spec])

        assert rules[0].source == source
        assert rules[0].target == target


# Tests for load_csv_rules
class TestLoadCsvRulesProperties:
    """Property-based tests for CSV rule loading."""

    @given(st.lists(
        st.tuples(locale_code(), locale_code(), st.booleans()),
        min_size=0,
        max_size=20
    ))
    def test_csv_roundtrip(self, rule_data: list[tuple[str, str, bool]]) -> None:
        """Rules should roundtrip through CSV correctly."""
        import tempfile

        rule_data = [
            (s, t, c) for s, t, c in rule_data
            if len(s) > 0 and len(t) > 0 and "," not in s and "," not in t
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            csv_file = tmp_path / "rules.csv"
            lines = ["source,target,is_case_sensitive"]
            for source, target, case_sensitive in rule_data:
                case_str = "true" if case_sensitive else "false"
                lines.append(f"{source},{target},{case_str}")
            csv_file.write_text("\n".join(lines), encoding='utf-8')

            rules = load_csv_rules(csv_file)

            assert len(rules) == len(rule_data)
            for rule, (expected_s, expected_t, expected_c) in zip(rules, rule_data):
                assert rule.source == expected_s
                assert rule.target == expected_t
                assert rule.case_sensitive == expected_c

    @given(csv_content(valid=False))
    def test_csv_parsing_never_crashes(self, content: str) -> None:
        """CSV parsing should handle malformed input gracefully."""
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            try:
                f.write(content)
                f.flush()
                csv_file = Path(f.name)

                try:
                    rules = load_csv_rules(csv_file)
                    assert isinstance(rules, list)
                except Exception as e:
                    assert isinstance(
                        e, (UnicodeDecodeError, csv.Error, KeyError))
            finally:
                try:
                    Path(f.name).unlink()
                except:
                    pass

    def test_csv_nonexistent_file_returns_empty(self) -> None:
        """Loading from non-existent file should return empty list."""
        rules = load_csv_rules(Path("/nonexistent/file.csv"))
        assert rules == []

    def test_csv_none_path_returns_empty(self) -> None:
        """Loading from None path should return empty list."""
        rules = load_csv_rules(None)
        assert rules == []

    @given(st.lists(
        st.tuples(locale_code(), locale_code()),
        min_size=1,
        max_size=10
    ))
    def test_csv_skips_empty_source(self, rule_data: list[tuple[str, str]]) -> None:
        """CSV loading should skip rows with empty source."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            csv_file = tmp_path / "rules.csv"
            lines = ["source,target,is_case_sensitive"]
            lines.append(",target,false")

            for source, target in rule_data:
                if len(source) > 0 and len(target) > 0 and "," not in source and "," not in target:
                    lines.append(f"{source},{target},false")

            csv_file.write_text("\n".join(lines), encoding='utf-8')

            rules = load_csv_rules(csv_file)

            assert all(r.source != "" for r in rules)

    @given(st.sampled_from([
        "source,target,is_case_sensitive",
        "SOURCE,TARGET,IS_CASE_SENSITIVE",
        "Source,Target,Is_Case_Sensitive",
    ]))
    def test_csv_headers_case_insensitive(self, header: str) -> None:
        """CSV headers should be case-insensitive."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            csv_file = tmp_path / "rules.csv"
            csv_file.write_text(
                f"{header}\n"
                "chi,zh,false\n",
                encoding='utf-8'
            )

            rules = load_csv_rules(csv_file)

            assert len(rules) == 1
            assert rules[0].source == "chi"


# Integration tests
class TestLocaleMapperIntegration:
    """Integration tests combining multiple components."""

    @given(st.lists(
        st.tuples(locale_code(), locale_code(), st.booleans()),
        min_size=1,
        max_size=10,
        unique_by=lambda x: x[0].lower()
    ))
    def test_csv_and_cli_together(self, rule_data: list[tuple[str, str, bool]]) -> None:
        """CSV and CLI rules should work together correctly."""
        import tempfile

        rule_data = [
            (s, t, c) for s, t, c in rule_data
            if len(s) > 0 and len(t) > 0 and "," not in s and "," not in t
        ]
        assume(len(rule_data) >= 2)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            csv_data = rule_data[:-1]
            cli_data = [rule_data[-1]]

            csv_file = tmp_path / "rules.csv"
            lines = ["source,target,is_case_sensitive"]
            for source, target, case_sensitive in csv_data:
                lines.append(
                    f"{source},{target},{'true' if case_sensitive else 'false'}")
            csv_file.write_text("\n".join(lines), encoding='utf-8')

            cli_specs = [
                f"{s},{t},{'true' if c else 'false'}"
                for s, t, c in cli_data
            ]

            csv_rules = load_csv_rules(csv_file)
            cli_rules = parse_cli_rules(cli_specs)
            mapper = LocaleMapper(csv_rules=csv_rules, cli_rules=cli_rules)

            cli_source, cli_target, _ = cli_data[0]
            assert mapper.resolve(cli_source) == cli_target

            for csv_source, csv_target, _ in csv_data:
                if csv_source.lower() != cli_source.lower():
                    result = mapper.resolve(csv_source)
                    assert result == csv_target or result == csv_source


# Invariant tests
class TestLocaleMapperInvariants:
    """Tests for invariants that should always hold."""

    @given(st.lists(mapping_rule(), max_size=20), locale_code())
    def test_resolve_always_returns_string(self, rules: list[Rule], token: str) -> None:
        """resolve() should always return a string."""
        mapper = LocaleMapper(csv_rules=rules, cli_rules=[])
        result = mapper.resolve(token)
        assert isinstance(result, str)

    @given(st.lists(mapping_rule(), max_size=20), locale_code())
    def test_resolve_never_returns_empty(self, rules: list[Rule], token: str) -> None:
        """resolve() should never return empty string if input is non-empty."""
        assume(len(token) > 0)

        mapper = LocaleMapper(csv_rules=rules, cli_rules=[])
        result = mapper.resolve(token)

        assert len(result) > 0

    @given(st.lists(mapping_rule(), max_size=20))
    def test_get_target_locales_returns_set(self, rules: list[Rule]) -> None:
        """get_target_locales() should always return a set."""
        mapper = LocaleMapper(csv_rules=rules, cli_rules=[])
        targets = mapper.get_target_locales()
        assert isinstance(targets, set)

    @given(st.lists(mapping_rule(), max_size=20), locale_code())
    def test_unmapped_locale_is_identity(self, rules: list[Rule], token: str) -> None:
        """If no rule matches, resolve() should return the input unchanged."""
        rules_with_different_sources = [
            Rule(source=r.source + "_different", target=r.target,
                 case_sensitive=r.case_sensitive)
            for r in rules
        ]

        assume(not any(r.source == token for r in rules_with_different_sources))

        mapper = LocaleMapper(
            csv_rules=rules_with_different_sources, cli_rules=[])
        result = mapper.resolve(token)

        assert result == token
