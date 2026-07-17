"""Subtitle locale mapping utilities."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Rule:
    """A locale mapping rule.

    Attributes:
        source: Source locale code to match
        target: Target locale code to map to
        case_sensitive: Whether matching should be case-sensitive
    """

    source: str
    target: str
    case_sensitive: bool


class LocaleMapper:
    """Maps subtitle locale codes using configurable rules.

    Rules can be loaded from CSV files or provided directly via CLI.
    CLI rules take precedence over CSV rules.
    """

    def __init__(self, csv_rules: list[Rule], cli_rules: list[Rule]) -> None:
        """Initialize LocaleMapper.

        Args:
            csv_rules: Rules loaded from CSV file
            cli_rules: Rules from command-line arguments
        """
        self.csv_rules = csv_rules
        self.cli_rules = cli_rules

    @staticmethod
    def _match(r: Rule, t: str) -> bool:
        """Check if a rule matches a token.

        Args:
            r: Rule to check
            t: Token to match against

        Returns:
            True if rule matches token
        """
        return (t == r.source) if r.case_sensitive else (t.lower() == r.source.lower())

    def resolve(self, token: str) -> str:
        """Resolve a locale token using mapping rules.

        CLI rules are checked first, then CSV rules.
        If no rule matches, returns the original token.

        Args:
            token: Locale code to resolve

        Returns:
            Mapped locale code, or original if no rule matches
        """
        for src in (self.cli_rules, self.csv_rules):
            for r in src:
                if self._match(r, token):
                    return r.target
        return token

    def get_target_locales(self) -> set[str]:
        """Get all target locales from mapping rules.

        Returns:
            Set of target locale codes
        """
        targets = set()
        for src in (self.cli_rules, self.csv_rules):
            for r in src:
                targets.add(r.target)
        return targets


def parse_cli_rules(cli_rules: list[str]) -> list[Rule]:
    """Parse locale mapping rules from CLI arguments.

    Args:
        cli_rules: List of rule specifications in format "source,target,case_sensitive"

    Returns:
        List of parsed Rule objects

    Raises:
        ValueError: If rule format is invalid
    """
    rules = []
    for spec in cli_rules:
        parts = [x.strip() for x in spec.split(',')]
        if len(parts) != 3:
            raise ValueError('Mapping rule must be: source,target,case_sensitive')
        rules.append(Rule(parts[0], parts[1], parts[2].lower() in ('1', 'true', 'yes', 'y')))
    return rules


def load_csv_rules(csv_path: Path | None) -> list[Rule]:
    """Load locale mapping rules from CSV file.

    Expected CSV columns: source, target, is_case_sensitive

    Args:
        csv_path: Path to CSV file, or None to return empty list

    Returns:
        List of parsed Rule objects

    Raises:
        FileNotFoundError: If csv_path exists but file is not found
    """
    if not csv_path or not csv_path.exists():
        return []
    rules: list[Rule] = []
    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return rules
        field = {k.lower(): k for k in reader.fieldnames}
        for row in reader:
            src = row[field.get('source', 'source')].strip()
            if not src:
                continue
            tgt = row[field.get('target', 'target')].strip()
            cs = row[field.get('is_case_sensitive', 'is_case_sensitive')].strip()
            rules.append(Rule(src, tgt, cs.lower() in ('1', 'true', 'yes', 'y')))
    return rules
