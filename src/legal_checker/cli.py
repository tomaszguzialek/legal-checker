"""Command-line interface: check a document fragment and print the result (SPEC Task 5.1).

Prints the ``ComplianceResult`` as JSON, then a report for a human: the assessment, the
provisions the agent saw (``consulted_units``) and the tools it called.
"""

import argparse
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from legal_checker.agent.graph import run
from legal_checker.agent.nodes import EvaluationError
from legal_checker.agent.state import GraphState
from legal_checker.config import get_settings
from legal_checker.eli_client import EliClientError
from legal_checker.eli_resolver import ActResolutionError
from legal_checker.models import CheckRequest, ComplianceResult, ComplianceStatus

STATUS_STYLES = {
    ComplianceStatus.COMPLIANT: "bold green",
    ComplianceStatus.PARTIALLY_COMPLIANT: "bold yellow",
    ComplianceStatus.NON_COMPLIANT: "bold red",
    ComplianceStatus.NOT_APPLICABLE: "bold",
}
NOT_RUN = "nie wykonano"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="legal-checker",
        description="Sprawdza zgodność fragmentu dokumentu firmowego z aktem prawnym z API ELI Sejmu.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="fragment dokumentu do oceny")
    source.add_argument("--file", type=Path, help="plik UTF-8 z fragmentem dokumentu ('-' czyta ze stdin)")
    parser.add_argument(
        "--hint",
        required=True,
        help="wskazówka aktu: ELI (DU/2014/827), link ELI, sygnatura (Dz.U. z 2014 r. poz. 827) lub tytuł",
    )
    return parser.parse_args(argv)


def read_fragment(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if str(args.file) == "-":
        return sys.stdin.read()
    return args.file.read_text(encoding="utf-8")


def tool_calls(state: GraphState) -> list[tuple[str, dict, list[str] | None]]:
    """Each tool call the agent requested, in order: name, arguments and the ``unit_id``s it returned.

    ``None`` instead of the ``unit_id``s marks a call that was not run (over the tool-call budget).
    """
    messages = state.get("messages", [])
    results = {
        message.tool_call_id: message.artifact
        for message in messages
        if isinstance(message, ToolMessage) and message.status != "error"
    }
    return [
        (call["name"], call["args"], results.get(call["id"]))
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]


def print_report(console: Console, state: GraphState) -> None:
    result: ComplianceResult = state["result"]
    basis = result.legal_basis
    console.print(
        Panel(
            f"[{STATUS_STYLES[result.status]}]{result.status.value}[/]  (pewność: {result.confidence:.2f})\n\n"
            f"{result.rationale}",
            title="Ocena zgodności",
        )
    )

    details = Table.grid(padding=(0, 2))
    details.add_column(style="bold")
    details.add_column()
    details.add_row("Akt", f"{basis.act_title} ({basis.eli})")
    details.add_row("Przepis", basis.provision or "—")
    details.add_row("Stan prawny na", basis.as_of_date)
    details.add_row("Źródło", basis.source_url)
    details.add_row("Cytat z dokumentu", result.document_excerpt or "—")
    details.add_row("Propozycja poprawki", result.suggested_fix or "—")
    console.print(details)

    console.print(f"\n[bold]Przepisy widziane przez agenta[/] ({len(result.consulted_units)}):")
    for unit_id in result.consulted_units:
        console.print(f"  • {unit_id}")

    calls = tool_calls(state)
    console.print(f"\n[bold]Wywołania narzędzi agenta[/] ({len(calls)}):")
    for name, args, unit_ids in calls:
        arguments = ", ".join(f"{key}={value!r}" for key, value in args.items())
        returned = NOT_RUN if unit_ids is None else ", ".join(unit_ids) or "brak wyników"
        console.print(f"  • {name}({arguments}) → {returned}", highlight=False)

    if result.warnings:
        console.print("\n[bold yellow]Ostrzeżenia:[/]")
        for warning in result.warnings:
            console.print(f"  • {warning}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    console = Console()
    errors = Console(stderr=True)
    try:
        request = CheckRequest(document_fragment=read_fragment(args), act_hint=args.hint)
        state = run(request, get_settings())
    except (OSError, ValidationError, ActResolutionError, EliClientError, EvaluationError) as exc:
        errors.print(f"[bold red]Błąd:[/] {exc}", highlight=False)
        return 1

    print(state["result"].model_dump_json(indent=2))
    console.print()
    print_report(console, state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
