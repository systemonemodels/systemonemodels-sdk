"""The `systemone` command line.

systemone login
systemone search routing --capability route
systemone show biplov/snake-balanced-multilingual
systemone pull biplov/snake-balanced-multilingual --variant onnx-int8
systemone push ./exports/my-model --repo me/my-model --variant onnx-int8
"""

from __future__ import annotations

import socket
import sys
import webbrowser
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table

from systemone import __version__, config
from systemone import cache as local_cache
from systemone.auth import can_open_browser, device_login
from systemone.client import Client
from systemone.errors import SystemOneError
from systemone.inspect import build_manifest, inspect
from systemone.transfer import pull_version, push_files

app = typer.Typer(
    name="systemone",
    help="Client for the System One decision-model registry.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
errs = Console(stderr=True)


# NoReturn, so the code after a fail() is understood as unreachable and the
# values it uses are not optional there.
def fail(message: str) -> NoReturn:
    errs.print(f"[red]error[/red]  {message}")
    raise typer.Exit(1)


def human(n: float) -> str:
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def client() -> Client:
    return Client(config.load())


@app.command()
def version() -> None:
    """Print the client version."""
    console.print(f"systemone {__version__}")


@app.command()
def login(
    token: Annotated[
        str | None, typer.Option(help="Use this token instead of the browser. For CI.")
    ] = None,
    paste: Annotated[
        bool, typer.Option("--paste", help="Paste a token at a hidden prompt instead.")
    ] = False,
    browser: Annotated[
        bool, typer.Option("--browser/--no-browser", help="Open the approval page automatically.")
    ] = True,
    endpoint: Annotated[str | None, typer.Option(help="For a self-hosted registry.")] = None,
) -> None:
    """Sign in through your browser.

    Shows a one-time code and opens the registry, where you approve this
    machine. Over SSH, open the printed address on any device instead.
    For CI, pass --token or set SYSTEMONE_TOKEN.
    """
    current = config.load()
    if endpoint:
        current.endpoint = endpoint.rstrip("/")

    if token or paste:
        if not token:
            # hide_input: a pasted token should not stay in the scrollback, nor
            # in the terminal's own buffer for the next person at this machine.
            token = typer.prompt("Token", hide_input=True)
        current.token = token.strip()
        with Client(current) as probe:
            try:
                me = probe.whoami()
            except SystemOneError as exc:
                fail(str(exc))
            if not me:
                fail("That token was not accepted.")
        username = str(me["username"])
    else:
        if not sys.stdin.isatty():
            fail("Not a terminal. Pass --token, or set SYSTEMONE_TOKEN.")
        current.token = None

        def show(codes: dict[str, Any]) -> None:
            console.print(f"\nYour one-time code: [bold cyan]{codes['user_code']}[/bold cyan]")
            opened = (
                browser
                and can_open_browser()
                and webbrowser.open(codes["verification_uri_complete"])
            )
            if opened:
                console.print(f"Opened [cyan]{codes['verification_uri']}[/cyan] in your browser.")
            else:
                console.print(
                    f"Open [cyan]{codes['verification_uri']}[/cyan] on any device "
                    "and enter the code."
                )

        with Client(current) as anonymous:
            try:
                with console.status("Waiting for approval in the browser (Ctrl+C to cancel)"):
                    granted = device_login(anonymous, socket.gethostname() or "systemone CLI", show)
            except SystemOneError as exc:
                fail(str(exc))
        current.token = granted["token"]
        username = str(granted["username"])

    current.username = username
    path = config.save(current)
    console.print(f"[green]✓[/green] Signed in as [bold]{username}[/bold]")
    console.print(f"[dim]Token stored in {path} (0600)[/dim]")


@app.command()
def logout() -> None:
    """Forget the stored token."""
    config.clear()
    console.print("Signed out. The token itself is still valid — revoke it in settings.")


@app.command()
def whoami() -> None:
    """Show who the stored token belongs to."""
    with client() as registry:
        me = registry.whoami()
    if not me:
        fail("Not signed in. Run `systemone login`.")
    console.print(f"[bold]{me['username']}[/bold]  {me['email']}")
    if not me.get("email_verified"):
        errs.print("[yellow]Email not verified — publishing is blocked until it is.[/yellow]")


@app.command()
def search(
    query: Annotated[str | None, typer.Argument(help="Free text.")] = None,
    capability: Annotated[list[str] | None, typer.Option(help="choice, score, rank…")] = None,
    architecture: Annotated[list[str] | None, typer.Option()] = None,
    license: Annotated[list[str] | None, typer.Option()] = None,
    sort: Annotated[
        str, typer.Option(help="relevance|recent|stars|downloads|accuracy|latency")
    ] = "relevance",
    limit: Annotated[int, typer.Option()] = 20,
) -> None:
    """Search the registry."""
    with client() as registry:
        try:
            results = registry.search(
                query,
                capability=capability,
                architecture=architecture,
                license=license,
                sort=sort,
                limit=limit,
            )
        except SystemOneError as exc:
            fail(str(exc))

    if not results["items"]:
        console.print("[dim]No models matched.[/dim]")
        return

    table = Table(box=None, pad_edge=False, header_style="dim")
    table.add_column("MODEL")
    table.add_column("ARCH", style="dim")
    table.add_column("CAPABILITIES", style="dim")
    table.add_column("ACC", justify="right")
    table.add_column("p50", justify="right", style="dim")
    table.add_column("↓", justify="right", style="dim")

    for item in results["items"]:
        accuracy = item.get("decision_accuracy")
        latency = item.get("median_latency_ms")
        table.add_row(
            item["full_name"],
            item.get("architecture") or "—",
            ", ".join(item.get("capabilities") or []) or "—",
            f"{accuracy * 100:.1f}%" if accuracy is not None else "—",
            f"{latency:.0f}ms" if latency is not None else "—",
            str(item.get("downloads", 0)),
        )

    console.print(table)
    console.print(f"[dim]{results['total']} total[/dim]")


@app.command()
def show(repo: Annotated[str, typer.Argument(help="namespace/name")]) -> None:
    """Print a model's metadata."""
    with client() as registry:
        try:
            model = registry.model(repo)
            versions = registry.versions(repo)
        except SystemOneError as exc:
            fail(str(exc))

    latest = model.get("latest_version") or ""
    console.print(f"[bold]{model['full_name']}[/bold]  [dim]{latest}[/dim]")
    if model.get("summary"):
        console.print(model["summary"])
    console.print()

    facts = Table(box=None, show_header=False, pad_edge=False)
    facts.add_column(style="dim")
    facts.add_column()
    facts.add_row("architecture", model.get("architecture") or "—")
    facts.add_row("capabilities", ", ".join(model.get("capabilities") or []) or "—")
    facts.add_row("license", model.get("license") or "—")
    if model.get("base_model"):
        facts.add_row("base model", model["base_model"])
    evaluation = model.get("evaluation") or {}
    if evaluation.get("decision_accuracy") is not None:
        facts.add_row("accuracy", f"{evaluation['decision_accuracy'] * 100:.1f}%")
    if evaluation.get("calibration_error") is not None:
        facts.add_row("calibration error", str(evaluation["calibration_error"]))
    if evaluation.get("median_latency_ms") is not None:
        facts.add_row("median latency", f"{evaluation['median_latency_ms']} ms")
    facts.add_row("downloads", str(model.get("downloads", 0)))
    console.print(facts)

    if versions:
        console.print("\n[dim]VERSIONS[/dim]")
        for entry in versions:
            size = sum(a.get("size_bytes") or 0 for a in entry["artifacts"])
            folders = sorted({(a.get("path") or "").split("/")[0] for a in entry["artifacts"]})
            console.print(
                f"  {entry['version']:<10} {len(entry['artifacts'])} files  {human(size):>9}"
                f"  [dim]{', '.join(f for f in folders if f)}[/dim]"
            )


@app.command()
def validate(manifest: Annotated[Path, typer.Argument(help="Path to systemone.yaml")]) -> None:
    """Check a manifest without publishing."""
    if not manifest.exists():
        fail(f"no such file: {manifest}")

    with client() as registry:
        result = registry.validate_manifest(manifest.read_text())

    if result["valid"]:
        console.print("[green]valid[/green]")
        return

    # Every problem at once, rather than one per attempt.
    for issue in result["issues"]:
        errs.print(f"[red]{issue['path']}[/red]  {issue['message']}")
    raise typer.Exit(1)


@app.command()
def pull(
    repo: Annotated[str, typer.Argument(help="namespace/name")],
    dest: Annotated[
        Path | None,
        typer.Option("--dest", "-d", help="Link the tree here. Omit to use the cache path."),
    ] = None,
    version: Annotated[str | None, typer.Option(help="Defaults to the latest.")] = None,
    variant: Annotated[str | None, typer.Option(help="Only this folder, e.g. onnx-int8.")] = None,
) -> None:
    """Download a model.

    Files land in a content-addressed cache shared by every repository, so a
    file you already have — the tokenizer six exports share — is never
    downloaded twice. The tree is then linked where you asked for it.
    """
    with (
        client() as registry,
        Progress(
            TextColumn("[dim]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} files"),
            TimeRemainingColumn(),
            console=console,
        ) as progress,
    ):
        task = progress.add_task("resolving", total=None)

        def tick(path: str, _size: int, count: int, _cached: bool) -> None:
            if progress.tasks[task].total is None:
                progress.update(task, total=count)
            progress.update(task, advance=1, description=path[-40:])

        try:
            result = pull_version(registry, repo, dest, version, variant, tick)
        except SystemOneError as exc:
            fail(str(exc))

    console.print(f"[bold]{result.root}[/bold]")
    parts = []
    if result.downloaded:
        parts.append(f"downloaded {human(result.downloaded)}")
    if result.from_cache:
        parts.append(f"{human(result.from_cache)} from cache")
    console.print(f"[dim]{result.files} files · {' · '.join(parts) or 'nothing to fetch'}[/dim]")


@app.command()
def create(
    repo: Annotated[str, typer.Argument(help="namespace/name")],
    private: Annotated[bool, typer.Option(help="Only your namespace can see it.")] = False,
) -> None:
    """Create an empty repository to push into later."""
    namespace, _, name = repo.partition("/")
    if not namespace or not name:
        fail("repository must be namespace/name")

    with client() as registry:
        try:
            registry.create_model(namespace, name, None, private)
        except SystemOneError as exc:
            fail(str(exc))

    endpoint = config.web_url(config.load().endpoint)
    console.print(f"Created [bold]{repo}[/bold]{' (private)' if private else ''}")
    console.print(f"[dim]{endpoint}/{repo}[/dim]")


cache_app = typer.Typer(help="Inspect or clear the local download cache.", no_args_is_help=True)
app.add_typer(cache_app, name="cache")


@cache_app.command("info")
def cache_info() -> None:
    """Where the cache is and how big it is."""
    console.print(f"{config.cache_dir()}")
    console.print(f"[dim]{human(local_cache.size())} in blobs[/dim]")


@cache_app.command("clear")
def cache_clear(
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask.")] = False,
) -> None:
    """Delete every cached file."""
    if not yes and not typer.confirm(f"Delete {human(local_cache.size())} of cached models?"):
        raise typer.Exit(0)
    console.print(f"Freed {human(local_cache.clear())}")


@app.command()
def push(
    source: Annotated[Path, typer.Argument(help="Directory to publish.")],
    repo: Annotated[str, typer.Option("--repo", "-r", help="namespace/name")],
    version: Annotated[str, typer.Option(help="Version string.")] = "0.1.0",
    variant: Annotated[str, typer.Option(help="Place files under this folder.")] = "",
    manifest: Annotated[
        Path | None, typer.Option(help="Use this instead of inferring one.")
    ] = None,
    notes: Annotated[str | None, typer.Option(help="Release notes.")] = None,
    private: Annotated[bool, typer.Option(help="Create the repository private.")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show what would happen.")] = False,
) -> None:
    """Publish a directory as a version."""
    source = source.expanduser().resolve()
    if not source.is_dir():
        fail(f"not a directory: {source}")
    namespace, _, name = repo.partition("/")
    if not namespace or not name:
        fail("--repo must be namespace/name")

    found = inspect(source)
    if not found.files:
        fail(f"no files under {source}")

    document = manifest.read_text() if manifest else build_manifest(found, namespace, name)

    console.print(
        f"[bold]{repo}[/bold]@{version}  {len(found.files)} files  {human(found.total_bytes)}"
    )
    if not manifest:
        source_of_truth = "inferred from the export" if found.recognised else "a starting point"
        console.print(f"[dim]manifest {source_of_truth}[/dim]")

    if dry_run:
        console.print()
        console.print(document.rstrip())
        console.print("\n[dim]files:[/dim]")
        for path in found.files:
            placed = path.relative_to(source).as_posix()
            console.print(f"  {f'{variant}/' if variant else ''}{placed}")
        return

    with client() as registry:
        check = registry.validate_manifest(document)
        if not check["valid"]:
            for issue in check["issues"]:
                errs.print(f"[red]{issue['path']}[/red]  {issue['message']}")
            fail("the manifest is not valid; fix it or pass --manifest")

        try:
            registry.model(repo)
        except SystemOneError:
            registry.create_model(namespace, name, document, private)
            console.print(f"created {repo}")

        deduplicated = 0
        with Progress(
            TextColumn("[dim]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} files"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("uploading", total=len(found.files))

            def tick(path: str, size: int, _total: int, skipped: bool) -> None:
                nonlocal deduplicated
                if skipped:
                    deduplicated += size
                progress.update(task, advance=1, description=path[-40:])

            try:
                artifacts = push_files(registry, repo, source, found.files, variant, tick)
            except SystemOneError as exc:
                fail(str(exc))

        try:
            registry.publish_version(repo, version, document, artifacts, notes)
        except SystemOneError as exc:
            fail(str(exc))

    console.print(f"Published [bold]{repo}@{version}[/bold] with {len(artifacts)} files")
    if deduplicated:
        console.print(
            f"[dim]{human(deduplicated)} was already in the registry and was not sent[/dim]"
        )


if __name__ == "__main__":
    app()
