"""The `systemone` command line.

systemone login
systemone search routing --capability route
systemone show biplov/snake-balanced-multilingual
systemone pull biplov/snake-balanced-multilingual --variant onnx-int8
systemone push                       # find the models here and choose
systemone push ./runs/my-run/model --repo me/my-model
"""

from __future__ import annotations

import os
import socket
import sys
import webbrowser
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

from systemone import __version__, config
from systemone import cache as local_cache
from systemone.auth import can_open_browser, device_login
from systemone.client import Client
from systemone.discover import Candidate, describe, discover, next_version, parse_selection
from systemone.errors import ApiError, AuthError, ConnectionFailed, NotFound, SystemOneError
from systemone.inspect import build_manifest, inspect, split_front_matter
from systemone.publish import Part, Release, home_mentions, merge, model_card, releases
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
    # Decimal units, as Finder, Hugging Face and the upload progress bar count.
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
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


def _revoke(stored: config.Config) -> str:
    """Revoke the stored token on its registry: "revoked", "invalid", or why not.

    Found by the display prefix the registry keeps for every token, which is the
    start of the token itself; the raw value is never sent anywhere but in the
    Authorization header, as on every other request.
    """
    token = stored.token or ""
    try:
        with Client(stored) as registry:
            mine = [
                entry
                for entry in registry.tokens()
                if entry.get("prefix") and token.startswith(str(entry["prefix"]))
            ]
            if len(mine) != 1:
                return "it is not among your account's tokens"
            registry.revoke_token(str(mine[0]["id"]))
    except AuthError:
        return "invalid"
    except ConnectionFailed:
        return "the registry could not be reached"
    except ApiError as exc:
        return f"the registry answered HTTP {exc.status}"
    except SystemOneError as exc:
        return str(exc)
    return "revoked"


@app.command()
def logout() -> None:
    """Sign out: revoke the stored token, and forget this machine's login.

    Everything `login` stored goes, the registry's address included, so the
    next `systemone login` goes to the default registry unless given
    --endpoint. A token given in SYSTEMONE_TOKEN is never touched.
    """
    stored = config.load(environment=False)
    if not stored.token:
        config.clear()
        console.print("Not signed in on this machine.")
    else:
        outcome = _revoke(stored)
        config.clear()
        site = config.web_url(stored.endpoint)
        if outcome == "revoked":
            console.print(f"Signed out of {site}. The token was revoked.")
        elif outcome == "invalid":
            console.print(f"Signed out of {site}. The token had already stopped working.")
        else:
            console.print(f"Signed out on this machine, but the token is still valid: {outcome}.")
            console.print(f"Revoke it at [cyan]{site}/settings/tokens[/cyan].")
    if os.environ.get(config.ENV_TOKEN):
        errs.print(
            "[yellow]SYSTEMONE_TOKEN is set in this shell and still signs you in. "
            "Unset it to stop.[/yellow]"
        )


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
        facts.add_row("median latency", f"{evaluation['median_latency_ms']:.1f} ms")
    if evaluation.get("p95_latency_ms") is not None:
        facts.add_row("p95 latency", f"{evaluation['p95_latency_ms']:.1f} ms")
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


def _interactive() -> bool:
    return sys.stdin.isatty()


def _home(path: Path) -> str:
    home = str(Path.home())
    text = str(path)
    return "~" + text[len(home) :] if text.startswith(home) else text


def _show_candidates(candidates: list[Candidate], root: Path) -> None:
    console.print(f"Found {len(candidates)} models under [bold]{escape(_home(root))}[/bold]\n")
    table = Table(box=None, pad_edge=False, header_style="dim")
    table.add_column("#", justify="right", style="cyan")
    table.add_column("MODEL", no_wrap=True)
    table.add_column("VARIANT", no_wrap=True)
    table.add_column("SIZE", justify="right", style="dim", no_wrap=True)
    table.add_column("FOLDER", style="dim", no_wrap=True, overflow="ellipsis")
    for number, candidate in enumerate(candidates, 1):
        table.add_row(
            str(number),
            candidate.name,
            candidate.variant,
            human(candidate.size_bytes),
            escape(candidate.relative_to(root)),
        )
    console.print(table)


def _ask_which(candidates: list[Candidate]) -> list[Candidate]:
    console.print("\n[dim]Variants of one model are published together, a folder each.[/dim]")
    while True:
        answer = typer.prompt("Publish which? (e.g. 1,3-5 or all)")
        try:
            return [candidates[i] for i in parse_selection(answer, len(candidates))]
        except ValueError as exc:
            errs.print(f"[red]{exc}[/red]")


def _existing_versions(registry: Client, repo: str, lenient: bool) -> list[str] | None:
    """The versions a repository already has, or None when it does not exist."""
    try:
        return [str(entry["version"]) for entry in registry.versions(repo)]
    except NotFound:
        return None
    except SystemOneError as exc:
        if lenient:
            return None
        fail(f"{repo}: {exc}")


def _publish(
    registry: Client,
    release: Release,
    document: str,
    card: str,
    notes: str | None,
    private: bool,
) -> None:
    check = registry.validate_manifest(document)
    if not check["valid"]:
        for issue in check["issues"]:
            errs.print(f"[red]{issue['path']}[/red]  {issue['message']}")
        raise SystemOneError("the manifest is not valid; fix it or pass --manifest")

    if not release.exists:
        registry.create_model(release.namespace, release.name, document, private)
        console.print(f"created {release.repo}{' (private)' if private else ''}")

    artifacts: list[dict[str, Any]] = []
    deduplicated = 0
    with Progress(
        TextColumn("[dim]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(release.repo, total=release.total_bytes)

        def on_file(path: str) -> None:
            progress.update(task, description=escape(path[-40:]))

        def on_bytes(count: int) -> None:
            progress.advance(task, count)

        def tick(_path: str, size: int, _total: int, skipped: bool) -> None:
            nonlocal deduplicated
            if skipped:
                deduplicated += size

        for part in release.parts:
            artifacts += push_files(
                registry,
                release.repo,
                part.source,
                part.found.files,
                part.folder,
                tick,
                on_file=on_file,
                on_bytes=on_bytes,
            )

    registry.publish_version(release.repo, release.version, document, artifacts, notes, card)
    site = config.web_url(registry.config.endpoint)
    console.print(
        f"[green]✓[/green] Published [bold]{release.repo}@{release.version}[/bold] "
        f"with {len(artifacts)} files  [cyan]{site}/{release.repo}[/cyan]"
    )
    if deduplicated:
        console.print(
            f"  [dim]{human(deduplicated)} was already in the registry and was not sent[/dim]"
        )


@app.command()
def push(
    source: Annotated[
        Path,
        typer.Argument(help="A model folder, or a folder to search for models. Default: here."),
    ] = Path("."),
    repo: Annotated[
        str | None,
        typer.Option("--repo", "-r", help="namespace/name. Default: yours, named after the model."),
    ] = None,
    namespace: Annotated[
        str | None, typer.Option("--namespace", "-n", help="Publish under this organization.")
    ] = None,
    version: Annotated[
        str | None, typer.Option(help="Default: the next minor version, from 0.1.0.")
    ] = None,
    variant: Annotated[
        str | None, typer.Option(help="Put one model's files under this folder.")
    ] = None,
    manifest: Annotated[
        Path | None,
        typer.Option(
            help="Use this systemone.yaml instead of inferring one.", exists=True, dir_okay=False
        ),
    ] = None,
    readme: Annotated[
        Path | None,
        typer.Option(
            help="The model card. Default: the model's README.md.", exists=True, dir_okay=False
        ),
    ] = None,
    license_: Annotated[
        str | None,
        typer.Option("--license", help="apache-2.0, mit, … Default: the model card's."),
    ] = None,
    notes: Annotated[str | None, typer.Option(help="Release notes.")] = None,
    private: Annotated[bool, typer.Option(help="Create new repositories private.")] = False,
    all_: Annotated[
        bool, typer.Option("--all", help="Publish every model found, without asking which.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation.")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show the plan and stop.")] = False,
) -> None:
    """Publish models from a folder.

    Point it at one model, or at a whole workspace: every model underneath is
    found — trained checkpoints, and their ONNX and Core ML exports — and you
    choose which to publish. Variants of one model share a repository, a folder
    each. A README.md becomes the model card, its front matter the licence and
    tags; without one, a card is written from the evaluation.
    """
    root = source.expanduser().resolve()
    if not root.is_dir():
        fail(f"not a directory: {root}")
    target: tuple[str, str] | None = None
    if repo is not None:
        owner, _, name = repo.strip().partition("/")
        if not owner or not name or "/" in name:
            fail("--repo must be namespace/name")
        target = (owner.lower(), name.lower())
    interactive = _interactive()

    # Before anything is scanned or asked: finding out after choosing models
    # that you were never signed in wastes the choosing.
    signed_in_as: str | None = None
    if not dry_run:
        with client() as probe:
            try:
                me = probe.whoami()
            except SystemOneError as exc:
                fail(str(exc))
        if not me:
            fail("Not signed in. Run `systemone login`.")
        signed_in_as = str(me["username"])

    here = describe(root)
    candidates = [here] if here else discover(root)
    listed = len(candidates) > 1

    if not candidates:
        if target is None:
            fail(
                f"no model found under {_home(root)}. A model folder holds model.onnx, "
                "model.safetensors, a .gguf file or a .mlpackage. To publish this folder "
                "as it is, pass --repo namespace/name."
            )
        found = inspect(root)
        if not found.files:
            fail(f"no files under {root}")
        plan = [Release(target[1], [Part(root, (variant or "").strip("/"), found)])]
    else:
        chosen = candidates
        if listed:
            _show_candidates(candidates, root)
            if not all_:
                if not interactive:
                    fail(
                        "several models found. Pass --all to publish every one, "
                        "or the folder of the one to publish."
                    )
                chosen = _ask_which(candidates)
        if variant and len(chosen) > 1:
            fail("--variant places one model, and several were chosen")
        plan = releases(chosen, variant)

    if target is not None:
        if len(plan) > 1:
            fail(
                f"--repo names one repository, but the models chosen belong to {len(plan)}. "
                "Choose one model's variants, or leave out --repo."
            )
        plan[0].namespace, plan[0].name = target
    if len(plan) > 1 and (manifest or readme):
        fail("--manifest and --readme describe one repository; choose one model")

    with client() as registry:
        owner_name = (namespace or "").strip().lower() or signed_in_as or registry.config.username
        if not owner_name and any(not r.namespace for r in plan):
            try:
                me = registry.whoami()
            except SystemOneError:
                me = None
            owner_name = str(me["username"]) if me else None
        if any(not r.namespace for r in plan) and not owner_name:
            if not dry_run:
                fail("Not signed in. Run `systemone login`.")
            owner_name = "you"
            errs.print("[yellow]Not signed in: 'you' stands in for your namespace.[/yellow]")

        for release in plan:
            release.namespace = release.namespace or str(owner_name)
            existing = _existing_versions(registry, release.repo, lenient=dry_run)
            release.exists = existing is not None
            release.version = version or next_version(existing or [])

        documents: list[tuple[Release, str, str, bool]] = []
        for release in plan:
            found = merge(release.parts)
            document = (
                manifest.read_text()
                if manifest
                else build_manifest(found, release.namespace, release.name, license_)
            )
            if readme:
                _, card = split_front_matter(readme.read_text())
                generated = False
            else:
                card, generated = model_card(release, found)
            documents.append((release, document, card, generated))

        console.print()
        table = Table(box=None, pad_edge=False, header_style="dim")
        table.add_column("REPOSITORY")
        table.add_column("VERSION")
        table.add_column("FILES", justify="right")
        table.add_column("SIZE", justify="right")
        table.add_column("CARD", style="dim")
        table.add_column("FOLDERS", style="dim")
        for release, _, _, generated in documents:
            table.add_row(
                f"[bold]{release.repo}[/bold]{'' if release.exists else ' [green]new[/green]'}",
                release.version,
                str(release.files),
                human(release.total_bytes),
                "generated" if generated else "README.md",
                ", ".join(release.folders) or "—",
            )
        console.print(table)
        if len(plan) == 1 and target is None:
            console.print("[dim]Another name? Pass --repo namespace/name.[/dim]")

        exposed = [
            f"{release.name}/{hit}" if len(plan) > 1 else hit
            for release in plan
            for hit in home_mentions(release.parts)
        ]
        if exposed:
            shown = ", ".join(exposed[:4]) + (
                f" and {len(exposed) - 4} more" if len(exposed) > 4 else ""
            )
            errs.print(
                f"[yellow]note[/yellow]  {len(exposed)} "
                f"{'file mentions' if len(exposed) == 1 else 'files mention'} your home folder "
                f"({escape(str(Path.home()))}) and will be readable by anyone who can see the "
                f"repository: {escape(shown)}"
            )

        if dry_run:
            for release, document, card, generated in documents:
                console.print(f"\n[bold]{release.repo}@{release.version}[/bold]  systemone.yaml")
                console.print(document.rstrip(), markup=False, highlight=False)
                console.print(
                    f"\n[dim]model card ({'generated' if generated else 'README.md'}):[/dim]"
                )
                console.print(card.rstrip(), markup=False, highlight=False)
                console.print("\n[dim]files:[/dim]")
                for part in release.parts:
                    for path in part.found.files:
                        placed = path.relative_to(part.source).as_posix()
                        console.print(
                            f"  {part.folder + '/' if part.folder else ''}{escape(placed)}"
                        )
            return

        # A folder named with --repo publishes straight away, as it always has;
        # anything the CLI worked out or the user picked from a list is confirmed.
        if interactive and not yes and (target is None or listed):
            count = len(plan)
            question = f"Publish {'it' if count == 1 else f'these {count}'}?"
            if not typer.confirm(question, default=True):
                raise typer.Exit(0)

        failures = 0
        for release, document, card, _ in documents:
            try:
                _publish(registry, release, document, card, notes, private)
            except SystemOneError as exc:
                failures += 1
                errs.print(f"[red]error[/red]  {release.repo}: {exc}")
        if failures:
            raise typer.Exit(1)


if __name__ == "__main__":
    app()
