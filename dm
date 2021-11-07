#!/usr/bin/env python3

__author__ = "TruncatedDinosour"
__email__ = "truncateddinosour@gmail.com"
__version__ = '0.1-pre'


import asyncio
import sys, os
import git
import shutil
import time
import json
import fuzzysearch

import urllib.request

from tqdm import tqdm

from configparser import ConfigParser
from plyer import notification
from colorama import Fore

from colorama import init as colorama_init
from dirsync import sync as sync_directories

colorama_init()

# Helpers and config


def get_path(p: str) -> str:
    return os.path.expanduser(p)


CONFIG = ConfigParser()
CONFIG.read(get_path("~/.config/dm/dm.ini"))


def log(s: str, header: str = "log", colour: str = Fore.LIGHTYELLOW_EX) -> None:
    show_colours: bool = CONFIG["ui"].getboolean("show_colours")

    if show_colours:
        return print(
            f"{Fore.LIGHTBLUE_EX}[{colour}{header.upper()}{Fore.LIGHTBLUE_EX}]{Fore.RESET} {s}"
        )

    return print(f"[{header.upper()}] {s}")  # Else condition


def usage(actions: dict) -> None:
    for action in actions:
        log(
            f"{action} <{', '.join(actions[action]['args'])}>  --  {actions[action]['desc']}",
            "usage",
            Fore.LIGHTGREEN_EX,
        )


def send_notification(msg: str) -> None:
    notification.notify(
        "Dm notification", msg, "dm", timeout=0, ticker=f"DM notification - {msg}"
    )


def check_args(args: tuple, count: int, msg: str) -> None:
    if len(args) < count:
        log(msg, "error", Fore.RED)
        sys.exit(1)


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_with_progress(url: str, output_path: str):
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


# Classes


class QuietLogger:
    @classmethod
    def info(*args):
        pass


class Sync:
    @classmethod
    def git(cls, url: str, to: str) -> None:
        if os.path.exists(to):
            log(f"Hard resetting repo {to}")
            repo = git.Repo(to)
            repo.git.reset("--hard")

            log(f"Pulling repo {to}")
            repo.git.pull()
            return

        log(f"Cloning repo from {url}")
        git.Repo.clone_from(url, get_path(to))

    @classmethod
    def local(cls, from_location: str, to_location: str) -> None:
        if not os.path.exists(to_location):
            log("creating local repository directory")
            os.mkdir(to_location)

        log(f"Syncing local repository {to_location} with {from_location}")
        sync_directories(from_location, to_location, "sync", logger=QuietLogger)


# Functionality


def sync(*args) -> None:
    repo = args[0] if args else None
    repos = ConfigParser()
    repos.read(get_path(CONFIG["repos"]["conf"]))

    def sync_repo(name: str, sync_type: str, url: str) -> None:
        log(f"Syncing {name}")

        repo_path = f"{get_path(CONFIG['sync']['location'])}/{name}"


        sync = {
            "git": {
                "args": [url, repo_path],
                "fn": Sync.git,
            },
            "local": {"args": [url, repo_path], "fn": Sync.local},
        }

        try:
            fn = sync[sync_type]["fn"]
        except KeyError:
            log(f"SyncType `{sync_type}` not found", "error", Fore.RED)
            sys.exit(1)

        fn(
            *sync[sync_type]["args"]
        )  # NOTE: mypy might see it as an object when it's not

    if repo is not None:
        sync_repo(repo, repos[repo]["sync_type"], repos[repo]["location"])
        return

    for repo_name in repos.sections():
        sync_repo(
            repo_name, repos[repo_name]["sync_type"], repos[repo_name]["location"]
        )


def rm_repo(*args) -> None:
    check_args(args, 1, "Required argument: repo")

    repo_path = f"{get_path(CONFIG['sync']['location'])}/{args[0]}"

    if not os.path.exists(repo_path):
        log(f"Repository {repo_path} does not exist", "error", Fore.RED)
        sys.exit(1)

    log(f"Removing {repo_path}")
    shutil.rmtree(repo_path)


def clean_repos(*args):
    synced_repos = os.listdir(get_path(CONFIG["sync"]["location"]))
    current_repos = ConfigParser()
    current_repos.read(get_path(CONFIG["repos"]["conf"]))

    for repo in current_repos.sections():
        try:
            synced_repos.remove(repo)
        except ValueError:
            log(f"{repo} not in synced repos", "warning", Fore.YELLOW)

    for repo in synced_repos:
        repo_path = f"{get_path(CONFIG['sync']['location'])}/{repo}"
        log(f"Removing {repo_path}")
        shutil.rmtree(repo_path)


def git_notification_daemon(*args) -> None:
    check_args(args, 1, "Required argument: sleep_time")

    time.sleep(10)  # Wait for internet connection

    while True:
        try:
            synced_repos = os.listdir(get_path(CONFIG["sync"]["location"]))
            repos = ConfigParser()
            repos.read(get_path(CONFIG["repos"]["conf"]))

            for repo in synced_repos:
                if repos[repo]["sync_type"] != "git": continue

                repo_obj = git.Repo(f"{get_path(CONFIG['sync']['location'])}/{repo}")
                repo_obj_branch = repo_obj.active_branch.name

                repo_obj.remotes.origin.fetch()
                commits_behind = len(
                    list(
                        repo_obj.iter_commits(
                            f"{repo_obj_branch}..origin/{repo_obj_branch}"
                        )
                    )
                )

                if commits_behind:
                    send_notification(f"Repository {repo} is out of date")

            time.sleep(int(args[0]))
        except Exception as e:
            err = str(e)
            time.sleep(5)  # Wait for the session to start

            send_notification(err)
            log(err, "error", Fore.RED)
            sys.exit(1)


def search(*args):
    check_args(args, 4, "Required arguments: data, fuzzy, sort, query")

    allowed_sort_keys = ["all", "size", "speed"]

    data = args[0]
    is_fuzzy = args[1] == "true"
    sort_key = args[2]
    query = " ".join(args[3:])

    if sort_key not in allowed_sort_keys or data in allowed_sort_keys and data != "all":
        log("illegal both or either sort_key or data_key", "error", Fore.RED)
        sys.exit(1)

    matches = {}

    for file in os.scandir(get_path(CONFIG["sync"]["location"])):

        with open(f"{file.path}/REPO.json", "r") as f:
            repo_info = json.load(f)

        for url in os.scandir(f"{file.path}/{repo_info['urls']}"):
            with open(url.path, "r") as f:
                url_info = json.load(f)

        collected_data = ""
        if data == "all":
            for url_info_data in url_info.values():
                collected_data += str(url_info_data)
        else:
            collected_data = url_info.get(data)
            if collected_data is None:
                log(f"Key {data} not found", "error", Fore.RED)
                sys.exit(1)

            collected_data = (
                " ".join(collected_data)
                if isinstance(collected_data, list)
                else collected_data
            )

        l_dist = len(collected_data) // 4 if is_fuzzy else 1
        if fuzzysearch.find_near_matches(query, collected_data, max_l_dist=l_dist):
            repo_name = os.path.split(file)[1]
            if matches.get(repo_name) is None:
                matches[repo_name] = []

            if sort_key == "all":
                sort_number = int(url_info["size"]) / int(url_info["speed"])
            else:
                sort_number = int(url_info[sort_key])

            matches[repo_name].append((sort_number, os.path.split(url)[1]))

    merged_data = [data for data in matches.items()]
    merged_data.sort()

    for repo, items in merged_data:
        log(f"In repo {repo}:", "match", Fore.LIGHTGREEN_EX)

        for item in items:
            print(f"\t{os.path.splitext(item[1])[0]}")


def download(*args) -> None:
    check_args(args, 2, "Required arguments: repo, atom-name")
    repo_path = f"{get_path(CONFIG['sync']['location'])}/{args[0]}"

    with open(f"{repo_path}/REPO.json") as f:
        urls = json.load(f)["urls"]

    download_file = f"{repo_path}/{urls}/{args[1]}.json"

    with open(download_file, "r") as f:
        download_url = json.load(f)["url"]

    download_with_progress(download_url, os.path.split(download_url)[1])


def show_version(*args) -> None:
    log(__version__, "version", Fore.GREEN)


async def main() -> int:
    ACTIONS: dict = {
        "sync": {
            "desc": "Download and sync repositories",
            "func": sync,
            "args": ["repository"],
            "is_async": False,
        },
        "rm-repo": {
            "desc": "Remove a repository from synced repsitories",
            "func": rm_repo,
            "args": ["repository"],
            "is_async": False,
        },
        "clean": {
            "desc": "Clean repositories that are not in repos.ini anymore",
            "func": clean_repos,
            "args": [],
            "is_async": False,
        },
        "git-notification-daemon": {
            "desc": "Check if local repositories are out of date from remote ones in the background, send updates as notifications",
            "func": git_notification_daemon,
            "args": ["sleep_time"],
            "is_async": False,
        },
        "search": {
            "desc": "Search for a fuzzy string in a specific portion of the info file, true/false to if check for words or fuzzy strings, speed/size are the sorting keys and query is the query",
            "func": search,
            "args": [
                "name|description|url|keywords|protocol|all",
                "true|false",
                "speed|size|all",
                "query",
            ],
            "is_async": False,
        },
        "download": {
            "desc": "Download a file",
            "func": download,
            "args": [
                "repo",
                "atom-name",
            ],
            "is_async": False,
        },
        "version": {
            "desc": "Show version and exit",
            "func": show_version,
            "args": [],
            "is_async": False,
        },
    }

    try:
        fn, fn_async = ACTIONS[sys.argv[1]]["func"], ACTIONS[sys.argv[1]]["is_async"]
    except (KeyError, IndexError):
        usage(ACTIONS)
        return 1

    if fn_async:
        await fn(*sys.argv[2:])
    else:
        fn(*sys.argv[2:])

    return 0


if __name__ == "__main__":
    assert main.__annotations__.get("return") is int, "main() should return an integer"
    exit(asyncio.run(main()))

