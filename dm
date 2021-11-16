#!/usr/bin/env python3

__author__ = "TruncatedDinosour"
__email__ = "truncateddinosour@gmail.com"
__version__ = "0.1-pre"


import asyncio
import sys
import os
import git
import shutil
import time
import json
import fuzzysearch
import hashlib

import urllib.request

from tqdm import tqdm

from configparser import ConfigParser
from plyer import notification
from colorama import Fore
from typing import Tuple

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
        sys.stderr.write(
            f"{Fore.LIGHTBLUE_EX}[{colour}{header.upper()}{Fore.LIGHTBLUE_EX}]{Fore.RESET} {s}\n"
        )
        return

    sys.stderr.write(f"[{header.upper()}] {s}\n")  # Else condition


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


class Download:
    @classmethod
    def http(cls, url: str, filename: str) -> None:
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
        ) as t:
            urllib.request.urlretrieve(url, filename=filename, reporthook=t.update_to)


class CheckChecksum:
    @classmethod
    def sha256(cls, filename: str, orig_sha256_hash: str) -> Tuple[bool, str]:
        """This function calculates tha SHA256 sum of a specified file"""
        sha256_hash = hashlib.sha256()

        with open(filename, "rb") as file:
            log(f"Calculating SHA256 hash for {filename}")

            for byte_block in iter(lambda: file.read(4096), b""):
                sha256_hash.update(byte_block)

        return (sha256_hash.hexdigest() == orig_sha256_hash, sha256_hash.hexdigest())

    @classmethod
    def sha512(cls, filename: str, orig_sha512_hash: str) -> Tuple[bool, str]:
        """This function calculates tha SHA512 sum of a specified file"""
        sha512_hash = hashlib.sha512()

        with open(filename, "rb") as file:
            log(f"Calculating SHA512 hash for {filename}")

            for byte_block in iter(lambda: file.read(4096), b""):
                sha512_hash.update(byte_block)

        return (sha512_hash.hexdigest() == orig_sha512_hash, sha512_hash.hexdigest())

    @classmethod
    def md5(cls, filename: str, orig_md5_hash: str) -> Tuple[bool, str]:
        """This function calculates tha MD5 sum of a specified file"""
        md5_hash = hashlib.md5()

        with open(filename, "rb") as file:
            log(f"Calculating MD5 hash for {filename}")

            for byte_block in iter(lambda: file.read(4096), b""):
                md5_hash.update(byte_block)

        return (md5_hash.hexdigest() == orig_md5_hash, md5_hash.hexdigest())


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
                if repos[repo]["sync_type"] != "git":
                    continue

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
    check_args(args, 3, "Required arguments: data, sort, query")

    allowed_sort_keys = ["all", "size", "speed"]
    allowed_data_keys = ["all", "name", "description", "url", "keywords", "protocol"]

    data = args[0]
    sort_key = args[1]
    query = " ".join(args[2:])

    if sort_key not in allowed_sort_keys or data not in allowed_data_keys:
        log("illegal both or either sort_key or data_key", "error", Fore.RED)
        sys.exit(1)

    matches = {}

    # TODO: fix this indentation hell
    for repo in os.scandir(get_path(CONFIG["sync"]["location"])):
        """Every iteration repo = full path to a reposiory"""
        with open(f"{repo.path}/REPO.json", "r") as f:
            """Repo_info = dict of repo info"""
            repo_info = json.load(f)

            for url_file in os.scandir(f"{repo.path}/{repo_info['urls']}"):
                """
                Every iteration url_file is a full path to a json file
                containing info about a URL
                """
                with open(url_file.path, "r") as url_info_file:
                    url_info = json.load(url_info_file)
                    """ url_info is a dict of info about a specific url """

                    if data == "all":
                        collected_data = " ".join(
                            [str(info) for info in url_info.values()]
                        )
                    else:
                        collected_data = str(url_info[data])

                # NOTE: `max_l_dist=0` ??works, but what??
                if fuzzysearch.find_near_matches(query, collected_data, max_l_dist=0):
                    atom_name = os.path.splitext(os.path.split(url_file.path)[1])[0]

                    if repo.name not in matches:
                        matches[repo.name] = []

                    if sort_key == "all":
                        sort_number = url_info["size"] / url_info["speed"]
                    else:
                        sort_number = url_info[sort_key]

                    matches[repo.name].append((sort_number, atom_name))

    for repository, matches in matches.items():
        matches.sort()

        log(f"Matches in repository {repository}", "match", Fore.LIGHTGREEN_EX)
        for _, match in matches:
            print(f"\t* {match}")


def download(*args) -> None:
    check_args(args, 1, "Required arguments: repo@atom-name")

    for atom in args:
        subatom, package = atom.split("@", 1)

        repo_path = f"{get_path(CONFIG['sync']['location'])}/{subatom}"

        with open(f"{repo_path}/REPO.json") as f:
            urls = json.load(f)["urls"]

        download_file = f"{repo_path}/{urls}/{package}.json"

        with open(download_file, "r") as f:
            download_info = json.load(f)

        filename = os.path.split(download_info["url"])[1]

        if os.path.exists(filename):
            log(
                f"File {filename} already exists",
                "error",
                Fore.RED,
            )
            sys.exit(1)

        protocols: dict = {
            "https": {
                "fn": Download.http,
                "args": [download_info["url"], filename],
            },
            "http": {
                "fn": Download.http,
                "args": [download_info["url"], filename],
            },
        }

        checksums: dict = {
            "sha256": CheckChecksum.sha256,
            "sha512": CheckChecksum.sha512,
            "md5": CheckChecksum.md5,
        }

        download_protocol_info = protocols.get(download_info["protocol"])

        if download_protocol_info is None:
            log(f"Unsupported protocol: {download_info['protocol']}", "error", Fore.RED)
            sys.exit(1)

        log(f"Downloading {filename} over the {download_info['protocol']} protocol")
        time.sleep(5)  # Give users time to think if they want to actually do it
        download_protocol_info["fn"](*download_protocol_info["args"])

        checksum_dict = download_info.get("checksums")

        if checksum_dict is None:
            log("No checksums found", "warning", Fore.YELLOW)

        for checksum_type, checksum in checksum_dict.items():
            checker = checksums.get(checksum_type)

            if checker is None:
                log(
                    f"Checksum type {checksum_type} is not found",
                    "warning",
                    Fore.LIGHTYELLOW_EX,
                )
                continue

            is_good_checksum, calculated_checksum = checker(filename, checksum)

            if not is_good_checksum:
                log(
                    f"{checksum_type} checksum check for {filename} failed",
                    "error",
                    Fore.RED,
                )
                log(f"Real checksum: {checksum}")
                log(f"Got: {calculated_checksum}")

                log("Be careful when you use this file", "warning", Fore.LIGHTYELLOW_EX)

                if input(f"Remove {filename}? [Y/n]: ").lower() != "n":
                    log(f"Removing {filename}")
                    os.remove(filename)

                sys.exit(1)


def show_version(*args) -> None:
    del args
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
            "desc": "Search for a fuzzy string in a specific portion of the info file, speed/size/all are the sorting keys and query is the query",
            "func": search,
            "args": [
                "name|description|url|keywords|protocol|all",
                "speed|size|all",
                "query",
            ],
            "is_async": False,
        },
        "download": {
            "desc": "Download a file",
            "func": download,
            "args": [
                "subatom@pkg",
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
