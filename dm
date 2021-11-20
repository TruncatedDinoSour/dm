#!/usr/bin/env python3
"""
DM (download manager)
"""

__author__ = "TruncatedDinosour"
__email__ = "truncateddinosour@gmail.com"
__version__ = "0.1-pre"
__source__ = "https://ari-web.netlify.app/gh/dm"


import sys
import os
import git
import shutil
import time
import json
import fuzzysearch
import hashlib
import tldextract
import random
import libtorrent as lt

import urllib.request
from urllib.parse import urlparse

from tqdm import tqdm

from configparser import ConfigParser
from colorama import Fore
from typing import Tuple, Dict, Set, Any

from colorama import init as colorama_init
from dirsync import sync as sync_directories

colorama_init()

# Constants

FILE_ENCODING = "utf-8"

# Helpers and config


def get_path(path: str) -> str:
    """Get the full path from an expandable path like ~/.config"""

    return os.path.expanduser(path)


CONFIG = ConfigParser()
CONFIG.read(get_path("~/.config/dm/dm.ini"))


def log(msg: str, header: str = "log", colour: str = Fore.LIGHTYELLOW_EX) -> None:
    """Show log/debug messages"""

    show_colours: bool = CONFIG["ui"].getboolean("show_colours")

    if show_colours:
        sys.stderr.write(
            f"{Fore.LIGHTBLUE_EX}[{colour}{header.upper()}{Fore.LIGHTBLUE_EX}]{Fore.RESET} {msg}\n"
        )
        return

    sys.stderr.write(f"[{header.upper()}] {msg}\n")  # Else condition


def usage(actions: dict) -> None:
    """Print usage in a formatted way"""

    for action in actions:
        log(
            f"{action} <{', '.join(actions[action]['args'])}>  --  {actions[action]['desc']}",
            "usage",
            Fore.LIGHTGREEN_EX,
        )


def check_args(args: tuple, count: int, msg: str) -> None:
    """Check if the ammount of arguments is correct"""

    if len(args) < count:
        log(msg, "error", Fore.RED)
        sys.exit(1)


def verify_checksum(
    checksum_type: str, filename: str, orig_hash: str
) -> Tuple[bool, str]:
    """This calculates and verifies a checksum of a file"""

    if checksum_type not in hashlib.algorithms_available or CONFIG.getboolean(
        "hash_ignore", checksum_type, fallback=False
    ):
        log(
            f"Checksum type {checksum_type} is not found",
            "warning",
            Fore.LIGHTYELLOW_EX,
        )

        return (True, "")

    hash_algo = getattr(hashlib, checksum_type)()

    with open(filename, "rb") as file:
        log(f"Calculating {checksum_type.upper()} hash for {filename}")
        for byte_block in iter(lambda: file.read(2048), b""):
            hash_algo.update(byte_block)

    return (hash_algo.hexdigest() == orig_hash, hash_algo.hexdigest())


def features_of(download_json: Dict[str, Any]) -> Dict[str, Dict[str, Set[str]]]:
    """Get and return features of a given atom"""

    features = {"domain": {}, "features": set()}

    download_url = download_json.get("url")

    if download is None:
        features["features"].add("download_protocol_empty")
    else:
        parsed_url, parsed_domain = urlparse(download_url), tldextract.extract(
            download_url
        )

        """
        do not repeat yourself
        do not repeat yourself
        do not repeat yourself
        oops
        """

        features["features"].add(f"download_protocol_{parsed_url.scheme}")
        features["features"].add(f"download_protocol_{download_json['protocol']}")
        features["features"].add(
            f"download_subdomain_{parsed_domain.subdomain.split('.', 1)[0]}"
        )
        features["features"].add(f"download_tld_{parsed_domain.suffix}")
        features["features"].add(
            f"download_filename_{os.path.split(parsed_url.path)[1]}"
        )
        features["domain"]["domain"] = ".".join(
            (parsed_domain.subdomain, parsed_domain.registered_domain)
        )

    return features


# Classes


class DownloadProgressBar(tqdm):
    """Download progress bar object"""

    def update_to(self, b=1, bsize=1, tsize=None):
        """Bar updator callback"""

        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class QuietLogger:
    """Remove logger messages using logging modules"""

    @classmethod
    def info(cls, *args):
        """Info log"""

        del args


class Sync:
    """Sync methods and types"""

    @classmethod
    def git(cls, url: str, to: str) -> None:
        """Sync repositories using git"""

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
        """Locally compare and copy directories"""

        if not os.path.exists(to_location):
            log("creating local repository directory")
            os.mkdir(to_location)

        log(f"Syncing local repository {to_location} with {from_location}")
        sync_directories(from_location, to_location, "sync", logger=QuietLogger)


class Download:
    """Download methods and types"""

    @classmethod
    def http(cls, url: str, filename: str) -> str:
        """HTTP file downloaer"""

        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
        ) as t:
            urllib.request.urlretrieve(url, filename=filename, reporthook=t.update_to)

        return filename

    @classmethod
    def torrent(cls, url: str, checksums: Dict[str, str]) -> None:
        """Torrent file downloaer"""

        time.sleep(random.randint(0, 3))  # Minimise colisions

        torrent_file = f"/tmp/dm_torrent_{random.randint(1, 80_000_000_000)}.torrent"

        log(f"Downloading torrent file {torrent_file}")
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
        ) as t:
            urllib.request.urlretrieve(
                url, filename=torrent_file, reporthook=t.update_to
            )

        if checksums:
            for checksum_type, checksum in checksums.items():
                is_good_checksum, calculated_checksum = verify_checksum(
                    checksum_type, torrent_file, checksum
                )

                if not is_good_checksum:
                    log(
                        f"Checksum for {torrent_file} was missmatched",
                        "error",
                        Fore.RED,
                    )
                    log(f"Calculated hash: {calculated_checksum}")
                    log(f"Supplied hash: {checksum}")

                    if input("Keep going [y/N]: ").lower() != "y":
                        log(f"Removing {torrent_file}")
                        os.remove(torrent_file)
                        sys.exit(1)
        else:
            log("No torrent checksums found", "warning", Fore.LIGHTYELLOW_EX)
            if input("Keep going [y/N]: ").lower() != "y":
                log(f"Removing {torrent_file}")
                os.remove(torrent_file)
                sys.exit()

        ses = lt.session({"listen_interfaces": "0.0.0.0:6881"})

        info = lt.torrent_info(torrent_file)
        torrent = ses.add_torrent({"ti": info, "save_path": "."})
        torrent_status = torrent.status()

        log(f"Starting downloading {torrent_status.name}")
        time.sleep(2)

        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=torrent.status().name
        ) as progress_bar:
            while not torrent_status.is_seeding:
                torrent_status = torrent.status()

                progress_bar.update_to(
                    b=torrent_status.total_download,
                    bsize=torrent_status.progress,
                    tsize=torrent_status.total_wanted,
                )

                alerts = ses.pop_alerts()
                for alert in alerts:
                    if alert.category() & lt.alert.category_t.error_notification:
                        log(str(alert), "alert", Fore.LIGHTMAGENTA_EX)

                time.sleep(1)

        log(f"Removing {torrent_file}")
        os.remove(torrent_file)

        return torrent_status.name


# Functionality


def sync(*args) -> None:
    """Sync a repository / repositories"""

    specified_repos = args if args else None
    repos = ConfigParser()
    repos.read(get_path(CONFIG["repos"]["conf"]))

    def sync_repo(name: str, sync_type: str, url: str) -> None:
        """Sync a repository - helper"""

        log(f"Syncing {name}")

        repo_path = f"{get_path(CONFIG['sync']['location'])}/{name}"

        sync_types = {
            "git": {
                "args": [url, repo_path],
                "fn": Sync.git,
            },
            "local": {"args": [get_path(url), repo_path], "fn": Sync.local},
        }

        try:
            fn = sync_types[sync_type]["fn"]
        except KeyError:
            log(f"SyncType `{sync_type}` not found", "error", Fore.RED)
            sys.exit(1)

        fn(
            *sync_types[sync_type]["args"]
        )  # NOTE: mypy might see it as an object when it's not

    if specified_repos is not None:
        for repo in specified_repos:
            sync_repo(repo, repos[repo]["sync_type"], repos[repo]["location"])
        return

    for repo_name in repos.sections():
        sync_repo(
            repo_name, repos[repo_name]["sync_type"], repos[repo_name]["location"]
        )


def rm_repos(*args) -> None:
    """Remove a repository from local synced repositories"""

    check_args(args, 1, "Required argument: repo(s)")

    for repo in args:
        repo_path = f"{get_path(CONFIG['sync']['location'])}/{repo}"

        if not os.path.exists(repo_path):
            log(f"Repository {repo_path} does not exist", "error", Fore.RED)
            sys.exit(1)

        log(f"Removing {repo_path}")
        shutil.rmtree(repo_path)


def clean_repos(*args):
    """Remove unused repositories from local FS"""

    del args

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


def search(*args):
    """Search for a file in repositories"""

    check_args(args, 3, "Required arguments: data, sort, query")

    allowed_sort_keys = ["all", "size", "speed"]
    allowed_data_keys = ["all", "name", "description", "url", "keywords", "protocol"]

    data = args[0]
    sort_key = args[1]
    query = " ".join(args[2:]).lower()

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

                with open(url_file.path, "r", encoding=FILE_ENCODING) as url_info_file:
                    """url_info is a dict of info about a specific url"""

                    url_info = json.load(url_info_file)

                    if data == "all":
                        collected_data = " ".join(
                            [str(info) for info in url_info.values()]
                        )
                    else:
                        collected_data = str(url_info[data])

                    collected_data = collected_data.lower()

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
    """Download files from repositories"""

    check_args(args, 1, "Required argument: repo@atom-name")

    for atom in args:
        repo, name = atom.split("@", 1)

        repo_path = f"{get_path(CONFIG['sync']['location'])}/{repo}"

        with open(f"{repo_path}/REPO.json", "r", encoding=FILE_ENCODING) as f:
            urls = json.load(f)["urls"]

        download_file = f"{repo_path}/{urls}/{name}.json"

        with open(download_file, "r", encoding=FILE_ENCODING) as f:
            download_info = json.load(f)

        features = features_of(download_info)
        filename = os.path.split(download_info["url"])[1]

        for masked_url in CONFIG["domain_ignore"]:
            is_masked_global = CONFIG["domain_ignore"].getboolean(
                masked_url, fallback=None
            )

            if is_masked_global is None:
                continue

            if is_masked_global and fuzzysearch.find_near_matches(
                masked_url, features["domain"]["domain"], max_l_dist=1
            ):
                log(f"Globally masked url: {masked_url}", "error", Fore.RED)
                sys.exit(1)
            elif not is_masked_global and features["domain"]["domain"] == masked_url:
                log(f"Masked url: {masked_url}", "error", Fore.RED)
                sys.exit(1)

        for masked_feature in CONFIG["mask"]:
            if masked_feature in features["features"]:
                log(
                    f"{name} ({filename}) masked by {masked_feature}", "error", Fore.RED
                )
                sys.exit(1)

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
            "torrent": {
                "fn": Download.torrent,
                "args": [download_info["url"], download_info.get("torrent_checksums")],
            },
        }

        download_protocol_info = protocols.get(download_info["protocol"])

        if download_protocol_info is None:
            log(f"Unsupported protocol: {download_info['protocol']}", "error", Fore.RED)
            sys.exit(1)

        log(f"Downloading {filename} over the {download_info['protocol']} protocol")
        time.sleep(5)  # Give users time to think if they want to actually do it
        downloaded_filename = download_protocol_info["fn"](
            *download_protocol_info["args"]
        )

        checksum_dict = download_info.get("checksums")
        checksum_dict = checksum_dict if checksum_dict else {}

        if not checksum_dict:
            log("No checksums found", "warning", Fore.YELLOW)

        for checksum_type, checksum in checksum_dict.items():
            is_good_checksum, calculated_checksum = verify_checksum(
                checksum_type, downloaded_filename, checksum
            )

            if not is_good_checksum:
                log(
                    f"{checksum_type} checksum check for {downloaded_filename} failed",
                    "error",
                    Fore.RED,
                )
                log(f"Real checksum: {checksum}")
                log(f"Got: {calculated_checksum}")

                log("Be careful when you use this file", "warning", Fore.LIGHTYELLOW_EX)

                if input(f"Remove {downloaded_filename}? [Y/n]: ").lower() != "n":
                    log(f"Removing {downloaded_filename}")
                    os.remove(filename)

                sys.exit(1)


def show_version(*args) -> None:
    """Show and print the version on DM"""

    del args
    log(__version__, "version", Fore.GREEN)


def show_info(*args) -> None:
    """Show info about an atom"""
    check_args(args, 2, "Required arguments: is_json, atom")

    is_json = args[0] == "true"

    for atom in args[1:]:
        repo, name = atom.split("@", 1)

        repo_path = f"{get_path(CONFIG['sync']['location'])}/{repo}"

        with open(f"{repo_path}/REPO.json", "r", encoding=FILE_ENCODING) as f:
            urls = json.load(f)["urls"]

        download_file = f"{repo_path}/{urls}/{name}.json"

        with open(download_file, "r", encoding=FILE_ENCODING) as file:
            print(f"\n{name}:")

            if is_json:
                for line in file:
                    print(line, end="")
                continue

            url_info = json.load(file)

            for info_name, value in url_info.items():
                log(f"{info_name} = {value}", "info", Fore.LIGHTCYAN_EX)


def atomise(*args) -> None:
    """Helps you create atoms"""

    check_args(args, 2, "Required arguments: input_file, output_atom")

    full_path_in, full_path_out = get_path(args[0]), get_path(args[1])
    if not os.path.exists(full_path_in) or os.path.exists(full_path_out):
        log(f"Invalid arguments: {full_path_in}, {full_path_out}", "error", Fore.RED)
        sys.exit(1)

    ATOM = {
        "name": input("Name: "),
        "description": input("Desctoption: "),
        "url": input("URL: "),
        "keywords": input("Keywords (seprate by ','): ").split(","),
        "protocol": input("Protocol: "),
        "speed": float(input("Speed (KbPS): ")),
        "size": os.path.getsize(full_path_in),
        "checksums": {},
    }

    for checksum_type in hashlib.algorithms_guaranteed:
        checksum = verify_checksum(checksum_type, full_path_in, "")[1]

        if checksum:
            ATOM["checksums"][checksum_type] = checksum

    with open(full_path_out, "w", encoding=FILE_ENCODING) as file:
        file.write(json.dumps(ATOM, indent=4))
        log("Configuration written", "done", Fore.LIGHTGREEN_EX)


def main() -> int:
    """Main function"""

    ACTIONS: dict = {
        "sync": {
            "desc": "Download and sync repositories",
            "func": sync,
            "args": ["repositor(y/ies)"],
        },
        "rm-repos": {
            "desc": "Remove a repository from synced repsitories",
            "func": rm_repos,
            "args": ["repositor(y/ies)"],
        },
        "clean": {
            "desc": "Clean repositories that are not in repos.ini anymore",
            "func": clean_repos,
            "args": [],
        },
        "search": {
            "desc": "Search for a fuzzy string in a specific portion of the info file, speed/size/all are the sorting keys and query is the query",
            "func": search,
            "args": [
                "name|description|url|keywords|protocol|all",
                "speed|size|all",
                "query",
            ],
        },
        "download": {
            "desc": "Download files",
            "func": download,
            "args": [
                "repo@file ...",
            ],
        },
        "version": {
            "desc": "Show version and exit",
            "func": show_version,
            "args": [],
        },
        "info": {
            "desc": "Show info about an atom",
            "func": show_info,
            "args": ["true|false", "repo@file ..."],
        },
        "atomise": {
            "desc": "A utility to help you create atoms",
            "func": atomise,
            "args": ["file", "outfile"],
        },
    }

    try:
        fn = ACTIONS[sys.argv[1]]["func"]
    except (KeyError, IndexError):
        usage(ACTIONS)
        return 1

    fn(*sys.argv[2:])

    return 0


if __name__ == "__main__":
    assert main.__annotations__.get("return") is int, "main() should return an integer"
    sys.exit(main())
