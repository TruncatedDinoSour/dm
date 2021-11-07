# Local repositories

These types of repositories live as a folder on your machine so they are not public.
They also work with no internet connection, but the downside is that you're the only maintainer.
Here's how to add one:

1. Add this to `repos.ini`

```ini
[MY_REPO_NAME]
location = /LOCATION/TO/LOCAL/REPO
sync_type = local
```

Make sure you fill in everything correctly, for example:

```ini
[test_local_repo]
location = ~/my_repository
sync_type = local
```

2. Sync them
```bash
$ dm sync
```

These types of repos are fast, require no internet connection, but are maintained by you only.

