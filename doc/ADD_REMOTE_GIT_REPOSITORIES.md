# Remote git repositories

These repositories live on github and are synchronised though the internet.
They are public so anyone can contribute.
To add a new remote repository:

1. Add this to `repos.ini`

```ini
[MY_REPO_NAME]
location = https://github.com/SOME_USER/SOME-REPO
sync_type = git
```

Make sure you fill in everything correctly, for example:

```ini
[test_repo]
location = https://github.com/NotTruncatedDinosour/my_dm_repo
sync_type = git
```

2. Sync them
```bash
$ dm sync
```

Note that these repositories require an internet connection to be synced.

