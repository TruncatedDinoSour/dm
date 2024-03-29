# Creation of git repositories

1. Make a new github repository

This is the easy part, just create a new github repository.

2. Add a file called `REPO.json`

This file will define how your URLs and stuff are laid out, the required keys are:

- `urls` \-\- defines where your URLs are located

Template for `REPO.json`:

```json
{
    "urls": "SOME_FOLDER"
}
```

For example:


```json
{
    "urls": "URLs"
}
```

3. Create the URLs directory

Create a new folder with the same name as the value of `urls` in `REPO.json` file, in my case
it would be `URLs`.

4. Add URLs

This is the hard part, this is the template for URLs json:

```json
{
    "name": "SOME NAME OF THE URL",
    "description": "SOME DESCRIPTION",
    "url": "HTTPS://SOME.SERVER.COM/PATH/TO/FILE.HEY",
    "keywords": ["SOME", "KEYWORDS", "TO", "HELP"],
    "protocol": "THE_URL_PROTOCOL",
    "speed": SPEED_IN_KBPS,
    "size": SIZE_IN_B,
    "checksums": {
        "md5": MD%_SUJM,
        "sha512": SHA512_SUM,
        "sha256": SHA256_SUM
    }
}
```

For example

```json
{
  "name": "canonical ubuntu 21.10",
  "description": "Ubuntu linux 21.10 release by canonical",
  "url": "https://releases.ubuntu.com/21.10/ubuntu-21.10-desktop-amd64.iso",
  "keywords": ["ubuntu2010", "ubuntu-desktop", "ubuntureleases", "ubuntu"],
  "protocol": "https",
  "speed": 100,
  "size": 3116482560,
  "checksums": {
        "sha256": "f8d3ab0faeaecb5d26628ae1aa21c9a13e0a242c381aa08157db8624d574b830"
    }
}
```

More protocols will be added soon, like torrents.

5. Done! You can use the main repo as the example

# Local repositories

The same applies to local ones, except that there is no git repository creation.

