CONF ?= $(HOME)/.config/dm
PREFIX ?= /usr/local
BINDIR ?= $(PREFIX)/bin
SHARE ?= $(HOME)/.local/share/dm
SU ?= sudo

deps:
	python3 -m pip install -r requirements.txt

install:
	mkdir -p $(DESTDIR)$(BINDIR)
	install -Dm755 dm $(DESTDIR)$(BINDIR)

uninstall:
	rm -f $(DESTDIR)$(BINDIR)/dm

unshare:
	rm -rfv "$(SHARE)"

dm-config:
	mkdir -p $(DESTDIR)$(CONF)
	cp -vi config/* $(DESTDIR)$(CONF)

dm-unconfig:
	rm -rfvi $(DESTDIR)$(CONF)

dm-setup:
	-dm rm-repo main
	dm sync


dm-full-setup:
	$(SU) make uninstall
	$(SU) make install
	make dm-config
	make dm-setup
	dm clean

dm-update:
	$(SU) make uninstall
	$(SU) make install
	make dm-setup
	dm clean

dm-purge:
	$(SU) make uninstall
	make dm-unconfig
	make unshare
