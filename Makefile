PREFIX ?= $(HOME)/.local
BINDIR = $(PREFIX)/bin

.PHONY: build install uninstall clean

build:
	cargo build --release

install: build
	mkdir -p $(BINDIR)
	cp target/release/npusearch $(BINDIR)/npusearch
	@echo ""
	@echo "Installed npusearch to $(BINDIR)/npusearch"
	@echo "Run 'npusearch doctor' to verify setup."

uninstall:
	rm -f $(BINDIR)/npusearch

clean:
	cargo clean
