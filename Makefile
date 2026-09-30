PROJECT := kilix-yolox
COMMAND_NAME := kilix-yolox
BUILD_DIR ?= build
PREFIX ?= /usr/local
DESTDIR ?=

CC ?= cc
AR ?= ar
INSTALL ?= install

CPPFLAGS += -D_POSIX_C_SOURCE=200809L -Iinclude
WARNINGS := \
	-Wall -Wextra -Wpedantic -Wconversion -Wshadow \
	-Wstrict-prototypes -Wmissing-prototypes -Wformat=2
CFLAGS ?= -O2 -g
override CFLAGS += -std=c11 -fPIC $(WARNINGS)
LDLIBS := -lm

# The self-test carries one fixture inside the binary, so it can run
# anywhere the binary can, with no path to get wrong.
SELFTEST_FIXTURE := tests/fixtures/raw_320.txt
FIXTURE_DIR := $(abspath tests/fixtures)

LIB_OBJECTS := $(BUILD_DIR)/kilix_yolox.o
STATIC_LIB := $(BUILD_DIR)/lib$(PROJECT).a
CMD_OBJECTS := $(BUILD_DIR)/main.o $(BUILD_DIR)/kyx_fixture.o
COMMAND := $(BUILD_DIR)/$(COMMAND_NAME)

TESTS := $(BUILD_DIR)/test-letterbox $(BUILD_DIR)/test-decode \
	$(BUILD_DIR)/test-nms $(BUILD_DIR)/test-reply

TOOLS := tools/kilix-yolox-detect tools/kilix-yolox-fetch tools/kilix-yolox-cut

.DEFAULT_GOAL := all
.PHONY: all test sanitize install clean

all: $(COMMAND) $(STATIC_LIB)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/selftest_fixture.h: $(SELFTEST_FIXTURE) | $(BUILD_DIR)
	awk 'BEGIN { print "static const char *const kyx_selftest_lines[] = {" } \
	     { gsub(/\\/, "\\\\"); gsub(/"/, "\\\""); print "    \"" $$0 "\\n\"," } \
	     END { print "    (const char *)0 };" }' $< > $@

$(BUILD_DIR)/%.o: src/%.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -Isrc -I$(BUILD_DIR) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/main.o: $(BUILD_DIR)/selftest_fixture.h

$(STATIC_LIB): $(LIB_OBJECTS)
	$(AR) rcs $@ $^

$(COMMAND): $(CMD_OBJECTS) $(STATIC_LIB) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

# $< and the objects by name, not $^: the recorded dependencies add the
# headers to the prerequisites, and a second `make test` after a header
# edit would hand them to the compiler as inputs - which gcc quietly
# accepts and clang refuses.  The same thing bit the sibling.
$(BUILD_DIR)/test-%: tests/test_%.c $(BUILD_DIR)/kyx_fixture.o $(STATIC_LIB) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -Isrc -Itests -DKYX_FIXTURE_DIR='"$(FIXTURE_DIR)"' \
		$(CFLAGS) $(LDFLAGS) -MMD -MP $< $(BUILD_DIR)/kyx_fixture.o \
		$(STATIC_LIB) $(LDLIBS) -o $@

test: $(TESTS) $(COMMAND)
	python3 -B -m unittest discover -s tests -p 'test_*.py' -v
	@set -e; for binary in $(TESTS); do \
		printf '\n== %s ==\n' "$$binary"; \
		"$$binary"; \
	done; \
	printf '\n== %s --selftest ==\n' "$(COMMAND)"; \
	$(COMMAND) --selftest; \
	for fixture in tests/fixtures/raw_320.txt tests/fixtures/flat_640.txt; do \
		printf '\n== %s decode --check %s ==\n' "$(COMMAND)" "$$fixture"; \
		$(COMMAND) decode --check "$$fixture" > /dev/null; \
	done; \
	printf '\nall test suites passed\n'

sanitize: CFLAGS += -fsanitize=address,undefined -fno-omit-frame-pointer
sanitize: LDFLAGS += -fsanitize=address,undefined
sanitize: clean
	@$(MAKE) --no-print-directory CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)" test

install: all
	$(INSTALL) -d $(DESTDIR)$(PREFIX)/bin
	$(INSTALL) -m 755 $(COMMAND) $(DESTDIR)$(PREFIX)/bin/
	$(INSTALL) -m 755 $(TOOLS) $(DESTDIR)$(PREFIX)/bin/
	$(INSTALL) -d $(DESTDIR)$(PREFIX)/include
	$(INSTALL) -m 644 include/kilix_yolox.h $(DESTDIR)$(PREFIX)/include/
	$(INSTALL) -d $(DESTDIR)$(PREFIX)/lib
	$(INSTALL) -m 644 $(STATIC_LIB) $(DESTDIR)$(PREFIX)/lib/
	$(INSTALL) -d $(DESTDIR)$(PREFIX)/share/$(PROJECT)
	$(INSTALL) -m 644 models/SHA256SUMS models/PROVENANCE.md \
		$(DESTDIR)$(PREFIX)/share/$(PROJECT)/

clean:
	rm -rf $(BUILD_DIR)

-include $(CMD_OBJECTS:.o=.d) $(LIB_OBJECTS:.o=.d) $(TESTS:=.d)
