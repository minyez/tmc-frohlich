PROJNAME = tmc-frohlich
OBJS = tmc-frohlich.py emc.py README.md Makefile
VERSION = $(shell \
		  awk '/__version__/ {printf("%s", $$3)}' $(PROJNAME).py | sed -e 's/"//g' \
		  )

dist: $(OBJS)
	mkdir -p $(PROJNAME)-v$(VERSION)
	rsync -a $(OBJS) $(PROJNAME)-v$(VERSION)/
	tar -zcf $(PROJNAME)-v$(VERSION).tar.gz $(PROJNAME)-v$(VERSION)
	rm -rf $(PROJNAME)-v$(VERSION)
