# Main Makefile

# Detect the operating system
UNAME := $(shell uname -s)

# Include the appropriate platform-specific Makefile
ifeq ($(UNAME), Linux)
    include Makefile.linux
else
    include Makefile.windows
endif