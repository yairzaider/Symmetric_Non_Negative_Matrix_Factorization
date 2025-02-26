GCC = gcc
ALLCFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors
SRC_FILE = symnmf.c

all: symnmf

symnmf: symnmf.o
	$(GCC) $(ALLCFLAGS) symnmf.o -o symnmf -lm

symnmf.o: $(SRC_FILE)
	$(GCC) -c $(SRC_FILE) $(ALLCFLAGS)

clean:
	rm -f symnmf symnmf.o
