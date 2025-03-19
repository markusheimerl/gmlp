CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lopenblas -lm -flto

all: mlp.out gmlp.out

gmlp.out: gmlp.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

run: gmlp.out
	@time ./gmlp.out

clean:
	rm -f *.out *.csv *.bin
