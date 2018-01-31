CC = mpiicc
CFLAGS = -std=c99 -O3
2dbox: 2dbox.o
	$(CC) $(CFLAGS) -o 2dbox 2dbox.o
2dboxo: 2dbox.c
	$(CC) $(CFLAGS) -c 2dbox.c
clean:
	rm 2dbox *.o

