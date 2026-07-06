#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>

extern int _end;

void* _sbrk(int incr)
{
    static unsigned char* heap_end = 0;
    unsigned char* prev_heap_end;

    if (heap_end == 0)
    {
        heap_end = (unsigned char*)&_end;
    }
    prev_heap_end = heap_end;
    heap_end += incr;
    return prev_heap_end;
}

int _close(int fd)
{
    (void)fd;
    return -1;
}

int _fstat(int fd, struct stat* st)
{
    (void)fd;
    st->st_mode = S_IFCHR;
    return 0;
}

int _isatty(int fd)
{
    (void)fd;
    return 1;
}

int _lseek(int fd, int ptr, int dir)
{
    (void)fd;
    (void)ptr;
    (void)dir;
    return 0;
}

int _read(int fd, char* ptr, int len)
{
    (void)fd;
    (void)ptr;
    (void)len;
    return 0;
}

int _write(int fd, char* ptr, int len)
{
    (void)fd;
    (void)ptr;
  return len;
}

void _exit(int code)
{
    (void)code;
    for (;;)
    {
    }
}

void _kill(int pid, int sig)
{
    (void)pid;
    (void)sig;
    for (;;)
    {
    }
}

int _getpid(void)
{
    return 1;
}
