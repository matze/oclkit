ACLOCAL_AMFLAGS = -I m4
AM_CFLAGS = -std=c99 -pedantic $(CL_CFLAGS)

noinst_LTLIBRARIES = libocl.la
libocl_la_SOURCES = ocl.c ocl.h
libocl_la_LIBADD = $(CL_LIBS)

bin_PROGRAMS = info leak ptransfer regr

info_SOURCES = info.c
info_CFLAGS = $(AM_CFLAGS)
info_LDADD = libocl.la

leak_SOURCES = leak.c
leak_CFLAGS = $(AM_CFLAGS)
leak_LDADD = libocl.la

ptransfer_SOURCES = ptransfer.c
ptransfer_CFLAGS = $(GLIB_CFLAGS) $(AM_CFLAGS)
ptransfer_LDADD = libocl.la $(GLIB_LIBS)

regr_SOURCES = regr.c
regr_CFLAGS = $(GLIB_CFLAGS) $(AM_CFLAGS)
regr_LDADD = libocl.la $(GLIB_LIBS)
