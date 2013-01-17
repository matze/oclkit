#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.68])
AC_INIT([oclkit], [0.1], [foo@bar.com])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_SRCDIR([.])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_FILES([Makefile])

AM_INIT_AUTOMAKE(oclkit, 0.1)
AM_PROG_CC_C_O
AM_SILENT_RULES([yes])

# Checks for libraries.
LT_INIT

PKG_CHECK_MODULES([GLIB], [glib-2.0 >= 2.20.0])

# Checks for header files.
AC_CHECK_HEADERS([stdlib.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_TYPE_SIZE_T

# Checks for library functions.
AC_FUNC_MALLOC

# Check for OpenCL
AX_CHECK_CL

if test "X$no_cl" = "Xyes"; then
    AC_MSG_FAILURE([You need OpenCL])
fi

AC_OUTPUT
