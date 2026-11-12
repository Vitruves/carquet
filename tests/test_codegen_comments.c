/**
 * @file test_codegen_comments.c
 * @brief carquet codegen keeps column names inside the comments it writes
 *
 * Column names come from the file footer and are echoed into block comments of
 * the generated reader. A name holding comment punctuation must stay a comment.
 * The input path is also written into a string literal (DEFAULT_FILE); a path
 * holding a double quote must stay inside that literal.
 */

#include <carquet/carquet.h>
#include "cli/cli.h"
#include "test_helpers.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static char* slurp(FILE* f) {
    long n;
    if (fseek(f, 0, SEEK_END) != 0 || (n = ftell(f)) < 0 || fseek(f, 0, SEEK_SET) != 0) return NULL;
    char* s = malloc((size_t)n + 1);
    if (!s) return NULL;
    size_t got = fread(s, 1, (size_t)n, f);
    s[got] = '\0';
    return s;
}

/* Blank out comments in place, as a C compiler would see the file. String and
 * character literals are skipped so punctuation inside them is not misread. */
static void blank_comments(char* s) {
    for (char* p = s; *p; ) {
        if (*p == '"' || *p == '\'') {
            char q = *p++;
            while (*p && *p != q) { if (*p == '\\' && p[1]) p++; p++; }
            if (*p) p++;
        } else if (p[0] == '/' && p[1] == '*') {
            while (*p && !(p[0] == '*' && p[1] == '/')) *p++ = ' ';
            if (*p) { *p++ = ' '; *p++ = ' '; }
        } else if (p[0] == '/' && p[1] == '/') {
            while (*p && *p != '\n') *p++ = ' ';
        } else {
            p++;
        }
    }
}

/* Blank out the contents of string and character literals in place, leaving
 * the quotes. Run after blank_comments() so quotes in comments are gone. */
static void blank_literals(char* s) {
    for (char* p = s; *p; ) {
        if (*p == '"' || *p == '\'') {
            char q = *p++;
            while (*p && *p != q) {
                if (*p == '\\' && p[1]) *p++ = ' ';
                *p++ = ' ';
            }
            if (*p) p++;
        } else {
            p++;
        }
    }
}

/* Decode the body of the C string literal starting just after the opening
 * quote at `p` (simple and octal escapes only, as codegen emits). Returns the
 * number of bytes written, or -1 on a malformed / unterminated literal. */
static int decode_literal(const char* p, char* out, size_t out_size) {
    size_t j = 0;
    while (*p && *p != '"') {
        if (j + 1 >= out_size) return -1;
        if (*p != '\\') { out[j++] = *p++; continue; }
        p++;
        if (*p >= '0' && *p <= '7') {
            int v = 0;
            for (int k = 0; k < 3 && *p >= '0' && *p <= '7'; k++) v = v * 8 + (*p++ - '0');
            out[j++] = (char)v;
        } else if (*p == '\\' || *p == '"' || *p == '?' || *p == '\'') {
            out[j++] = *p++;
        } else {
            return -1;
        }
    }
    if (*p != '"') return -1;
    out[j] = '\0';
    return (int)j;
}

/* Write a one-column file whose column is called `column`, run the reader
 * generator on it as `input_path`, and return the generated source. */
static char* generate_reader(const char* file, const char* column, const char* input_path) {
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_schema_t* schema = carquet_schema_create(&err);
    if (!schema) return NULL;
    if (carquet_schema_add_column(schema, column, CARQUET_PHYSICAL_INT32, NULL,
                                  CARQUET_REPETITION_REQUIRED, 0, 0) != CARQUET_OK) {
        carquet_schema_free(schema);
        return NULL;
    }
    carquet_writer_options_t wopts;
    carquet_writer_options_init(&wopts);
    carquet_writer_t* w = carquet_writer_create(file, schema, &wopts, &err);
    if (!w) { carquet_schema_free(schema); return NULL; }
    int32_t vals[4] = {1, 2, 3, 4};
    carquet_status_t st = carquet_writer_write_batch(w, 0, vals, 4, NULL, NULL);
    st = st == CARQUET_OK ? carquet_writer_close(w) : (carquet_writer_abort(w), st);
    carquet_schema_free(schema);
    if (st != CARQUET_OK) return NULL;

    char* src = NULL;
    carquet_reader_t* reader = carquet_reader_open(file, NULL, &err);
    FILE* out = tmpfile();
    if (reader && out) {
        codegen_opts_t opts;
        memset(&opts, 0, sizeof(opts));
        opts.input_path = input_path;
        opts.batch_size = 1024;
        codegen_hints_t hints;
        memset(&hints, 0, sizeof(hints));
        if (cmd_codegen_read(out, reader, &opts, &hints) == 0) src = slurp(out);
    }
    if (out) fclose(out);
    if (reader) carquet_reader_close(reader);
    return src;
}

static int test_codegen_comment_injection(void) {
    char path[512];
    carquet_test_temp_path(path, sizeof(path), "codegen_names");

    /* The name closes the schema comment and reopens one after its payload.
     * '=' and ';' never survive into an identifier, so "= 1337;" outside a
     * comment can only be the payload being compiled. */
    const char* evil = "x */ int marker = 1337; /* y";
    char* src = generate_reader(path, evil, path);
    remove(path);
    if (!src) TEST_FAIL("codegen_comment_injection", "generation failed");

    blank_comments(src);
    bool injected = strstr(src, "= 1337;") != NULL;
    free(src);
    if (injected)
        TEST_FAIL("codegen_comment_injection", "column name escaped the generated comment");

    TEST_PASS("codegen_comment_injection");
    return 0;
}

static int test_codegen_default_file_literal(void) {
    char path[512];
    carquet_test_temp_path(path, sizeof(path), "codegen_path");

    /* The path does not exist, so codegen uses it verbatim rather than a
     * realpath() result. It closes the DEFAULT_FILE literal, declares the
     * marker, and reopens a literal to swallow the closing quote. It also
     * carries a backslash, a would-be trigraph, a newline, and a control byte
     * followed by a digit (which a variable-width octal escape would eat). */
    const char* evil_path =
        "/nonexistent/x\"; int marker = 1337; char* y = \" \\ ?\?/ a\nb \0017";
    char* src = generate_reader(path, "value", evil_path);
    remove(path);
    if (!src) TEST_FAIL("codegen_default_file_literal", "generation failed");

    /* The literal must decode back to exactly the input path. */
    const char* def = strstr(src, "#define DEFAULT_FILE \"");
    char decoded[512];
    int n = def ? decode_literal(def + strlen("#define DEFAULT_FILE \""),
                                 decoded, sizeof(decoded)) : -1;
    if (n < 0 || strcmp(decoded, evil_path) != 0) {
        free(src);
        TEST_FAIL("codegen_default_file_literal",
                  "DEFAULT_FILE does not round-trip the input path");
    }

    blank_comments(src);
    blank_literals(src);
    bool injected = strstr(src, "= 1337;") != NULL;
    free(src);
    if (injected)
        TEST_FAIL("codegen_default_file_literal",
                  "input path escaped the DEFAULT_FILE string literal");

    TEST_PASS("codegen_default_file_literal");
    return 0;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int failures = 0;
    failures += test_codegen_comment_injection();
    failures += test_codegen_default_file_literal();
    if (failures) {
        printf("%d test(s) failed\n", failures);
        return 1;
    }
    printf("All tests passed\n");
    return 0;
}
