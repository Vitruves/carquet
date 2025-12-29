/**
 * @file schema.c
 * @brief Schema management
 */

#include <carquet/carquet.h>
#include "reader/reader_internal.h"
#include "thrift/parquet_types.h"
#include "core/arena.h"
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Schema Creation
 * ============================================================================
 */

carquet_schema_t* carquet_schema_create(carquet_error_t* error) {
    carquet_schema_t* schema = calloc(1, sizeof(carquet_schema_t));
    if (!schema) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate schema");
        return NULL;
    }

    if (carquet_arena_init_size(&schema->arena, 4096) != CARQUET_OK) {
        free(schema);
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate schema arena");
        return NULL;
    }

    /* Allocate initial arrays (capacity 16) */
    schema->num_elements = 1;  /* Root element */
    schema->elements = carquet_arena_calloc(&schema->arena, 16, sizeof(parquet_schema_element_t));
    if (!schema->elements) {
        carquet_arena_destroy(&schema->arena);
        free(schema);
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate schema elements");
        return NULL;
    }

    /* Initialize root element */
    schema->elements[0].name = carquet_arena_strdup(&schema->arena, "schema");
    schema->elements[0].num_children = 0;

    /* Allocate leaf tracking arrays */
    schema->leaf_indices = carquet_arena_calloc(&schema->arena, 16, sizeof(int32_t));
    schema->max_def_levels = carquet_arena_calloc(&schema->arena, 16, sizeof(int16_t));
    schema->max_rep_levels = carquet_arena_calloc(&schema->arena, 16, sizeof(int16_t));
    schema->num_leaves = 0;

    return schema;
}

void carquet_schema_free(carquet_schema_t* schema) {
    if (schema) {
        carquet_arena_destroy(&schema->arena);
        free(schema);
    }
}

/* ============================================================================
 * Schema Building
 * ============================================================================
 */

carquet_status_t carquet_schema_add_column(
    carquet_schema_t* schema,
    const char* name,
    carquet_physical_type_t physical_type,
    const carquet_logical_type_t* logical_type,
    carquet_field_repetition_t repetition,
    int32_t type_length) {

    /* schema and name are nonnull per API contract */
    /* Add element to schema */
    int32_t elem_idx = schema->num_elements;
    parquet_schema_element_t* elem = &schema->elements[elem_idx];
    memset(elem, 0, sizeof(*elem));

    elem->name = carquet_arena_strdup(&schema->arena, name);
    elem->has_type = true;
    elem->type = physical_type;
    elem->has_repetition = true;
    elem->repetition_type = repetition;
    elem->type_length = type_length;

    if (logical_type) {
        elem->has_logical_type = true;
        elem->logical_type = *logical_type;
    }

    schema->num_elements++;
    schema->elements[0].num_children++;

    /* Track as leaf */
    schema->leaf_indices[schema->num_leaves] = elem_idx;
    schema->max_def_levels[schema->num_leaves] = (repetition == CARQUET_REPETITION_OPTIONAL) ? 1 : 0;
    schema->max_rep_levels[schema->num_leaves] = (repetition == CARQUET_REPETITION_REPEATED) ? 1 : 0;
    schema->num_leaves++;

    return CARQUET_OK;
}

int32_t carquet_schema_add_group(
    carquet_schema_t* schema,
    const char* name,
    carquet_field_repetition_t repetition,
    int32_t parent_index) {

    /* schema and name are nonnull per API contract */
    /* For now, only support adding to root */
    if (parent_index != -1 && parent_index != 0) {
        return -1;
    }

    int32_t elem_idx = schema->num_elements;
    parquet_schema_element_t* elem = &schema->elements[elem_idx];
    memset(elem, 0, sizeof(*elem));

    elem->name = carquet_arena_strdup(&schema->arena, name);
    elem->has_type = false;  /* Groups don't have a type */
    elem->has_repetition = true;
    elem->repetition_type = repetition;
    elem->num_children = 0;

    schema->num_elements++;
    schema->elements[0].num_children++;

    return elem_idx;
}

/* ============================================================================
 * Schema Queries
 * ============================================================================
 */

int32_t carquet_schema_find_column(
    const carquet_schema_t* schema,
    const char* name) {

    /* schema and name are nonnull per API contract */
    /* Simple linear search */
    for (int32_t i = 0; i < schema->num_leaves; i++) {
        int32_t elem_idx = schema->leaf_indices[i];
        if (schema->elements[elem_idx].name &&
            strcmp(schema->elements[elem_idx].name, name) == 0) {
            return i;
        }
    }

    return -1;
}

int32_t carquet_schema_num_columns(const carquet_schema_t* schema) {
    /* schema is nonnull per API contract */
    return schema->num_leaves;
}

int32_t carquet_schema_num_elements(const carquet_schema_t* schema) {
    /* schema is nonnull per API contract */
    return schema->num_elements;
}

const carquet_schema_node_t* carquet_schema_get_element(
    const carquet_schema_t* schema,
    int32_t index) {

    /* schema is nonnull per API contract */
    if (index < 0 || index >= schema->num_elements) {
        return NULL;
    }

    /* Return pointer to element (cast as schema_node) */
    return (const carquet_schema_node_t*)&schema->elements[index];
}

/* ============================================================================
 * Schema Node Accessors
 * ============================================================================
 */

const char* carquet_schema_node_name(const carquet_schema_node_t* node) {
    /* node is nonnull per API contract */
    const parquet_schema_element_t* elem = (const parquet_schema_element_t*)node;
    return elem->name;
}

bool carquet_schema_node_is_leaf(const carquet_schema_node_t* node) {
    /* node is nonnull per API contract */
    const parquet_schema_element_t* elem = (const parquet_schema_element_t*)node;
    return elem->has_type;
}

carquet_physical_type_t carquet_schema_node_physical_type(const carquet_schema_node_t* node) {
    /* node is nonnull per API contract */
    const parquet_schema_element_t* elem = (const parquet_schema_element_t*)node;
    return elem->type;
}

const carquet_logical_type_t* carquet_schema_node_logical_type(const carquet_schema_node_t* node) {
    /* node is nonnull per API contract */
    const parquet_schema_element_t* elem = (const parquet_schema_element_t*)node;
    return elem->has_logical_type ? &elem->logical_type : NULL;
}

carquet_field_repetition_t carquet_schema_node_repetition(const carquet_schema_node_t* node) {
    /* node is nonnull per API contract */
    const parquet_schema_element_t* elem = (const parquet_schema_element_t*)node;
    return elem->repetition_type;
}

int16_t carquet_schema_node_max_def_level(const carquet_schema_node_t* node) {
    /* node is nonnull per API contract */
    const parquet_schema_element_t* elem = (const parquet_schema_element_t*)node;
    return (elem->repetition_type == CARQUET_REPETITION_OPTIONAL) ? 1 : 0;
}

int16_t carquet_schema_node_max_rep_level(const carquet_schema_node_t* node) {
    /* node is nonnull per API contract */
    const parquet_schema_element_t* elem = (const parquet_schema_element_t*)node;
    return (elem->repetition_type == CARQUET_REPETITION_REPEATED) ? 1 : 0;
}
