#pragma once

/* Defines implicit conversion operator for specified type
 * The class must provide the get() method
 *
 * Main use - implicit dereference operation for classes derived from std::unique_ptr
 */
#define UTILS_DEREF(type) \
    operator type() \
    { \
        return this->get(); \
    }
