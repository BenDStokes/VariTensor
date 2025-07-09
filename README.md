# *VariTensor*
Ergonomic tensors with variance

<a href="#sparkles-features"> Features </a> | <a href="#%EF%B8%8F-installation"> Installation </a> | <a href="#arrow_forward-usage"> Usage </a> | <a href="#%EF%B8%8F-performance"> Performance </a> | <a href="#scroll-license"> Licence </a>

## :sparkles: Features

VariTensor provides a beautiful, natural interface for full-featured, runtime-dynamic tensors.

- All common tensor operations, implemented with lazy evaluation
- Compatability with std::ranges
- Highly flexible indexing, slicing and iteration
- Implicit summation and contraction
- Full support for contravariant and covariant indices
- Index raising and lowering with metric tensors
- Pre-defined, n-dimensional Kronecker Delta and Levi-Civita Symbols
- Geometric pretty-printing of tensors up to rank 4

## 🛠️ Installation

### Direct Download

VariTensor is a **header-only** library, so to install it directly, you can download the project's source code and add "varitensor/include" to your includes.

## :arrow_forward: Usage

### Basic Functionality

```c++
#include <varitensor/Tensor.h>
#include <numeric>

using namespace varitensor;

int main() {
    // VatiTensor uses Index objects to define tensors and understand their operations
    Index mu{GREEK}, nu{5};

    // Tensors can be constructed from any number of indices
    Tensor T{mu}; // Indices default to covarient
    Tensor U{
        {mu, CONTRAVARIANT},
        {nu, COVARIANT}
    };
    // The alternative aliases UPPER and LOWER can also be used for index variance

    // Tensors are forward iterable so are compatible with many STL algorithms
    std::ranges::iota(T, 0);
    std::ranges::iota(U, 0);

    // VariTensor uses operator[] to index tensors
    // We could omit the indices here as they are the same we declared with, but they aid readability
    Tensor V = T[mu] * U[mu, nu];
}
```

> :warning: **CAUTION**
> 
> It is generally recommended to not use auto when assigning tensor expressions, as the complier will deduce the type as one of the library's proxy objects and delay resolving of the expression. A common exception to this is keeping a view from an indexing operation.
> ```c++
> Tensor T = U + V; // Prefered
> auto T = U + V; // Delays evaluation until T is converted to varitensor::Tensor
> auto view = T[j, k] // Gives a view object that can be iterated over
> ```

### Pretty Printing

```c++
#include <varitensor/Tensor.h>
#include <varitensor/pretty_print.h>
#include <numeric>

using namespace varitensor;

int main() {
    // Named indices of the same size always compare equal and keep their names when printing
    Index i{"i", LATIN}, j{"j", LATIN}, k{"k", LATIN};

    // VariTensor supports a wide range of punctuation, e.g. '
    Tensor T_primed{"T'", {
        {i, CONTRAVARIANT},
        {j, CONTRAVARIANT},
        {k, COVARIANT}
    }
    };
    // We could also use T.set_name() after initialisation; this is useful for naming expressions

    std::ranges::iota(T_primed, 0);
    std::cout << T_primed;
}
```
Result:
```
 _____ ,-, 
|_   _||/  i j   
  | |      
  | |             =
  |_|          k 
Rank 3 Tensor

 /                                               \
|                                 18.0 19.0 20.0  |
|                  9.0 10.0 11.0  21.0 22.0 23.0  |
|  0.0  1.0  2.0  12.0 13.0 14.0  24.0 25.0 26.0  |
|  3.0  4.0  5.0  15.0 16.0 17.0                  |
|  6.0  7.0  8.0                                  |
 \                                               /

```

> :information_source: **Note**
>
> By default, all tensors are named "VariTensor" and all indices are named "idxN" where N is the underlying index ID.

> :information_source: **Note**
>
> The precision of the display and the width of the fields can be set with varitensor::set_print_precision() and varitensor::set_print_data_width(). If a data element exceeds the field width, it will be truncated with a ~ symbol.

<h3> Flexible Indexing </h3>

```c++
#include <varitensor/Tensor.h>
#include <numeric>

using namespace varitensor;

int main() {
    // VariTensor's interface can perform a variety of indexing operations
    Index i{"i", LATIN};
    Index mu{"mu", GREEK}, nu{"nu", GREEK};

    Tensor T{mu, nu};
    std::ranges::iota(T, 1);

    T[1, 0] = 10; // Set a particular value of T
 
    Tensor U = T[mu, 0]; // U is the first column of T
    Tensor V = U[i]; // V is the first 3 elements of U 

    // You can create intervals on idices by passing the start and finish to ()
    // If only the start is passed, the end is taken to be the index's maximum value
    Tensor W = U[mu(1, 2)]; // W is the middle 2 elements of U
    Tensor X = U[mu(1)]; // X is the last 3 elements of U

    T[mu(1), 1] = X; // assign a slice of T to have the values of X

    // Finally, contract T to a scalar
    Tensor Y = T[mu, mu];
    double z{Y}; // Scalar tensors are convertable to double
}
```

> :information_source: **Note**
>
> The indexing operator, [ ], returns a View object that can be used to iterate over tensors or initialise new ones.

### Metric Tensor

```c++
#include <varitensor/Tensor.h>

using namespace varitensor;

// A metric tensor can be created with a user-defined function
// The signature must be double(int, int) where the ints give the position in the metric
// varitensor::MetricFunction is provided as a shorthand
MetricFunction my_minkowski_metric = [](int i, int j) -> double {
    if (i==j) return i == 3 ? -1 : 1;
    return 0;
};

int main() {
    Index i{"i", LATIN}, j{"j", LATIN}, k{"k", LATIN};

    Tensor T{i, j};
    Tensor g = metric_tensor({
            {i, CONTRAVARIANT},
            {k, CONTRAVARIANT}
        },
        my_minkowski_metric // The default is varitensor::EUCLIDEAN_METRIC
    );

    // Since g is a metric tensor, k will positionally replace i in U
    Tensor U = T[i, j] * g[i, k]; // gives U[k, j] with contravariant k
}
```
### Other Features

```c++
// Reference iteration
for (auto& value: T) {...}
for (auto& value: T[mu, 0]) {...}

// Range-based iteration exposes more information about position
for (auto iter=T.begin(), end=T.end(); iter!=end; ++iter) {
        std::cout << iter.positions[mu];
        std::cout << iter.positions[3];
}

// Pre-defined mathematical symbols
varitensor::kronecker_delta(...);
varitensor::levi_civita_symbol(...);

// Numerous information and manipulation functions
T.transpose(i, j);
T.relable(i, j);
Index first_index = T.indices(0);
// ...and many more!
```

## ⏱️ Performance

VariTensor generally performs towards the slower end of Tensor libraries. At this time, this library is more oriented towards usability than speed, though I hope to improve this in the future.

> :information_source: **Note**
>
> By default, VariTensor validates all tensor operations at runtime and throws if any of them are mathematically ill-formed. For increased performance, these checks can be toggled off at compile time by compiling with:
> ```
> -DVARITENSOR_VALIDATION_ON=0
> ```
>Or, in cmake:
> ```cmake
> set(VARITENSOR_VALIDATION_ON 0)
> ```

## :scroll: License

VariTensor is licenced under the [Mozilla Public Licence v2.0](https://www.mozilla.org/en-US/MPL/2.0/).

<br/>

*Copyright &copy; Ben Stokes, 2025*
