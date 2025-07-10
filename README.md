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

VariTensor is a **header-only** library, so to install it directly, you can download the project's source code and add "VariTensor/include" to your includes.

## :arrow_forward: Usage

### Basic Functionality

```c++
#include <numeric>
#include <varitensor/Tensor.h>

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

    // Tensors are forward iterable so are compatible with many STL algorithms
    std::ranges::iota(T, 0);
    std::ranges::iota(U, 0);

    // VariTensor uses operator[] to index tensors
    // We could omit the indices as they are the same that we declared with, but they add to readability
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
#include <numeric>
#include <varitensor/Tensor.h>
#include <varitensor/pretty_print.h>

using namespace varitensor;

int main() {
    Index i{"i", LATIN}, j{"j", LATIN}, k{"k", LATIN};

    // VariTensor supports a wide range of punctuation, e.g. '
    Tensor T_primed{"T'", {
            {i, CONTRAVARIANT},
            {j, CONTRAVARIANT},
            {k, COVARIANT}
        }
    };
    // We could also use T.set_name() after initialisation; this is useful for naming expressions

    std::ranges::iota(T, 0);
    std::cout << T;
}
```

```
 _____ ,-, 
|_   _||/  i j   
  | |      
  | |             =
  |_|          k 
Rank 3 Tensor

 /                                                        \
|                                       18.00 19.00 20.00  |
|                     9.00 10.00 11.00  21.00 22.00 23.00  |
|  0.00  1.00  2.00  12.00 13.00 14.00  24.00 25.00 26.00  |
|  3.00  4.00  5.00  15.00 16.00 17.00                     |
|  6.00  7.00  8.00                                        |
 \                                                        /

```

> :information_source: **Note**
>
> By default, all tensors are named "VariTensor" and all indices are named "idxN" where N is the underlying index ID.

> :information_source: **Note**
>
> The precision of the display and the width of the fields can be set with varitensor::set_print_precision() and varitensor::set_print_data_width(). If a data element exceeds the field width, it will be truncated with a ~ symbol.

<h3> Flexible Indexing </h3>

```c++
#include <numeric>
#include <varitensor/Tensor.h>

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
    double z = Y; // Scalar tensors are convertable to double
}
```

> :information_source: **Note**
>
> The indexing operator, [ ], returns a View object that can be used to iterate over tensors or construct new ones.

### Metric Tensor

```c++
#include <varitensor/Tensor.h>

using namespace varitensor;

// You can pass a user-defined function to initialise our metric
MetricFunction my_unusual_metric = [](int i, int j) -> double {
    if (i==j) return 3;
    return i - j;
}

int main() {
    Index i{"i", LATIN}, j{"j", LATIN}, k{"k", LATIN};

    Tensor T{i, j};
    Tensor g = metric_tensor({
            {i, CONTRAVARIANT},
            {k, CONTRAVARIANT}
        },
        my_unusual_metric // The default is varitensor::EUCLIDEAN_METRIC
    );

    // Note that unlike normal multiplication, k will positionally replace i in U
    Tensor U = T[i, j] * g[i, k];
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
<span style="font-size:10px; color:#666666; text-align:right">
Copyright &copy; Ben Stokes, 2025
</span>
<br/>
