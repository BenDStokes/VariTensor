/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * Licence, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef VARITENSOR_PRETTY_PRINT_H
#define VARITENSOR_PRETTY_PRINT_H

#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>

#include "Tensor.h"

namespace varitensor {

namespace impl {

inline size_t print_data_width = 5;
inline int print_data_precision = 1;

// every entry in this string must be EXACTLY 5 lines; the name and width of each
// letter must be recorded in the subsequent letter_widths map
const auto letters = R"(
  __ ,
/ _  |
\__, |
\___/

 ___
/ __)
> _)
\___)
 / ,__|
 \ \
 ,  \
| () |
 \__/
,_______
\  ,---.|
 | |   L|
 | |
 |_|





,-,
|/







 ____



 ----

   _
 _| |_
|_   _|
  |_|

\|/
/|\



_.
\ \
 \ \
  \ \
   \_\
    ,_,
   / /
  / /
 / /
/_/

 _____
|_____|
 _____
|_____|
(_) ,_,
   / /
  / /
 / / _
/_/ ( )

    _
  /  \
 / /\ \
/_/  \_\



 _
'_'



,_,
|/
  /
 |
 |
 |
  \
\
 |
 |
 |
/
 ,-
 |
 |
 |
 '_
-.
 |
 |
 |
_'
 ,-
 |
<
 |
 '_
-.
 |
  >
 |
_'
    _
   / \
  / _ \
 / ___ \
/_/   \_\
 ____
|  _ )
|  _ \
| |_) |
|____/
  ____
 / ___|
| |
| |___
 \____|
 ____
|  _ \
| | | |
| |_| |
|____/
 _____
| ____|
|  _|
| |___
|_____|
 _____
|  ___|
| |_
|  _|
|_|
  ____
 / ___|
| |  _
| |_| |
 \____|
 _   _
| |_| |
|  _  |
| | | |
|_| |_|
 ___
|_ _|
 | |
 | |
|___|
   ___
  |__ |
 _  | |
| |_| |
 \___/
 _  __
| |/ /
| ' /
| . \
|_|\_\
 _
| |
| |
| |___
|_____|
 __  __
|  \/  |
| |\/| |
| |  | |
|_|  |_|
 _   _
| \ | |
|  \| |
| |\  |
|_| \_|
  ___
 / _ \
| | | |
| |_| |
 \___/
 ____
|  _ \
| |_) |
|  __/
|_|
  ___
 / _ \
| | | |
| |_| |
 \__\_\
 ____
|  _ \
| |_) |
|  _ <
|_| \_\
 ____
/ ___|
\___ \
 ___) |
|____/
 _____
|_   _|
  | |
  | |
  |_|
 _   _
| | | |
| | | |
| |_| |
 \___/
 _     _
| |   | |
 \ \ / /
  \ V /
   \_/
 _      _
| |    | |
| \ /\ / |
 \ V  V /
  \_/\_/
__  __
\ \/ /
 \  /
 /  \
/_/\_\
__   __
\ \ / /
 \ V /
  | |
  |_|
_____
|__  /
  / /
 / /_
/____|


 __,
/  |
\__|,
 |
 |
 |__
 |  \
,|__/


 ,_
/
\__,
   |
   |
 __|
/  |
\__|,


 __
/__|
\__
  _
 / `
-|-
 |
 |


,__,
\__|
___/
|
|
| _
|/ |
|  |

 .
 _
 |
 \_
  .
  _
  |
  |
\_/
|
|__
|  \
|__/
|  \
 |
 |
 |
 |
 \___


, _  _
|/ |/ |
|  |  |


, _
|/ |
|  |


 __
/  \
\__/


,__,
|__/
|


,__,
\__|
   |/


. ,-
 |
 |


 __
|__,
.__|

_|_
 |
 |
 \_


_   _
|   |
|__/|,


.    ,
 \  /
  \/


.        ,
 \  /\  /
  \/  \/


\ /
 x
/ \


,   _
\__/|
 ___/


___
 \
__\
)";

// controls the width of each letter when printed; this is necessary to
// make every line of each letter the same width to prevent misalignment
const std::vector<std::pair<int, int>> letter_widths = {
    {METRIC_TENSOR, 6},
    {LEVI_CIVITA_SYMBOL, 5},
    {KRONECKER_DELTA, 7},
    {CHRISTOFFEL_SYMBOL, 9},
    {' ', 3},
    {'\'', 3},
    {'_', 6},
    {'-', 6},
    {'+', 7},
    {'*', 3},
    {'\\', 6},
    {'/', 6},
    {'=', 7},
    {'%', 7},
    {'^', 8},
    {'.', 3},
    {',', 3},
    {'(', 3},
    {')', 3},
    {'[', 3},
    {']', 3},
    {'{', 3},
    {'}', 3},
    {'A', 9},
    {'B', 7},
    {'C', 7},
    {'D', 7},
    {'E', 7},
    {'F', 7},
    {'G', 7},
    {'H', 7},
    {'I', 5},
    {'J', 7},
    {'K', 6},
    {'L', 7},
    {'M', 8},
    {'N', 7},
    {'O', 7},
    {'P', 7},
    {'Q', 7},
    {'R', 7},
    {'S', 7},
    {'T', 7},
    {'U', 7},
    {'V', 9},
    {'W', 10},
    {'X', 6},
    {'Y', 7},
    {'Z', 6},
    {'a', 5},
    {'b', 5},
    {'c', 4},
    {'d', 5},
    {'e', 4},
    {'f', 4},
    {'g', 4},
    {'h', 4},
    {'i', 4},
    {'j', 4},
    {'k', 4},
    {'l', 5},
    {'m', 7},
    {'n', 4},
    {'o', 4},
    {'p', 4},
    {'q', 5},
    {'r', 4},
    {'s', 4},
    {'t', 4},
    {'u', 7},
    {'v', 6},
    {'w', 10},
    {'x', 3},
    {'y', 5},
    {'z', 4}
};

inline void add_letter(std::vector<std::string>& lines, const int letter) {
    auto* letter_ptr = letters;

    for (auto [cur_letter, width]: letter_widths) {
        if (cur_letter == letter) {
            // we found the letter we need, so output it and return
            for (auto& line: lines) {
                for (int i = 0; i < width; ++i) {
                    if (*letter_ptr != '\n') {
                        line += *letter_ptr;
                        ++letter_ptr;
                    }
                    else line += ' ';
                }
                ++letter_ptr;
            }
            return;
        }

        // step 5 lines to get to the next letter
        int step = 5;
        while (step) {
            if (*letter_ptr == '\n') --step;
            ++letter_ptr;
        }
    }
    // if we didn't find the letter, just skip
}

inline void add_header_name(
    const std::string& name,
    std::vector<std::string>& head_lines,
    const TensorClass tensor_class
) {
    if (tensor_class != TENSOR) { // some tensors have special names, e.g. the kronecker delta
        add_letter(head_lines, tensor_class);
    }
    else {
        for (const auto letter: name) {
            add_letter(head_lines, letter);
        }
    }
}

inline void add_header_indices(const Dimensions& dimensions, std::vector<std::string>& header) {
    for (auto& dimension: dimensions) {
        std::string blank;
        for(size_t i=0; i<dimension.index.name().size(); ++i) blank += " ";

        if (dimension.variance == CONTRAVARIANT){ // Upper index
            header[1] += dimension.index.name() + " ";
            header[3] += blank + " ";
            header[4] += blank + " ";
        }
        else { // Lower index
            header[1] += blank + " ";
            header[3] += blank + " ";
            header[4] += dimension.index.name() + " ";
        }
    }
}

inline void write_header(
    std::ostream& stream,
    const Dimensions& dimensions,
    const std::string& name,
    const TensorClass tensor_type
) {
    /** Outputs the "title" of the tensor in fancy lettering with the indices in their correct positions
     * e.g.   ______
     *       |__  __|
     *         | |    mu
     *         | |          =
     *         |_|      nu
     */

    auto add_space = [](std::vector<std::string>& header) {
        for(auto& line: header) line += " ";
    };

    auto add_equals_sign = [&](std::vector<std::string>& header) {
        if (tensor_type == METRIC_TENSOR) header[3] += "  ";
        else header[3] += " =";
    };

    auto output_header = [&](const std::vector<std::string>& header) {
        for (int i = 0; i < 5; ++i) stream << header[i] << "\n";
    };

    std::vector<std::string> header{5};

    add_header_name(name, header, tensor_type);
    add_space(header);
    add_header_indices(dimensions, header);
    add_equals_sign(header);
    output_header(header);
}

inline void write_subtitle(std::ostream& stream, const Tensor& tensor, const TensorClass print_type) {
    /** writes a small note of what "sort" of tensor we have
     * e.g. "Rank 5 Tensor", "Column Vector", etc.
     */
    if (print_type == METRIC_TENSOR) stream << "Metric Tensor\n\n";
    else if (print_type == LEVI_CIVITA_SYMBOL) stream << "Levi-Civita Symbol\n\n";
    else if (print_type == KRONECKER_DELTA) stream << "Kronecker Delta\n\n";
    else {
        if (const auto rank = tensor.rank(); rank == 0) stream << "Scalar Tensor\n\n";
        else if (rank == 1) {
            if (tensor.variance(0) == CONTRAVARIANT) stream << "Column";
            else stream << "Row";
            stream << " Vector\n\n";
        }
        else if (rank == 2) stream << " Matrix Tensor\n\n";
        else stream << "Rank " << rank << " Tensor\n\n";
    }
}

inline auto format_value(const double value) -> std::string {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(print_data_precision) << value;
    auto result = stream.str();

    // pad any strings shorter than print_data_width
    if (result.size() < print_data_width) {
        result.insert(0, std::string(print_data_width - result.size(), ' '));
    }

    // truncate any strings longer than print_data_width and add the truncation character (~)
    else if (result.size() > print_data_width) {
        result = result.substr(0, print_data_width -1) + "~"; // -1 to accommodate ~
    }

    return result;
}

inline void output_row(
    std::ostream& stream,
    const double* data,
    const size_t size
) {
    // for scalar and row vector
    stream << "< ";
    for (size_t i=0; i<size; ++i) stream << format_value(data[i]) << " ";
    stream << ">" << "\n";
}

inline void output_column(
    std::ostream& stream,
    const double* data,
    const size_t size
) {
    // for scalar and row vector
    stream << " /" << format_value(data[0]) << "\\\n" ;
    for (size_t i=1; i<size-1; ++i) stream << "| " << format_value(data[i]) << " |\n";
    stream << " \\" << format_value(data[size-1]) << "/\n" ;
}

inline void output_grid(
    const std::vector<std::vector<std::string>>& value_matrix,
    std::ostream& stream,
    const int total_width,
    const int total_height,
    const int i_size,
    const int k_size
) {
    std::vector<std::stringstream> output_rows(total_height + 2); // +2 for the bracket heads at the top and bottom
    const int n_whitespace = i_size * k_size + k_size; // spaces between characters + spaces between slices
    const int total_fields_length = total_width * static_cast<int>(print_data_width);
    const int total_matrix_width = total_fields_length + n_whitespace - 1;

    // add the starting bracket
    output_rows[0] << " /";
    for (int ii=1; ii<total_height+1; ++ii) output_rows[ii] << "|";
    output_rows[total_height+1] << " \\";

    // add the value data in between
    output_rows[0] << std::string(total_matrix_width, ' ');
    for (size_t ii=1; ii<output_rows.size()-1; ++ii) {
        for (int jj=0; jj<total_width; ++jj) {
            if (jj % i_size == 0) output_rows[ii] << " "; // adds an extra space before every slice
            output_rows[ii] << value_matrix[ii - 1][jj] << " "; // note the space between every field
        }
    }
    output_rows[total_height+1] << std::string(total_matrix_width, ' ');

    // add the ending bracket
    output_rows[0] << "\\\n";
    for (int ii=1; ii<total_height+1; ++ii) output_rows[ii] << " |\n";
    output_rows[total_height+1] << "/\n";

    // finally, output all the rows
    for (auto& row: output_rows) stream << row.str();
}

inline void output_234(const Tensor& tensor, const Dimensions& dimensions, std::ostream& stream) {
    // for tensors of rank 2, 3, or 4 (by far the most complicated case)

    const int i_size = dimensions[0].index.size();
    const int j_size = dimensions[1].index.size();
    const int k_size = dimensions.size() >= 3 ? dimensions[2].index.size() : 1;

    const int n_major_row = dimensions.size() == 4 ? dimensions[3].size() : 1; // number of big grids to output
    const int n_squares = dimensions.size() >= 3 ? dimensions[2].size() : 1; // number of squares in each grid

    // create a set of indices to slice each square out of the tensor
    Indexables indices;
    switch (dimensions.size()) {
        default:
            indices.insert(indices.begin(), -1);
            [[fallthrough]];
        case 3:
            indices.insert(indices.begin(), -1);
            [[fallthrough]];
        case 2:
            indices.insert(indices.begin(), dimensions[1].index);
            indices.insert(indices.begin(), dimensions[0].index);
            break;
    }

    // make a string matrix to store the values in
    // storing in height-by-width form makes the output stage easier
    const int total_width = i_size * n_squares;
    const int total_height = j_size + n_squares - 1;
    std::vector value_matrix(
        total_height,
        std::vector(total_width, std::string(print_data_width, ' '))
    );

    for (int i=0; i<n_major_row; ++i) {
        // for each major row
        if (dimensions.size() == 4) indices[3] = i;

        int start_height = total_height - j_size;
        int start_width = 0;
        for (int j=0; j<n_squares; ++j) { // for each square in the major row
            if (dimensions.size() >= 3) indices[2] = j;

            // slice the tensor and fill in the slice's values starting at (start_height, start_width)
            Tensor square = tensor[indices];
            for (int jj=0; jj<j_size; ++jj) {
                for (int ii=0; ii<i_size; ++ii) {
                    value_matrix[start_height+jj][start_width + ii] = format_value(square[ii, jj]);
                }
            }

            // move the starting position to the next square
            start_width += i_size;
            --start_height;
        }

        output_grid(value_matrix, stream, total_width, total_height, i_size, k_size);
    }
}

} // namespace impl

inline void set_print_data_width(const int width) {
    // sets the width of the data elements when pretty printing
    impl::print_data_width = width;
}

inline void set_print_precision(const int precision) {
    // set the precision of the output stream when pretty printing
    impl::print_data_precision = precision;
}

inline void write_data(std::ostream& stream, const Tensor& tensor) {
    // outputs just the tensor data in pretty format
    if(tensor.m_dimensions.empty()) impl::output_row(stream, tensor.m_data, tensor.m_size);
    else if(tensor.m_dimensions.size() == 1) {
        if (tensor.variance(0) == CONTRAVARIANT) impl::output_column(stream, tensor.m_data, tensor.m_size);
        else impl::output_row(stream, tensor.m_data, tensor.m_size);
    }
    else if (tensor.m_dimensions.size() <= 4) impl::output_234(tensor, tensor.m_dimensions, stream);
    else tensor.dump(stream);
}

inline std::ostream& pretty_print(std::ostream& ostream, const Tensor& tensor) {
    // display the tensor in detailed geometric format, including name and index information
    impl::write_header(ostream, tensor.m_dimensions, tensor.name(), tensor.m_tensor_class);
    impl::write_subtitle(ostream, tensor, tensor.m_tensor_class);
    write_data(ostream, tensor);
    return ostream;
}

} // namespace varitensor

inline std::ostream& operator<<(std::ostream& stream, const varitensor::Tensor& tensor) {
    return varitensor::pretty_print(stream, tensor);
}

#endif
