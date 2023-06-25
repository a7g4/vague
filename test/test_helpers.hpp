#include "Eigen/Core"
#include "catch.hpp"

// TODO: Rewrite this all with macros so that it gets the right call site in the error messages
template <typename M1, typename M2>
// NOLINTNEXTLINE(readability-identifier-naming)
void TEST_MATRIX_NEARLY_EQUAL(const M1& m1,
                              const M2& m2,
                              const typename M1::Scalar& epsilon,
                              const Catch::ResultDisposition::Flags& result_disposition) {
    INTERNAL_CATCH_TEST("CHECK", result_disposition, std::is_same<typename M1::Scalar, typename M2::Scalar>::value);
    INTERNAL_CATCH_TEST("CHECK", result_disposition, m1.rows() == m2.rows());
    INTERNAL_CATCH_TEST("CHECK", result_disposition, m1.cols() == m2.cols());

    INTERNAL_CATCH_TEST("CHECK", result_disposition, (m1 - m2).cwiseAbs().maxCoeff() < epsilon);
}

template <typename M1, typename M2>
// NOLINTNEXTLINE(readability-identifier-naming)
void CHECK_MATRIX_NEARLY_EQUAL(const M1& m1,
                               const M2& m2,
                               const typename M1::Scalar& epsilon = std::numeric_limits<typename M1::Scalar>::epsilon()) {
    TEST_MATRIX_NEARLY_EQUAL(m1, m2, epsilon, Catch::ResultDisposition::ContinueOnFailure);
}