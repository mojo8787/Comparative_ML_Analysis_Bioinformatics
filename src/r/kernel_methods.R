library(kernlab)

# Custom kernel function
rbf_kernel <- function(x, y, sigma = 1) {
    exp(-sigma * sum((x - y)^2))
}

polynomial_kernel <- function(x, y, degree = 3, scale = 1, offset = 1) {
    (scale * sum(x * y) + offset)^degree
}

# Example usage with kernlab
svm_with_custom_kernel <- function(X, y, kernel_func, ...) {
    custom_kernel <- function(x, y) kernel_func(x, y, ...)
    model <- ksvm(x = X, y = y, kernel = custom_kernel)
    return(model)
}
