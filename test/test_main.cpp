#include <nodeflow/node.h>
#include <nodeflow/operator.h>

#include <Eigen/Dense>

using namespace nodeflow;

/*
 * To finalize the graph created
 *      call .foward() then .finished() on the last node
 */

int main() {
    Node a = Node::Vector({0.1, -0.2, 0.1});

    LeakyReLU f({&a}, 0.1);

    f.forward();
    f.finished();
    f.print();

    f.backward();
    std::cout << a.getGrad() << std::endl;

    return 0;
}

//TODO: col Vector with initializer_list
//TODO:
// Pow ---------
// Square Root ----------
// Invert ----------
// Subtract -------
// Resieprocal -----
//
// Sin, Cos, Tan -------
//
// Exp, Ln, Log -------
//
// Max, Min
//
// PiecesWise
//
// Sigmoid------
// Tanh-------
// Softmax
