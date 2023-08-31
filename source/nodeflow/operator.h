#pragma once

#include <nodeflow/node.h>
#include <cmath>

/*
 * Below is a template for create a custom operator node
 */

/*
class FunctionName: public OperatorNode<[number of input]>{
    //default constructor (check Pow for custom constructor)
    using OperatorNode<[number of input]>::OperatorNode;

    void compute() override {
        //Input Node(s) can be access by this->getInput(INPUT_INDEX)
        //      or this->getInput() for a signal input op 
        //      (too lazy to use 0 index)
        //Result from the calculation MUST be assigned to this->value
    }

    Eigen::MatrixXd derivative(size_t input_wrt_index) override {
        //MUST: return a matrix as the result to the taking the partial derivative on the function with respect to this->inputs[input_wrt_index]
        //MUST: use this->outer_derivative according to chain rule
        return ;
    }
};
*/

namespace nodeflow{

class Add: public OperatorNode<2>{
    using OperatorNode<2>::OperatorNode;

    void compute() override {
        this->value =
            this->getInput(0)
            +
            this->getInput(1)
        ;
    }
    Eigen::MatrixXd derivative(size_t input_wrt_index) override {
        return this->outer_derivative;
    }
};

class Mul: public OperatorNode<2>{
    using OperatorNode<2>::OperatorNode;

    void compute() override {
        this->value =
            this->getInput(0)
            *
            this->getInput(1)
        ;
    }
    Eigen::MatrixXd derivative(size_t input_wrt_index) override{
        if(input_wrt_index){ //second input
            return 
                this->getInput(0).transpose() 
                * 
                this->outer_derivative;
        }else{ // first input
            return 
                this->outer_derivative
                *
                this->getInput(1).transpose();
        }
    }
};

class Pow: public OperatorNode<1>{
    private:
        double exponent;
    public:
        Pow(std::initializer_list<Node*> input_list, double exponent)
        :OperatorNode<1>(input_list), exponent(exponent) { }

    void compute() override{
        this->value = this->getInput().array().pow(this->exponent);
    }
    Eigen::MatrixXd derivative(size_t input_wrt_index) override {
        return this->getInput()
            .array()
            .pow(this->exponent - 1) 
            * this->exponent
            * this->outer_derivative.array()
        ;
    }
};

class Sqrt:public OperatorNode<1>{
    using OperatorNode<1>::OperatorNode;

    void compute() override{
        this->value = this->getInput().array().sqrt();
    }
    Eigen::MatrixXd derivative(size_t input_wrt_index) override {
        return 
            (
                1 / (2 * this->getInput().array().sqrt())
            ) * this->outer_derivative.array() 
        ;
    }
};

class Invert: public OperatorNode<1>{
    using OperatorNode<1>::OperatorNode;

    void compute() override {
        this->value = - this->getInput();
    }

    Eigen::MatrixXd derivative(size_t input_wrt_index) override {
        return -this->outer_derivative;
    }
};

class Subtract: public OperatorNode<2>{
    using OperatorNode<2>::OperatorNode;

    void compute() override {
        this->value = this->getInput(0) - this->getInput(1);
    }

    Eigen::MatrixXd derivative(size_t input_wrt_index) override {
        if(input_wrt_index){ // Second input
            return -this->outer_derivative;
        }else{ // First input
            return this->outer_derivative;
        }
    }
};

class Inverse: public OperatorNode<1>{
    using OperatorNode<1>::OperatorNode;

    void compute() override {
        this->value = this->getInput().array().inverse();
    }
    Eigen::MatrixXd derivative(size_t input_wrt_index) override {
        return -this->getInput().array().pow(2).inverse()
            *
            this->outer_derivative.array();
    }
};

class Sin: public OperatorNode<1>{
    using OperatorNode<1>::OperatorNode;

    void compute() override {
        this->value = this->getInput().array().sin();
    }
    Eigen::MatrixXd derivative(size_t input_wrt_index) override {
        return this->getInput().array().cos()
            *
            this->outer_derivative.array();
    }
};

class Cos: public OperatorNode<1>{
    using OperatorNode<1>::OperatorNode;

    void compute() override {
        this->value = this->getInput().array().cos();
    }
    Eigen::MatrixXd derivative(size_t input_wrt_index) override {
        return -this->getInput().array().sin()
            *
            this->outer_derivative.array();
    }
};

class Tan: public OperatorNode<1>{
    using OperatorNode<1>::OperatorNode;

    void compute() override {
        this->value = this->getInput().array().tan();
    }
    Eigen::MatrixXd derivative(size_t input_wrt_index) override {
        return this->getInput().array().cos().pow(2).inverse()
            *
            this->outer_derivative.array();
    }
};

class Sinh: public OperatorNode<1>{
    using OperatorNode<1>::OperatorNode;

    void compute() override {
        this->value = this->getInput().array().sinh();
    }
    Eigen::MatrixXd derivative(size_t input_wrt_index) override {
        return this->getInput().array().cosh()
            *
            this->outer_derivative.array();
    }
};

class Cosh: public OperatorNode<1>{
    using OperatorNode<1>::OperatorNode;

    void compute() override {
        this->value = this->getInput().array().cosh();
    }
    Eigen::MatrixXd derivative(size_t input_wrt_index) override {
        return this->getInput().array().sinh() 
            *
            this->outer_derivative.array();
    }
};

class Tanh: public OperatorNode<1>{
    using OperatorNode<1>::OperatorNode;

    void compute() override {
        this->value = this->getInput().array().tanh();
    }
    Eigen::MatrixXd derivative(size_t input_wrt_index) override {
        return this->getInput().array().cosh().pow(2).inverse() 
            * 
            this->outer_derivative.array();
    }
};

class Exp: public OperatorNode<1>{
    using OperatorNode<1>::OperatorNode;

    void compute() override {
        this->value = this->getInput().array().exp();
    }
    Eigen::MatrixXd derivative(size_t input_wrt_index) override {
        return this->getInput().array().exp() 
            * 
            this->outer_derivative.array();
    }
};

class Loge: public OperatorNode<1>{
    using OperatorNode<1>::OperatorNode;

    void compute() override {
        this->value = this->getInput().array().log();
    }
    Eigen::MatrixXd derivative(size_t input_wrt_index) override {
        return this->getInput().array().inverse() 
            * 
            this->outer_derivative.array();
    }
};

class ReLU: public OperatorNode<1>{
    using OperatorNode<1>::OperatorNode;

    void compute() override{
        this->value = 
            this->getInput().cwiseMax(0)
        ;
    }

    Eigen::MatrixXd derivative(size_t input_wrt_index) override {
        return (this->getInput().array() > 0).cast<double>().array() 
            * 
            this->outer_derivative.array()
        ;
    }
};

class LeakyReLU: public OperatorNode<1>{
    private:
        double leak_value;
    public:
        LeakyReLU(std::initializer_list<Node*> input_list)
        :OperatorNode<1>(input_list), leak_value(0.1) { }

        LeakyReLU(std::initializer_list<Node*> input_list, double leak_value)
        :OperatorNode<1>(input_list), leak_value(leak_value) { }

    void compute() override{
        this->value =
            this->getInput().cwiseMax(this->leak_value * this->getInput())
        ;
    }
    Eigen::MatrixXd derivative(size_t input_wrt_index) override {
        size_t rows = this->getInput().rows();
        size_t cols = this->getInput().cols();

        Eigen::MatrixXd one_m = Eigen::MatrixXd::Constant(rows, cols, 1);
        Eigen::MatrixXd leak_m = Eigen::MatrixXd::Constant(rows, cols, this->leak_value);

        return (this->getInput().array() > 0).select(one_m, leak_m).array()
            *
            this->outer_derivative.array()
        ;
    }
};


class Sigmoid: public OperatorNode<1>{
    using OperatorNode<1>::OperatorNode;

    void compute() override{
        //1 / (1 + exp(-x))
        this->value = 
            (
                1+(-this->getInput().array())
                .exp()
            ).inverse()
        ;
    }

    Eigen::MatrixXd derivative(size_t input_wrt_index) override {
        Eigen::MatrixXd temp = (-this->getInput().array()).exp();
        return temp.array() * this->outer_derivative.array() 
            / 
            ( 1 + temp.array() ).pow(2);
    }
};
class Softmax: public OperatorNode<1>{
    using OperatorNode<1>::OperatorNode;

    void compute() override{
        Eigen::MatrixXd exp = this->getInput().array().exp(); 
        double sum = exp.sum();

        this->value = exp / sum;
    }

    //TODO: check if sum of row of jacobian always approach zero
    Eigen::MatrixXd derivative(size_t input_wrt_index) override {
        //(Diagonal(x * sum) - OuterProd(X, X)) / sum^2
        
        Eigen::MatrixXd exp = this->getInput().array().exp();
        double sum = exp.sum();

        // Eigen::DiagonalMatrix<double, Eigen::Dynamic> diagonal_exp_sum;
        Eigen::MatrixXd diagonal_exp_sum = Eigen
            ::MatrixXd
            ::Constant(exp.rows(), exp.rows(), 0);
        diagonal_exp_sum.diagonal() = exp * sum;

        Eigen::MatrixXd jacobian = 
            ( diagonal_exp_sum - (exp * exp.transpose()) ) 
            / 
            std::pow(sum, 2)
        ;
        return jacobian * this->outer_derivative;
    }
};

}//namespace nodeflow ----------------------------------------
