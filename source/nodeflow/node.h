#pragma once

#include <iostream>
#include <Eigen/Dense>

namespace nodeflow{

class Node{
    protected:
        bool is_differentiatable = true;
        bool is_value_ready = false;

        Eigen::MatrixXd value;
        Eigen::MatrixXd outer_derivative;

        unsigned int rows = 1;
        unsigned int cols = 1;

        int num_parent = 0;
        int parent_called_count = 0;

    public:
        Node() = default;
        Node(Eigen::MatrixXd initial_value){
            this->value = initial_value;
        }
        void operator=(Eigen::MatrixXd new_value){
            this->value = new_value;
        }

        void print(){
            std::cout << "Node: " << " (Matrix) " << "\n"
                << "----\n"
                << this->value << "\n"
                << "----\n";
        }

        Eigen::MatrixXd& operator()(){
            return this->value;
        }
        double& operator()(size_t row, size_t col){
            return this->value(row, col);
        }
        Eigen::MatrixXd& getValue(){
            return this->value;
        }
        Eigen::MatrixXd& getGrad(){
            if(!this->is_differentiatable) 
                std::cout 
                    << "Warning: Calling 'getGrad()' on a constant node" 
                    << std::endl;
            return this->outer_derivative;
        }

        void finished(){
            this->num_parent++;
            this->rows = this->value.rows();
            this->cols = this->value.cols();
        }
        void reset(){
            this->is_value_ready = false;
            this->parent_called_count = 0;
        }
        bool isDifferentiatable(){
            return this->is_differentiatable;
        }
        void constant(){
            this->is_differentiatable = false;
        }

        virtual Eigen::MatrixXd& forward(){
            return this->value;
        }
        virtual void backward(Eigen::MatrixXd partial_outer_derivative){
            if(!this->is_differentiatable) return;

            if(!this->parent_called_count) {
                this->outer_derivative = partial_outer_derivative;
            }else{
                this->outer_derivative += partial_outer_derivative;
            }

            this->parent_called_count++;
        }

        static Node Constant(size_t rows, size_t cols, double fill_value){
            Node temp = Eigen::MatrixXd::Constant(rows, cols, fill_value); 
            return temp;
        }
        static Node Random(size_t rows){
            Node temp = Eigen::MatrixXd::Random(rows, 1);
            return temp;
        }
        static Node Random(size_t rows, size_t cols){
            Node temp = Eigen::MatrixXd::Random(rows, cols);
            return temp;
        }
        static Node Scalar(double initial_value){
            Node temp = Eigen::MatrixXd::Constant(1, 1, initial_value);
            return temp;
        }
        static Node Vector(size_t rows){
            Node temp = Eigen::MatrixXd::Constant(rows, 1, 0);
            return temp;
        }
        static Node Vector(std::initializer_list<double> initial_vector){
            Node temp = Eigen::MatrixXd(initial_vector.size(), 1);
            for(int i=0; i<initial_vector.size(); i++){
                temp.getValue()(i, 0) = *(initial_vector.begin() + i);
            }
            return temp;
        }
        // static Node Matrix(size_t rows, size_t cols){
        //     Node temp = Eigen::MatrixXd::Constant(rows, cols, 0);
        //     return temp;
        // }
        // static Node Matrix(
        //     std::initializer_list<std::initializer_list<double>> initial_matrix
        // ){
        //     Node temp = Eigen::MatrixXd(initial_matrix);
        //     return temp;
        // }
};

template <unsigned int NINPUT>
class OperatorNode: public Node{
    protected:
        Node* inputs[NINPUT];

        virtual void compute() = 0;
        virtual Eigen::MatrixXd derivative(size_t input_wrt_index) = 0;

    public:
        OperatorNode() = default;
        OperatorNode(std::initializer_list<Node*> input_list){
            this->initializeInput(input_list);
        }

        void initializeInput(std::initializer_list<Node*> input_list){
            for(size_t i=0; i<NINPUT; i++){
                auto input = *(input_list.begin() + i);
                this->inputs[i] = input;
            }
        }

        Eigen::MatrixXd getInput(size_t input_index){
            return this->inputs[input_index]->getValue();
        }
        Eigen::MatrixXd getInput(){
            return this->inputs[0]->getValue();
        }
        // REASON: answer was not correct
        //      Try rerun it again
        //      Maybe because I use .constant on one of the Node
        // std::vector<Eigen::MatrixXd> getGrad(){
        //     if(!this->is_differentiatable) 
        //         std::cout 
        //             << "Warning: Calling 'getGrad()' on a constant node" 
        //             << std::endl;
        //
        //     std::vector<Eigen::MatrixXd> result;
        //     for(size_t i=0; i<NINPUT; i++){
        //         result.push_back(this->outer_derivative * this->derivative(i));
        //     }
        //     return result;
        // }

        void reset(){
            if(!this->is_value_ready) return;

            this->is_value_ready = false;
            this->parent_called_count = 0;

            for(size_t i=0; i<NINPUT; i++){
                this->inputs[i]->reset();
            }
        }

        void finished(){
            this->num_parent++;

            bool is_diff_temp = false;
            for(size_t i=0; i<NINPUT; i++){
                this->inputs[i]->finished();
                is_diff_temp = 
                    this->inputs[i]->isDifferentiatable() 
                    || 
                    is_diff_temp;
            }
            this->is_differentiatable = is_diff_temp;

            this->rows = this->value.rows();
            this->cols = this->value.cols();
        }

        Eigen::MatrixXd& forward() override{
            if(this->is_value_ready) return this->value;

            for(size_t i=0; i<NINPUT; i++){
                this->inputs[i]->forward();
            }
            this->compute();

            this->is_value_ready = true;
            return this->value;
        }

        void backward(){
            this->backward(
                Eigen::MatrixXd::Constant(
                    this->rows,
                    this->cols,
                    1.0
                )
            );
        }

        void backward(Eigen::MatrixXd partial_outer_derivative) override {
            if(!this->is_differentiatable) return;

            if(!this->parent_called_count) {
                this->outer_derivative = partial_outer_derivative;
            }else{
                this->outer_derivative += partial_outer_derivative;
            }

            this->parent_called_count++;

            if(this->parent_called_count >= this->num_parent){
                for(size_t i=0; i<NINPUT; i++){
                    Eigen::MatrixXd partial_derivative = this->derivative(i);
                    this->inputs[i]->backward(partial_derivative);
                }
            }
        }
};


}//namespace nodeflow

