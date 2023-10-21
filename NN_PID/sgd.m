%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Yassine Kebbati
% Date: 20/12/2019
% Control NN-PID-Autonomous_Driving
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [weights_input_hidden, weights_hidden_output]=sgd(weights_input_hidden,weights_hidden_output,X,Y, learnrate)

hidden_layer_in = weights_input_hidden'*X;
hidden_layer_out = sig(hidden_layer_in);

output_layer_in = weights_hidden_output'*hidden_layer_out;
output = output_layer_in; %sig(output_layer_in);

error = Y - output;
output_error_term = error; %.* output .* (1 - output);

hidden_error_term = (weights_hidden_output * output_error_term) .* (hidden_layer_out.*(1 - hidden_layer_out));

delta_w_h_o = learnrate*output_error_term' .* hidden_layer_out;

delta_w_i_h = learnrate * hidden_error_term' .* X;


weights_input_hidden =  weights_input_hidden + delta_w_i_h;   
weights_hidden_output =  weights_hidden_output + delta_w_h_o;
