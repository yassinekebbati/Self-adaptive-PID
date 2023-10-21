%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Yassine Kebbati
% Date: 20/12/2019
% Control NN-PID-Autonomous_Driving
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [weights_input_hidden, weights_hidden_output,output]=SGD2(weights_input_hidden,weights_hidden_output,dy,In,error,lr)

hidden_layer_in = weights_input_hidden'*In;
hidden_layer_out = sig(hidden_layer_in);

% %hidden_layer_out(2) = hidden_layer_in(2)+ hidden_layer_out2;
% %hidden_layer_out(3) = hidden_layer_in(3)- hidden_layer_in3;
% 
% for i=1:3;
%     if hidden_layer_out(i) <-1
%         hidden_layer_out(i)=-1;
%     elseif hidden_layer_out(i) >1;
%         hidden_layer_out(i) =1;
%     end
% end

% hidden_layer_out2 = hidden_layer_out(2);
% hidden_layer_out3 = hidden_layer_out(3);


output_layer_in = weights_hidden_output*hidden_layer_out;
output = output_layer_in; 


output_error_term = error * dy; 

hidden_error_term = weights_hidden_output * output_error_term.* (hidden_layer_out.*(1 - hidden_layer_out))';

delta_w_h_o = -(lr*output_error_term' .* hidden_layer_out)';

delta_w_i_h = -lr * hidden_error_term .* In;


weights_input_hidden =  weights_input_hidden + delta_w_i_h;   
weights_hidden_output =  weights_hidden_output + delta_w_h_o;
