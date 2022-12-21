#ifndef _SETTINGS_H_
#define _SETTINGS_H_

// model setting
const int input_dim = 784;
int hidden_dim = 300;
const int output_dim = 10;

void set_hidden_layer_size(int new_size) { hidden_dim = new_size; }

// data type
using data_t = float;

#endif